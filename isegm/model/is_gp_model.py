import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from isegm.model.ops import DistMaps, BatchImageNormalize
from einops import rearrange, repeat
from opt_einsum import contract
import math

class ISGPModel(nn.Module):
    def __init__(self, use_rgb_conv=False, feature_stride = 4, with_aux_output=False,
                 norm_radius=260, use_disks=False, cpu_dist_maps=False,
                 clicks_groups=None, with_prev_mask=False, use_leaky_relu=False,
                 binary_prev_mask=False, conv_extend=False, norm_layer=nn.BatchNorm2d,
                 norm_mean_std=([.485, .456, .406], [.229, .224, .225])):
        super().__init__()
        self.with_aux_output = with_aux_output
        self.clicks_groups = clicks_groups
        self.with_prev_mask = with_prev_mask
        self.binary_prev_mask = binary_prev_mask
        self.normalization = BatchImageNormalize(norm_mean_std[0], norm_mean_std[1])
        self.dist_maps = DistMaps(norm_radius=5, spatial_scale=1.0,
                                      cpu_mode=False, use_disks=True)
        

    def prepare_input(self, image):
        prev_mask = None
        if self.with_prev_mask:
            prev_mask = image[:, 3:, :, :]
            image = image[:, :3, :, :]
            if self.binary_prev_mask:
                prev_mask = (prev_mask > 0.5).float()

        image = self.normalization(image)
        return image, prev_mask

    def get_coord_features(self, image, prev_mask, points):
        coord_features = self.dist_maps(image, points)
        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)
        return coord_features

    def load_pretrained_weights(self, path_to_weights= ''):    
        state_dict = self.state_dict()
        pretrained_state_dict = torch.load(path_to_weights, map_location='cpu')['state_dict']
        ckpt_keys = set(pretrained_state_dict.keys())
        own_keys = set(state_dict.keys())
        missing_keys = own_keys - ckpt_keys
        unexpected_keys = ckpt_keys - own_keys
        print('Missing Keys: ', missing_keys)
        print('Unexpected Keys: ', unexpected_keys)
        state_dict.update(pretrained_state_dict)
        self.load_state_dict(state_dict, strict= False)
        '''
        if self.inference_mode:
            for param in self.backbone.parameters():
                param.requires_grad = False
        '''

    def get_coord_features(self, image, prev_mask, points):
        coord_features = self.dist_maps(image, points)
        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)
        return coord_features

    def prepare_points_labels(self, points,feature):
        pss = []
        label_list = []
        point_labels = torch.ones([points.size(1),1], dtype=torch.float32, device=feature.device)
        point_labels[points.size(1)//2:,:] = -1.
        for i in range(points.size(0)):
            ps, _ = torch.split(points[i],[2,1],dim=1)
            valid_points = torch.logical_and(torch.logical_and(torch.min(ps, dim=1, keepdim=False)[0] >= 0,
                                             ps[:,0] < feature.size(2)), ps[:,1] < feature.size(3) )
            ps = ps[valid_points] # n, 2
            pss.append(ps)
            label = point_labels[valid_points,:] #n,1
            label_list.append(label)
        return pss, label_list

    def Pathwise_GP_prior(self, feature, omega):
        b,d,h,w = feature.size()
        phi_f = math.sqrt(2/self.L)*torch.sin(rearrange(self.theta(rearrange(feature, 'b d h w -> (b h w) d')), '(b h w) d->b d h w',b=b,h=h,w=w))
        prior = contract('blhw,ls->bshw',phi_f,omega)  # b,1,h,w
        return prior

    def Pathwise_GP_update(self, points, feature,pss,label_list,result,omega):
        b,d,h,w = feature.size()
        inv_Kmm_list = []
        zf_list = []
        point_nums = []
        weight = F.softplus(self.weights)

        for i in range(points.size(0)):
            ps = pss[i]
            if ps.size(0)==0:
                point_nums.append(0)
                continue
            ps = torch.cat([ps[:,[0]].clamp(min=0., max=feature.size(2)-1),ps[:,[1]].clamp(min=0., max=feature.size(3)-1)],1)

            point_nums.append(ps.size(0))
            zf = feature[i,:,ps[:,0].long(),ps[:,1].long()].T #n,d
            zf_list.append(zf)
            norm = torch.norm(torch.exp(self.logsigma2/2)*zf[:,:-3], dim=1,p=2)**2/2 # n,
            Kmm = torch.exp(contract('nd,md,d->nm',zf[:,:-3],zf[:,:-3],torch.exp(self.logsigma2))-\
                  norm.unsqueeze(0).repeat(ps.size(0),1)-norm.unsqueeze(1).repeat(1,ps.size(0)))+\
                  weight*torch.exp(-torch.sum((zf[:,-3:].unsqueeze(1)-zf[:,-3:])**2,2)/2)
            
            inv_Kmm_list.append(torch.inverse(Kmm+self.eps2*torch.eye(Kmm.size(0),device=Kmm.device)))

        inv_Kmm = torch.block_diag(*inv_Kmm_list) #n,n
        zf = torch.cat(zf_list,dim=0) # n,d
        label = torch.cat(label_list,dim=0) # n,1
        m = F.softplus(self.u_mlp(zf))*label #n,1  
        
        if self.training:
            u = m + 0.01*torch.randn(m.size()).to(feature.device)
            u_loss = self.u_loss(inv_Kmm.detach(),m,u,label/2+0.5)
        else:
            u = m
            u_loss = torch.tensor([0.],device=feature.device)
            
        phi = math.sqrt(2/self.L)*torch.sin(self.theta(zf))

        phi_omega = torch.matmul(phi,omega) # n,1

        v = torch.matmul(inv_Kmm, u-phi_omega) # n,1
        num_prev = 0
        offset = 0
        weight = F.softplus(self.weights)
        for i in range(points.size(0)):
            if point_nums[i]==0:
                offset+=1
                continue
            norm1 = torch.norm(torch.exp(self.logsigma2/2).view(self.feature_dim,1,1)*feature[i,:-3], dim=0,p=2)**2/2 #h w
            norm2 = torch.norm(torch.exp(self.logsigma2/2)*zf_list[i-offset][:,:-3], dim=1,p=2)**2/2 # n,
            norm_rgb1 = torch.norm(feature[i,-3:], dim=0,p=2)**2/2 #h w
            norm_rgb2 = torch.norm(zf_list[i-offset][:,-3:], dim=1,p=2)**2/2 # n,
            Knm = torch.exp(contract('dhw,nd,d->nhw',feature[i,:-3],zf_list[i-offset][:,:-3],torch.exp(self.logsigma2)) -\
            repeat(norm1, 'h w -> n h w',n=point_nums[i]) - repeat(norm2, 'n -> n h w',h=h, w=w)) + \
            weight*torch.exp(contract('dhw,nd->nhw',feature[i,-3:],zf_list[i-offset][:,-3:])-\
                repeat(norm_rgb1, 'h w -> n h w',n=point_nums[i]) - repeat(norm_rgb2, 'n -> n h w',h=h, w=w))
            result[i,...] += contract('nhw,ns->shw',Knm, v[num_prev:num_prev+point_nums[i]])
            num_prev += point_nums[i]
        return result, 0.001*u_loss

    def u_loss(self,invK, m, u, y):
        n = invK.size(0)
        loss = F.binary_cross_entropy_with_logits(u,y)+ torch.matmul(torch.matmul(m.T,invK),m)/n
        return loss

def split_points_by_order(tpoints: torch.Tensor, groups):
    points = tpoints.cpu().numpy()
    num_groups = len(groups)
    bs = points.shape[0]
    num_points = points.shape[1] // 2

    groups = [x if x > 0 else num_points for x in groups]
    group_points = [np.full((bs, 2 * x, 3), -1, dtype=np.float32)
                    for x in groups]

    last_point_indx_group = np.zeros((bs, num_groups, 2), dtype=np.int)
    for group_indx, group_size in enumerate(groups):
        last_point_indx_group[:, group_indx, 1] = group_size

    for bindx in range(bs):
        for pindx in range(2 * num_points):
            point = points[bindx, pindx, :]
            group_id = int(point[2])
            if group_id < 0:
                continue

            is_negative = int(pindx >= num_points)
            if group_id >= num_groups or (group_id == 0 and is_negative):  # disable negative first click
                group_id = num_groups - 1

            new_point_indx = last_point_indx_group[bindx, group_id, is_negative]
            last_point_indx_group[bindx, group_id, is_negative] += 1

            group_points[group_id][bindx, new_point_indx, :] = point

    group_points = [torch.tensor(x, dtype=tpoints.dtype, device=tpoints.device)
                    for x in group_points]

    return group_points
