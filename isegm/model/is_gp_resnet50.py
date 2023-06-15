import torch.nn as nn
import torch
import torch.nn.functional as F
from isegm.utils.serialization import serialize
from .is_gp_model import ISGPModel
from isegm.model.ops import ScaleLayer
from .modeling.deeplab_v3_gp import DeepLabV3Plus
from isegm.model.modifiers import LRMult


class GpModel(ISGPModel):
    @serialize
    def __init__(self, backbone='resnet50', deeplab_ch=256, aspp_dropout=0.,
                 backbone_norm_layer=None, backbone_lr_mult=0.1,
                 norm_layer=nn.BatchNorm2d, weight_dir=None, **kwargs):
        super().__init__(norm_layer=norm_layer, **kwargs)

        self.model = DeepLabV3Plus(backbone=backbone, ch=deeplab_ch,
                                   project_dropout=aspp_dropout, norm_layer=norm_layer,
                                   backbone_norm_layer=backbone_norm_layer, weight_dir=weight_dir)

        side_feature_ch = 256

        self.model.apply(LRMult(backbone_lr_mult))


        mt_layers = [
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Conv2d(in_channels=16, out_channels=side_feature_ch, kernel_size=3, stride=1, padding=1),
                ScaleLayer(init_value=0.05, lr_mult=1)
            ]
        self.maps_transform = nn.Sequential(*mt_layers)
        self.L=256
        self.feature_dim = 48
        self.theta = nn.Linear(self.feature_dim+3,self.L)
        omega = 0.25*torch.randn(self.L,1)
        self.omega = nn.Parameter(omega, requires_grad=True)
        omega_var = torch.tensor(0.025)
        self.omega_var = nn.Parameter(omega_var, requires_grad=True)
        
        logsigma2 = torch.ones(self.feature_dim)
        self.logsigma2 = nn.Parameter(logsigma2, requires_grad=True)
        self.u_mlp = nn.Sequential(
            nn.Linear(self.feature_dim+3,96),
            nn.ReLU(True),
            nn.Linear(96,1)
        )

        weight = torch.zeros(1)
        self.weights = nn.Parameter(weight, requires_grad=True)
        self.eps2 = 1e-2

    def set_status(self, training):
        if training:
            self.eps2=1e-2
        else:
            self.eps2=1e-7 

    def forward(self, image, points):
        image, prev_mask = self.prepare_input(image)
        coord_features = self.get_coord_features(image, prev_mask, points)
        coord_features = self.maps_transform(coord_features)

        feature = self.model(image, coord_features)
        feature = F.normalize(feature, dim=1)

        feature = nn.functional.interpolate(feature, size=image.size()[2:],
                                            mode='bilinear', align_corners=True)
        feature = torch.cat([feature, image],1)
        
        pss, label_list = self.prepare_points_labels(points,feature)
        if self.training:
            omega = self.omega+self.omega_var.clamp(min=0.01,max=0.05)*torch.randn(self.L,1).to(feature.device)
        else:
            omega = self.omega  
        prior= self.Pathwise_GP_prior(feature, omega)
        out, u_loss =self.Pathwise_GP_update(points, feature,pss,label_list,prior,omega)
        outputs = {'instances': out, 'u_loss':u_loss}
        return outputs
