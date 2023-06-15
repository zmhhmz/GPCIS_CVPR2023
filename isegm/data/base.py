import random
import pickle
import numpy as np
import torch
from torchvision import transforms
from .points_sampler import MultiPointSampler
from .sample import DSample
import cv2
from isegm.utils.crop_local import random_choose_target,get_bbox_from_mask,getLargestCC,expand_bbox, expand_bbox_with_bias
from skimage import morphology
# import pdb

class ISDataset(torch.utils.data.dataset.Dataset):
    def __init__(self,
                 augmentator=None,
                 points_sampler=MultiPointSampler(max_num_points=12),
                 min_object_area=0,
                 keep_background_prob=0.0,
                 with_image_info=False,
                 samples_scores_path=None,
                 samples_scores_gamma=1.0,
                 epoch_len=-1,
                 with_refiner = False):
        super(ISDataset, self).__init__()
        self.epoch_len = epoch_len
        self.augmentator = augmentator
        self.min_object_area = min_object_area
        self.keep_background_prob = keep_background_prob
        self.points_sampler = points_sampler
        self.with_image_info = with_image_info
        self.samples_precomputed_scores = self._load_samples_scores(samples_scores_path, samples_scores_gamma)
        self.to_tensor = transforms.ToTensor()
        self.with_refiner = with_refiner
        self.dataset_samples = None

    def __getitem__(self, index):
        while(1):
            try:
                if self.samples_precomputed_scores is not None:
                    index = np.random.choice(self.samples_precomputed_scores['indices'],
                                            p=self.samples_precomputed_scores['probs'])
                else:
                    if self.epoch_len > 0:
                        index = random.randrange(0, len(self.dataset_samples))

                sample = self.get_sample(index)
                sample = self.augment_sample(sample)
                sample.remove_small_objects(self.min_object_area)

                self.points_sampler.sample_object(sample)
                points = np.array(self.points_sampler.sample_points())
                mask = self.points_sampler.selected_mask
                mask = self.remove_small_regions(mask)            
                image = sample.image
                mask_area = mask[0].shape[0] * mask[0].shape[1]

                if self.with_refiner:
                    trimap = self.get_trimap(mask[0])
                    if mask[0].sum() < 3600: # 80 * 80
                        y1,x1,y2,x2 = self.sampling_roi_full_object(mask[0])
                    else:
                        if np.random.rand() < 0.4:
                            y1,x1,y2,x2 = self.sampling_roi_on_boundary(mask[0])
                        else:
                            y1,x1,y2,x2 = self.sampling_roi_full_object(mask[0])

                    roi = torch.tensor([x1, y1, x2, y2])
                    h,w = mask[0].shape[0], mask[0].shape[1]
                    image_focus = image[y1:y2,x1:x2,:]
                    image_focus = cv2.resize(image_focus, (h,w))

                    mask_255 = (mask[0] * 255).astype(np.uint8)
                    mask_focus = mask_255[y1:y2,x1:x2]
                    mask_focus = cv2.resize(mask_focus, (h,w)) > 128
                    mask_focus = np.expand_dims(mask_focus,0).astype(np.float32)

                    trimap_255 = (trimap[0] * 255).astype(np.uint8)
                    trimap_focus = trimap_255[y1:y2,x1:x2]
                    trimap_focus = cv2.resize(trimap_focus, (h,w)) > 128
                    trimap_focus = np.expand_dims(trimap_focus,0).astype(np.float32)

                    hc,wc = y2-y1, x2-x1
                    ry,rx = h/hc, w/wc
                    bias = np.array([y1,x1,0])
                    ratio = np.array([ry,rx,1])
                    points_focus = (points - bias) * ratio

                    if mask.sum() > self.min_object_area and mask.sum() < mask_area * 0.85:
                        
                        output = {
                            'images': self.to_tensor(image),
                            'points': points.astype(np.float32),
                            'instances': mask,
                            'trimap':trimap,
                            'images_focus':self.to_tensor(image_focus),
                            'instances_focus':mask_focus,
                            'trimap_focus': trimap_focus,
                            'points_focus': points_focus.astype(np.float32),
                            'rois':roi.float()
                        }

                        if self.with_image_info:
                            output['image_info'] = sample.sample_id
                        return output
                    else:
                        index = np.random.randint(len(self.dataset_samples)-1)
                else:
                    if mask.sum() > self.min_object_area and mask.sum() < mask_area * 0.85:
                        output = {
                            'images': self.to_tensor(image),
                            'points': points.astype(np.float32),
                            'instances': mask,
                        }

                        if self.with_image_info:
                            output['image_info'] = sample.sample_id
                        return output
                    else:
                        index = np.random.randint(len(self.dataset_samples)-1)
            except:
                
                index = np.random.randint(len(self.dataset_samples)-1)

    # def __getitem__(self, index):
    #     if self.samples_precomputed_scores is not None:
    #         index = np.random.choice(self.samples_precomputed_scores['indices'],
    #                                  p=self.samples_precomputed_scores['probs'])
    #     else:
    #         if self.epoch_len > 0:
    #             index = random.randrange(0, len(self.dataset_samples))

    #     sample = self.get_sample(index)
    #     sample = self.augment_sample(sample)
    #     sample.remove_small_objects(self.min_object_area)

    #     self.points_sampler.sample_object(sample)
    #     points = np.array(self.points_sampler.sample_points())
    #     mask = self.points_sampler.selected_mask
    #     output = {
    #         'images': self.to_tensor(sample.image),
    #         'points': points.astype(np.float32),
    #         'instances': mask
    #     }

    #     if self.with_image_info:
    #         output['image_info'] = sample.sample_id

    #     return output

    def remove_small_regions(self,mask):
        mask = mask[0] > 0.5
        mask = morphology.remove_small_objects(mask,min_size= 900)
        mask = np.expand_dims(mask,0).astype(np.float32)
        return mask


    def sampling_roi_full_object(self, gt_mask, min_size=32):
        max_mask = getLargestCC(gt_mask)
        y1,y2,x1,x2 = get_bbox_from_mask(max_mask)
        ratio = np.random.randint(11,17)/10
        y1,y2,x1,x2 = expand_bbox_with_bias(gt_mask,y1,y2,x1,x2,ratio,min_size,0.3)
        return y1,x1,y2,x2

    def sampling_roi_on_boundary(self,gt_mask):
        h,w = gt_mask.shape[0], gt_mask.shape[1]
        rh = np.random.randint(15,40)/10
        rw = np.random.randint(15,40)/10
        new_h,new_w = h/rh, w/rw
        crop_size = (int(new_h), int(new_w))

        alpha = gt_mask > 0.5
        alpha = alpha.astype(np.uint8)
        kernel = np.ones((5,5),np.uint8)
        dilate = cv2.dilate(alpha,kernel,iterations = 1)
        boundary = np.logical_and( dilate, np.logical_not(alpha))
        y1,x1,y2,x2 = random_choose_target(boundary,crop_size)
        return y1,x1,y2,x2


    def get_trimap(self, mask):
        h,w = mask.shape[0],mask.shape[1]
        hs,ws = h//8,w//8
        mask_255_big = (mask * 255).astype(np.uint8)
        mask_255_small = (cv2.resize(mask_255_big, (ws,hs)) > 128) * 255
        mask_resized = cv2.resize(mask_255_small.astype(np.uint8),(w,h)) > 128
        diff_mask = np.logical_xor(mask, mask_resized).astype(np.uint8)

        kernel = np.ones((3, 3), dtype=np.uint8)
        diff_mask = cv2.dilate(diff_mask, kernel, iterations=2) # 1:迭代次数，也就是执行几次膨胀操作

        diff_mask = diff_mask.astype(np.float32)
        diff_mask = np.expand_dims(diff_mask,0)
        return diff_mask
        

    def augment_sample(self, sample) -> DSample:
        valid_augmentation = False
        while not valid_augmentation:
            sample.augment(self.augmentator)
            keep_sample = (self.keep_background_prob < 0.0 or
                           random.random() < self.keep_background_prob)
            valid_augmentation = len(sample) > 0 or keep_sample

        return sample

    def get_sample(self, index) -> DSample:
        raise NotImplementedError

    def __len__(self):
        if self.epoch_len > 0:
            return self.epoch_len
        else:
            return self.get_samples_number()

    def get_samples_number(self):
        return len(self.dataset_samples)

    @staticmethod
    def _load_samples_scores(samples_scores_path, samples_scores_gamma):
        if samples_scores_path is None:
            return None

        with open(samples_scores_path, 'rb') as f:
            images_scores = pickle.load(f)

        probs = np.array([(1.0 - x[2]) ** samples_scores_gamma for x in images_scores])
        probs /= probs.sum()
        samples_scores = {
            'indices': [x[0] for x in images_scores],
            'probs': probs
        }
        print(f'Loaded {len(probs)} weights with gamma={samples_scores_gamma}')
        return samples_scores
