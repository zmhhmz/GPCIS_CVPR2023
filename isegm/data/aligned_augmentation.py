import cv2
import numpy as np
from scipy.stats import truncnorm

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

#X1 = get_truncated_normal(mean=0.7, sd=0.3, low=0.2, upp=1)
#x1 = X1.rvs(1)[0]

        
class AlignedAugmentator:
    def __init__(self, ratio = [0.3,1], target_size = (256,256), flip = True, 
                    distribution = 'Uniform', gs_center = 0.8, gs_sd = 0.4,
                    color_augmentator = None):
        '''
        distribution belongs to [ 'Uniform, Gaussian' ]
        '''
        self.ratio = ratio
        self.target_size = target_size
        self.flip = flip
        self.distribution = distribution
        self.gaussian = get_truncated_normal(mean=gs_center, sd=gs_sd, low=ratio[0], upp=ratio[1])
        self.color_augmentator = color_augmentator

    def __call__(self, image, mask):
        '''
        image: np.array (267, 400, 3) np.uint8
        mask:  np.array (267, 400, 1) np.int32
        '''
       
        if self.distribution == 'Uniform':
            hr,wr = np.random.uniform(*self.ratio),np.random.uniform(*self.ratio)
        elif self.distribution == 'Gaussian':
            hr,wr = self.gaussian.rvs(2)

        H,W = image.shape[0], image.shape[1]
        h,w = int(H*hr), int(W*wr)
        if hr > 1 or wr > 1:
            image, mask = self.pad_image_mask(image, mask, hr, wr)
            H,W = image.shape[0], image.shape[1]
        
        y1 = np.random.randint(0,H-h)
        x1 = np.random.randint(0,W-w)
        y2 = y1 + h
        x2 = y1 + W
        
        image_crop = image[y1:y2,x1:x2,:]
        image_crop = cv2.resize(image_crop, tuple(self.target_size))
        mask_crop = mask[y1:y2,x1:x2,:].astype(np.uint8)
        mask_crop = (cv2.resize(mask_crop, tuple(self.target_size))).astype(np.int32)
        if len(mask_crop.shape) == 2:
            mask_crop = np.expand_dims(mask_crop,-1)

        if self.flip:
            if np.random.rand() < 0.3:
                image_crop = np.flip(image_crop,0)
                mask_crop = np.flip(mask_crop,0)
            if np.random.rand() < 0.3:
                image_crop = np.flip(image_crop,1)
                mask_crop = np.flip(mask_crop,1)

        image_crop = np.ascontiguousarray(image_crop)
        mask_crop = np.ascontiguousarray(mask_crop)

        if self.color_augmentator is not None:
            image_crop = self.color_augmentator(image=image_crop)['image']
        
        aug_output = {}
        aug_output['image'] = image_crop
        aug_output['mask'] = mask_crop
        return aug_output

    def pad_image_mask(self, image, mask, hr, wr):
        H,W = image.shape[0], image.shape[1]
        if hr > 1:
            new_h = int(H * hr) + 1
            pad_h = new_h - H
            pad_h1 = np.random.randint(0,pad_h)
            pad_h2 = pad_h - pad_h1
            image = np.pad(image, ((pad_h1, pad_h2),(0,0),(0,0)), 'constant')
            mask = np.pad(mask, ((pad_h1, pad_h2),(0,0),(0,0)), 'constant')
        
        if wr > 1:
            new_w = int(W * wr) + 1
            pad_w = new_w - W
            pad_w1 = np.random.randint(0,pad_w)
            pad_w2 = pad_w - pad_w1
            image = np.pad(image, ((0,0), (pad_w1, pad_w2),(0,0)), 'constant')
            mask = np.pad(mask, ( (0,0), (pad_w1, pad_w2),(0,0)), 'constant')
        return image, mask
        







    