from time import time

import numpy as np
import torch
import os
from isegm.inference import utils
from isegm.inference.clicker import Clicker
import shutil
import cv2
from isegm.utils.vis import add_tag



try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def evaluate_dataset(dataset, predictor, vis = True, vis_path = './experiments/vis_val/',**kwargs):
    all_ious = []
    if vis:
        save_dir =  vis_path + dataset.name + '/'
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    else:
        save_dir = None

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)

        _, sample_ious, _ = evaluate_sample(sample.image, sample.gt_mask, sample.init_mask, predictor,
                                            sample_id=index, vis= vis, save_dir = save_dir,
                                            index = index, **kwargs)
        all_ious.append(sample_ious)
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time

def Progressive_Merge(pred_mask, previous_mask, y, x):
    diff_regions = np.logical_xor(previous_mask, pred_mask)
    num, labels = cv2.connectedComponents(diff_regions.astype(np.uint8))
    label = labels[y,x]
    corr_mask = labels == label
    if previous_mask[y,x] == 1:
        progressive_mask = np.logical_and( previous_mask, np.logical_not(corr_mask))
    else:
        progressive_mask = np.logical_or( previous_mask, corr_mask)
    return progressive_mask


def evaluate_sample(image, gt_mask, init_mask, predictor, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, vis = True, save_dir = None, index = 0,  callback=None,
                    progressive_mode = True,
                    ):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    prev_mask = pred_mask
    ious_list = []

    with torch.no_grad():
        predictor.set_input_image(image)
        if init_mask is not None:
            predictor.set_prev_mask(init_mask)
            pred_mask = init_mask
            prev_mask = init_mask
            num_pm = 0
        else:
            num_pm = 999
            
        for click_indx in range(max_clicks):
            vis_pred = prev_mask
            clicker.make_next_click(pred_mask)
            pred_probs = predictor.get_prediction(clicker)
            pred_mask = pred_probs > pred_thr

            if progressive_mode:
                clicks = clicker.get_clicks()
                if len(clicks) >= num_pm: 
                    last_click = clicks[-1]
                    last_y, last_x = last_click.coords[0], last_click.coords[1]
                    pred_mask = Progressive_Merge(pred_mask, prev_mask,last_y, last_x)
                    predictor.transforms[0]._prev_probs = np.expand_dims(np.expand_dims(pred_mask,0),0)
            if callback is not None:
                callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)

            iou = utils.get_iou(gt_mask, pred_mask)
            ious_list.append(iou)
            prev_mask = pred_mask

            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                break

        if vis:
            clicks_list = clicker.get_clicks()
            last_y, last_x = predictor.last_y, predictor.last_x
            out_image = vis_result_base(image, pred_mask, gt_mask, init_mask, iou,click_indx+1,clicks_list, vis_pred, last_y, last_x)
            cv2.imwrite(save_dir+str(index)+'.png', out_image)
        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs


def vis_result_base(image, pred_mask, instances_mask, init_mask,  iou, num_clicks,  clicks_list, prev_prediction, last_y, last_x):

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    pred_mask = pred_mask.astype(np.float32)
    prev_mask = prev_prediction.astype(np.float32)
    instances_mask = instances_mask.astype(np.float32)
    image = image.astype(np.float32)

    pred_mask_3 = np.repeat(pred_mask[...,np.newaxis],3,2)
    prev_mask_3 = np.repeat(prev_mask[...,np.newaxis],3,2)
    gt_mask_3 = np.repeat( instances_mask[...,np.newaxis],3,2  )

    color_mask_gt = np.zeros_like(pred_mask_3)
    color_mask_gt[:,:,0] = instances_mask * 255

    color_mask_pred = np.zeros_like(pred_mask_3) #+ 255
    color_mask_pred[:,:,0] = pred_mask * 255

    color_mask_prev = np.zeros_like(prev_mask_3) #+ 255
    color_mask_prev[:,:,0] = prev_mask * 255


    fusion_pred = image * 0.4 + color_mask_pred * 0.6
    fusion_pred = image * (1-pred_mask_3) + fusion_pred * pred_mask_3

    fusion_prev = image * 0.4 + color_mask_prev * 0.6
    fusion_prev = image * (1-prev_mask_3) + fusion_prev * prev_mask_3


    fusion_gt = image * 0.4 + color_mask_gt * 0.6

    color_mask_init = np.zeros_like(pred_mask_3)
    if init_mask is not None:
        color_mask_init[:,:,0] = init_mask * 255

    fusion_init = image * 0.4 + color_mask_init * 0.6
    fusion_init = image * (1-color_mask_init) + fusion_init * color_mask_init


    #cv2.putText( image, 'click num: '+str(num_clicks)+ '  iou: '+ str(round(iou,3)), (50,50),
    #            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255 ), 1   )

    for i in range(len(clicks_list)):
        click_tuple =  clicks_list[i]

        if click_tuple.is_positive:
            color = (0,0,255)
        else:
            color = (0,255,0)

        coord = click_tuple.coords
        x,y = coord[1], coord[0]
        if x < 0 or y< 0:
            continue
        cv2.circle(fusion_pred,(x,y),4,color,-1)
        #cv2.putText(fusion_pred, str(i+1), (x-10, y-10),  cv2.FONT_HERSHEY_COMPLEX, 0.6 , color,1 )

    cv2.circle(fusion_pred,(last_x,last_y),2,(255,255,255),-1)
    image = add_tag(image, 'nclicks:'+str(num_clicks)+ '  iou:'+ str(round(iou,3)))
    fusion_init = add_tag(fusion_init,'init mask')
    fusion_pred = add_tag(fusion_pred,'pred')
    fusion_gt = add_tag(fusion_gt,'gt')
    fusion_prev = add_tag(fusion_prev,'prev pred')

    h,w = image.shape[0],image.shape[1]
    if h < w:
        out_image = cv2.hconcat([image.astype(np.float32),fusion_init.astype(np.float32),fusion_pred.astype(np.float32), fusion_gt.astype(np.float32),fusion_prev.astype(np.float32)])
    else:
        out_image = cv2.hconcat([image.astype(np.float32),fusion_init.astype(np.float32), fusion_pred.astype(np.float32), fusion_gt.astype(np.float32),fusion_prev.astype(np.float32)])

    return out_image

