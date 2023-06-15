import numpy as np
import cv2
from skimage.measure import label   

def map_point_in_bbox(y,x,y1,y2,x1,x2,crop_l):
    h,w = y2-y1, x2-x1
    ry,rx = crop_l/h, crop_l/w
    y = (y - y1) * ry
    x = (x - x1) * rx
    return y,x


 
def get_focus_cropv1(pred_mask, previous_mask, global_roi, y,x, ratio):
    pred_mask = pred_mask > 0.49
    previous_mask = previous_mask > 0.49
    ymin,ymax,xmin,xmax =  global_roi
    diff_regions = np.logical_xor(previous_mask, pred_mask)
    if previous_mask.sum() == 0:
        y1,y2,x1,x2 = get_bbox_from_mask(pred_mask)
    else:
        num, labels = cv2.connectedComponents( diff_regions.astype(np.uint8))
        label = labels[y,x]
        diff_conn_mask = labels == label    
        y1d,y2d,x1d,x2d = get_bbox_from_mask(diff_conn_mask)
        hd,wd = y2d - y1d, x2d - x1d
        
        y1p,y2p,x1p,x2p= get_bbox_from_mask(pred_mask)
        hp,wp = y2p - y1p, x2p - x1p
        
        if hd < hp/3 or wd < wp/3:
            r = 0.2
            l = max(hp,wp)
            y1,y2,x1,x2 = y - r *l, y + r * l, x - r * l, x + r * l
        else:
            y1,y2,x1,x2 = y1d,y2d,x1d,x2d
    
    y1,y2,x1,x2 = expand_bbox(pred_mask,y1,y2,x1,x2,ratio )
    y1 = max(y1,ymin)
    y2 = min(y2,ymax)
    x1 = max(x1,xmin)
    x2 = min(x2,xmax)
    return y1,y2,x1,x2


def get_focus_cropv2(pred_mask, previous_mask, global_roi, y,x, ratio):
    pred_mask = pred_mask > 0.5
    previous_mask = previous_mask > 0.5
    ymin,ymax,xmin,xmax =  global_roi
    diff_regions = np.logical_xor(previous_mask, pred_mask)
    num, labels = cv2.connectedComponents( diff_regions.astype(np.uint8))
    label = labels[y,x]
    diff_conn_mask = labels == label    
    
    y1d,y2d,x1d,x2d = get_bbox_from_mask(diff_conn_mask)
    hd,wd = y2d - y1d, x2d - x1d
        
    y1p,y2p,x1p,x2p= get_bbox_from_mask(pred_mask)
    hp,wp = y2p - y1p, x2p - x1p

    if previous_mask.sum() == 0: 
        y1,y2,x1,x2 = y1p,y2p,x1p,x2p
    else:
        if hd < hp/3 or wd < wp/3:
            r = 0.16
            l = max(hp,wp)
            y1,y2,x1,x2 = y - r *l, y + r * l, x - r * l, x + r * l
        else:
            if diff_conn_mask.sum() > diff_regions.sum() * 0.5:
                y1,y2,x1,x2 = y1d,y2d,x1d,x2d
            else:
                y1,y2,x1,x2 = y1p,y2p,x1p,x2p
    y1,y2,x1,x2 = expand_bbox(pred_mask,y1,y2,x1,x2,ratio )
    y1 = max(y1,ymin)
    y2 = min(y2,ymax)
    x1 = max(x1,xmin)
    x2 = min(x2,xmax)
    return y1,y2,x1,x2


def get_object_crop(pred_mask, previous_mask, global_roi, y,x, ratio):
    pred_mask = pred_mask > 0.49
    y1,y2,x1,x2 = get_bbox_from_mask(pred_mask)
    y1,y2,x1,x2 = expand_bbox(pred_mask,y1,y2,x1,x2,ratio )
    ymin,ymax,xmin,xmax = global_roi
    y1 = max(y1,ymin)
    y2 = min(y2,ymax)
    x1 = max(x1,xmin)
    x2 = min(x2,xmax)
    return y1,y2,x1,x2



def get_click_crop(pred_mask, previous_mask, global_roi, y,x, ratio):
    pred_mask = pred_mask > 0.49
    y1p,y2p,x1p,x2p= get_bbox_from_mask(pred_mask)
    hp,wp = y2p - y1p, x2p - x1p
    r = 0.2
    l = max(hp,wp)
    y1,y2,x1,x2 = y - r *l, y + r * l, x - r * l, x + r * l
    y1,y2,x1,x2 = expand_bbox(pred_mask,y1,y2,x1,x2,ratio )
    ymin,ymax,xmin,xmax = global_roi
    y1 = max(y1,ymin)
    y2 = min(y2,ymax)
    x1 = max(x1,xmin)
    x2 = min(x2,xmax)
    return y1,y2,x1,x2




def getLargestCC(segmentation):
    if segmentation.sum()<10:
        return segmentation
    labels = label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC
    


def get_diff_region(pred_mask, previous_mask, y, x):
    y,x = int(y), int(x)
    diff_regions = np.logical_xor(previous_mask, pred_mask)
    if diff_regions.sum() > 1000:
        num, labels = cv2.connectedComponents( diff_regions.astype(np.uint8))
        label = labels[y,x]
        corr_mask = labels == label
    else:
        corr_mask = pred_mask
    return corr_mask




def get_bbox_from_mask(mask):
    h,w = mask.shape[0],mask.shape[1]

    if mask.sum() < 10:
        return 0,h,0,w
    rows = np.any(mask,axis=1)
    cols = np.any(mask,axis=0)
    y1,y2 = np.where(rows)[0][[0,-1]]
    x1,x2 = np.where(cols)[0][[0,-1]]
    return y1,y2,x1,x2

def expand_bbox(mask,y1,y2,x1,x2,ratio, min_crop=0):
    H,W = mask.shape[0], mask.shape[1]
    xc, yc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    h = ratio * (y2-y1+1)
    w = ratio * (x2-x1+1)
    h = max(h,min_crop)
    w = max(w,min_crop)

    x1 = int(xc - w * 0.5)
    x2 = int(xc + w * 0.5)
    y1 = int(yc - h * 0.5)
    y2 = int(yc + h * 0.5)

    x1 = max(0,x1)
    x2 = min(W,x2)
    y1 = max(0,y1)
    y2 = min(H,y2)
    return y1,y2,x1,x2


def expand_bbox_with_bias(mask,y1,y2,x1,x2,ratio, min_crop=0, bias = 0.3):
    H,W = mask.shape[0], mask.shape[1]
    xc, yc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    h = ratio * (y2-y1+1)
    w = ratio * (x2-x1+1)
    h = max(h,min_crop)
    w = max(w,min_crop)
    hmax, wmax = int(h * bias), int(w * bias)
    h_bias = np.random.randint(-hmax,hmax+1)
    w_bias = np.random.randint(-wmax,wmax+1)

    x1 = int(xc - w * 0.5) + w_bias
    x2 = int(xc + w * 0.5) + w_bias
    y1 = int(yc - h * 0.5) + h_bias
    y2 = int(yc + h * 0.5) + h_bias

    x1 = max(0,x1)
    x2 = min(W,x2)
    y1 = max(0,y1)
    y2 = min(H,y2)
    return y1,y2,x1,x2



def CalBox(mask,last_y = None, last_x = None, expand = 1.5):
    y1,y2,x1,x2 = get_bbox_from_mask(mask)
    H,W = mask.shape[0], mask.shape[1]
    if last_y is not  None:
        y1 = min(y1,last_y)
        y2 = max(y2,last_y)
        x1 = min(x1, last_x)
        x2 = max(x2,last_x)

    xc, yc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    h = expand * (y2-y1+1)
    w = expand * (x2-x1+1)
    x1 = int(xc - w * 0.5)
    x2 = int(xc + w * 0.5)
    y1 = int(yc - h * 0.5)
    y2 = int(yc + h * 0.5)

    x1 = max(0,x1)
    x2 = min(W,x2)
    y1 = max(0,y1)
    y2 = min(H,y2)
    return y1,y2,x1,x2

def points_back(p_np, y1, x1):
    if p_np is None:
        return None
    bias = np.array( [[y1,x1]]).reshape((1,2))
    return p_np + bias




def PointsInBox(points,y1,y2,x1,x2, H, W ):
    if points is None:
        return None
    
    y_ratio = H/(y2-y1)
    x_ratio = W/(x2-x1)
    num_pos = points.shape[0] // 2
    new_points = np.full_like(points,-1)

    valid_pos = 0
    for i in range(num_pos):
        y,x,index = points[i,0], points[i,1],points[i,2]
        if y>y1 and y< y2 and x>x1 and x<x2:
            new_x, new_y  =  (x-x1) * x_ratio, (y-y1) * y_ratio
            new_points[valid_pos,0], new_points[valid_pos,1], new_points[valid_pos,2] = new_y, new_x, index
            valid_pos += 1
        if y == -1:
            break

    valid_neg = num_pos
    for i in range(num_pos, num_pos * 2):
        y,x,index = points[i,0], points[i,1],points[i,2]
        if y>y1 and y< y2 and x>x1 and x<x2:
            new_x, new_y  =  (x-x1) * x_ratio, (y-y1) * y_ratio
            new_points[valid_neg,0], new_points[valid_neg,1], new_points[valid_neg,2] = new_y, new_x, index
            valid_neg += 1
        if y == -1:
            break
    return np.array(new_points)



def random_choose_target(unknown, crop_size):
    """Randomly choose an unknown start (top-left) point for a given crop_size.

    Args:
        unknown (np.ndarray): The binary unknown mask.
        crop_size (tuple[int]): The given crop size.

    Returns:
        tuple[int]: The top-left point of the chosen bbox.
    """
    h, w = unknown.shape
    crop_h, crop_w = crop_size
    if crop_h > h or crop_w > w:
        return 0,0,h,w


    delta_h = center_h = crop_h // 2
    delta_w = center_w = crop_w // 2

    # mask out the validate area for selecting the cropping center
    mask = np.zeros_like(unknown)
    mask[delta_h:h - delta_h, delta_w:w - delta_w] = 1
    if np.any(unknown & mask):
        center_h_list, center_w_list = np.where(unknown & mask)
    elif np.any(unknown):
        center_h_list, center_w_list = np.where(unknown)
    else:
        #print_log('No unknown pixels found!', level=logging.WARNING)
        center_h_list = [center_h]
        center_w_list = [center_w]
    num_unknowns = len(center_h_list)
    rand_ind = np.random.randint(num_unknowns)
    center_h = center_h_list[rand_ind]
    center_w = center_w_list[rand_ind]

    # make sure the top-left point is valid
    top = np.clip(center_h - delta_h, 0, h - crop_h)
    left = np.clip(center_w - delta_w, 0, w - crop_w)
    y1,x1,y2,x2 = top, left, top + crop_h, left + crop_w

    return y1,x1,y2,x2