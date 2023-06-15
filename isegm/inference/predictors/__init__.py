from .baseline import BaselinePredictor
from isegm.inference.transforms import ZoomIn



def get_predictor(net, brs_mode, device,
                  prob_thresh=0.49,
                  infer_size = 256,
                  focus_crop_r= 1.4,
                  with_flip=False,
                  zoom_in_params=dict(),
                  predictor_params=None,
                  brs_opt_func_params=None,
                  lbfgs_params=None):

    predictor_params_ = {
        'optimize_after_n_clicks': 1
    }

    if zoom_in_params is not None:
        zoom_in = ZoomIn(**zoom_in_params)
    else:
        zoom_in = None

    if predictor_params is not None:
        predictor_params_.update(predictor_params)
    predictor = BaselinePredictor(net, device, zoom_in=zoom_in, with_flip=with_flip, infer_size =infer_size, **predictor_params_)



    return predictor
