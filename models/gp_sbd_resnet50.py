from isegm.utils.exp_imports.default import *
MODEL_NAME = 'resnet50'
# from isegm.data.compose import ComposeDataset,ProportionalComposeDataset
import torch.nn as nn
from isegm.data.aligned_augmentation import AlignedAugmentator
from isegm.engine.gp_trainer import ISTrainer
import importlib

def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (cfg.crop_size, cfg.crop_size)
    model_cfg.num_max_points = cfg.num_max_points
    GpModel = importlib.import_module('isegm.model.'+cfg.gp_model).GpModel
    model = GpModel(backbone = 'resnet50', use_leaky_relu=True, use_disks=(not cfg.nodisk),  binary_prev_mask=False,
                       with_prev_mask=(not cfg.noprev_mask), weight_dir=cfg.IMAGENET_PRETRAINED_MODELS.RESNET50_v1s)
    model.to(cfg.device)
    model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=2.0))
    model.model.load_pretrained_weights()
    return model, model_cfg


def train(model, cfg, model_cfg):
    cfg.batch_size = 32 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0

    train_augmentator = AlignedAugmentator(ratio=[0.3,1.3], target_size=crop_size,flip=True, distribution='Gaussian', gs_center=0.8)

    val_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.25)),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size)
    ], p=1.0)

    points_sampler = MultiPointSampler(model_cfg.num_max_points, prob_gamma=0.70,
                                       merge_objects_prob=0.15,
                                       max_num_merged_objects=2,
                                       use_hierarchy=False,
                                       first_click_center=True)

    trainset = SBDDataset(
        cfg.SBD_PATH,
        split='train',
        augmentator=train_augmentator,
        min_object_area=80,
        keep_background_prob=0.01,
        points_sampler=points_sampler,
        samples_scores_gamma=1.25
    )

    valset = SBDDataset(
        cfg.SBD_PATH,
        split='val',
        augmentator=val_augmentator,
        min_object_area=80,
        points_sampler=points_sampler,
        epoch_len=500
    )

    optimizer_params = {
        'lr': cfg.lr, 'betas': (0.9, 0.999), 'eps': 1e-8
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=cfg.milestones[:-1], gamma=0.1)
    trainer = ISTrainer(model, cfg, model_cfg, loss_cfg,
                        trainset, valset,
                        optimizer='adam',
                        optimizer_params=optimizer_params,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=[(0, 50), (200, 10)],
                        image_dump_interval=cfg.image_dump_interval,
                        metrics=[AdaptiveIoU()],
                        max_interactive_points=model_cfg.num_max_points,
                        max_num_next_clicks=cfg.max_num_next_clicks)
    trainer.run(num_epochs=cfg.milestones[-1])
