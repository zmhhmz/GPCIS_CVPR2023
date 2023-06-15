# training
python3 train.py models/gp_sbd_resnet50.py \
--gpus=0,1 \
--workers=12 \
--batch-size=32 \
--milestones 190 220 230 \
--max_num_next_clicks=3 \
--num_max_points=24 \
--crop_size=256 \
--gp_model=is_gp_resnet50 \
--exp-name=GP_Resnet50_SBD_230epo

# Evaluation
# python3 scripts/evaluate_model.py Baseline \
# --model_dir=checkpoints/ \
# --checkpoint=GPCIS_Resnet50.pth \
# --datasets=GrabCut,Berkeley,SBD,DAVIS \
# --gpus=0 \
# --n-clicks=20 \
# --target-iou=0.90 \
# --thresh=0.50 