import os

_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_panoptic.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
checkpoint_file = os.getenv('SPIKE2FORMER_BACKBONE_CKPT')

data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=True,
        seg_pad_value=255)

model = dict(
    type='PanopticFPN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=True,
        seg_pad_value=255),
    backbone=dict(
        init_cfg=(dict(type='Pretrained', checkpoint=checkpoint_file)
                  if checkpoint_file else None),
        type='Spiking_vit_MetaFormer',
        img_size_h=1024,
        img_size_w=1024,
        patch_size=16,
        embed_dim=[64, 128, 256, 360],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=150,
        qkv_bias=False,
        depths=8,
        sr_ratios=1,
        T=4,
        decode_mode='Qsnn'),
    neck=dict(
        type='FPN',
        in_channels=[32, 64, 128, 360],
        out_channels=256,
        num_outs=5),
    semantic_head=dict(
        type='PanopticFPNHead',
        num_things_classes=80,
        num_stuff_classes=53,
        in_channels=256,
        inner_channels=128,
        start_level=0,
        end_level=4,
        loss_seg=dict(
            type='CrossEntropyLoss', ignore_index=255, loss_weight=0.5)),
    panoptic_fusion_head=dict(
        type='HeuristicFusionHead',
        num_things_classes=80,
        num_stuff_classes=53),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.6,
            nms=dict(type='nms', iou_threshold=0.5, class_agnostic=True),
            max_per_img=100,
            mask_thr_binary=0.5),
        # used in HeuristicFusionHead
        panoptic=dict(mask_overlap=0.5, stuff_area_limit=4096)))

# Forced to remove NumClassCheckHook
custom_hooks = []


log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=12)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.0001, by_epoch=False, begin=0, end=500), # 0.001
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]
train_dataloader = dict(batch_size=5)
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)