import os

auto_scale_lr = dict(base_batch_size=36, enable=True)
backbone_norm_multi = dict(decay_mult=0.0, lr_mult=0.01)
batch_size = 8
checkpoint_file = os.getenv('SPIKE2FORMER_BACKBONE_CKPT')
crop_size = (
    512,
    512,
)
custom_hooks = [
    dict(type='ResetModelHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmseg.engine.hooks.resetmodel_hook',
    ])
custom_keys = dict(
    backbone=dict(decay_mult=1.0, lr_mult=0.1),
    level_embed=dict(decay_mult=0.0, lr_mult=1.0),
    query_embed=dict(decay_mult=0.0, lr_mult=1.0),
    query_feat=dict(decay_mult=0.0, lr_mult=1.0))
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        512,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = os.getenv('VOC2012_ROOT', 'data/VOCdevkit/VOC2012')
dataset_type = 'PascalVOCDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=10000,
        save_best='mIoU',
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
embed_dim = 256
embed_multi = dict(decay_mult=0.0, lr_mult=1.0)
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
lr_config = dict(warmup_iters=1500)
model = dict(
    backbone=dict(
        T=1,
        decode_mode='Qsnn',
        depths=8,
        embed_dim=[
            64,
            128,
            256,
            360,
        ],
        img_size_h=512,
        img_size_w=512,
        in_channels=3,
        init_cfg=(dict(
            checkpoint=checkpoint_file,
            type='Pretrained'
        ) if checkpoint_file else None),
        mlp_ratios=4,
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        norm_eval=True,
        num_classes=150,
        num_heads=8,
        patch_size=16,
        qkv_bias=False,
        sr_ratios=1,
        type='Spiking_vit_MetaFormer'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        enforce_decoder_input_project=False,
        feat_channels=256,
        in_channels=[
            32,
            64,
            128,
            360,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_cls=dict(
            class_weight=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.1,
            ],
            loss_weight=1.0,
            reduction='mean',
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False),
        loss_dice=dict(
            activate=True,
            eps=1.0,
            loss_weight=1.0,
            naive_dice=True,
            reduction='mean',
            type='mmdet.DiceLoss',
            use_sigmoid=True),
        loss_mask=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=20.0,
            reduction='mean',
            type='mmdet.FocalLoss',
            use_sigmoid=True),
        num_classes=150,
        num_queries=100,
        out_channels=256,
        pixel_decoder=dict(
            T=4,
            encoder=dict(
                layer_cfg=dict(
                    ffn_cfg=dict(
                        embed_dims=256, feedforward_channels=1024, num_fcs=2),
                    self_attn_cfg=dict(
                        batch_first=True,
                        dw_kernel_size=5,
                        embed_dims=256,
                        group=32,
                        num_heads=8)),
                num_layers=6),
            norm_cfg=dict(requires_grad=True, type='SyncBN'),
            positional_encoding=dict(normalize=True, num_feats=128),
            type='mmdet.DCNTransformerEncoderPixelDecoder'),
        positional_encoding=dict(normalize=True, num_feats=128),
        train_cfg=dict(
            assigner=dict(
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=1.0),
                    dict(
                        binary_input=True,
                        type='mmdet.FocalLossCost',
                        weight=20.0),
                    dict(
                        eps=1.0,
                        pred_act=True,
                        type='mmdet.DiceCost',
                        weight=1.0),
                ],
                type='mmdet.HungarianAssigner'),
            sampler=dict(type='mmdet.MaskPseudoSampler')),
        transformer_decoder=dict(
            init_cfg=None,
            layer_cfg=dict(
                cross_attn_cfg=dict(
                    attn_type='CA',
                    batch_first=True,
                    embed_dims=256,
                    num_heads=8),
                ffn_cfg=dict(
                    add_identity=True,
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2),
                self_attn_cfg=dict(
                    attn_type='SA',
                    batch_first=True,
                    embed_dims=256,
                    num_heads=8)),
            num_layers=6,
            return_intermediate=True),
        type='MaskFormerHead'),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
num_classes = 150
num_gpus = 8
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.01, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.001, type='AdamW', weight_decay=0.005),
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(decay_mult=1.0, lr_mult=0.1),
            level_embed=dict(decay_mult=0.0, lr_mult=1.0),
            query_embed=dict(decay_mult=0.0, lr_mult=1.0),
            query_feat=dict(decay_mult=0.0, lr_mult=1.0))),
    type='OptimWrapper')
optimizer = dict(
    betas=(
        0.9,
        0.999,
    ),
    lr=0.001,
    momentum=0.9,
    type='AdamW',
    weight_decay=0.005)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1500, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=1500,
        by_epoch=False,
        end=160000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
ps_dim = 128
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='ImageSets/Segmentation/val.txt',
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        data_root=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                512,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='PascalVOCDataset'),
    num_workers=16,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        2048,
        512,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(
    max_iters=120000, type='IterBasedTrainLoop', val_interval=2500)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='ImageSets/Segmentation/train.txt',
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        data_root=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    2048,
                    512,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    512,
                    512,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='PascalVOCDataset'),
    num_workers=16,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            2048,
            512,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        512,
        512,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='ImageSets/Segmentation/val.txt',
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        data_root=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                512,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='PascalVOCDataset'),
    num_workers=16,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/Ablation/v2_spike2former_voc2012_1x4'
