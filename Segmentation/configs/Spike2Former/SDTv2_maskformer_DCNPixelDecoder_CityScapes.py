_base_ = [
    '../_base_/datasets/cityscapes.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

checkpoint_file = '/public/liguoqi/lzx/ckpt/sdtv2/best_acc.pth'
batch_size = 8
num_gpus = 8
embed_dim = 256
ps_dim = 128
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 1024)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))
num_classes = 19
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        type='Spiking_vit_MetaFormer',
        img_size_h=512,
        img_size_w=512,
        patch_size=16,
        embed_dim=[64, 128, 256, 360],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=150,
        qkv_bias=False,
        depths=8,
        sr_ratios=1,
        T=1,
        norm_eval=True,
        norm_cfg=norm_cfg,
        decode_mode='Qsnn'),
    decode_head=dict(
        type='MaskFormerHead',
        in_channels=[32, 64, 128, 360],  # input channels of pixel_decoder modules
        feat_channels=embed_dim,
        in_index=[0, 1, 2, 3],
        num_classes=num_classes,
        out_channels=embed_dim,
        num_queries=100,
        pixel_decoder=dict(
            type='mmdet.DCNTransformerEncoderPixelDecoder',
            norm_cfg=norm_cfg,
            T=4,
            encoder=dict(  # DetrTransformerEncoder
                num_layers=6,
                layer_cfg=dict(  # DetrTransformerEncoderLayer
                    self_attn_cfg=dict(  # MultiheadAttention
                        embed_dims=embed_dim,
                        num_heads=8,
                        batch_first=True,
                        dw_kernel_size=5,
                        group=32),
                    ffn_cfg=dict(
                        embed_dims=embed_dim,
                        feedforward_channels=2048,
                        num_fcs=2))),
            positional_encoding=dict(num_feats=ps_dim, normalize=True)),
        enforce_decoder_input_project=False,
        positional_encoding=dict(  # SinePositionalEncoding
            num_feats=ps_dim, normalize=True),
        transformer_decoder=dict(  # DetrTransformerDecoder
            return_intermediate=True,
            num_layers=6,
            layer_cfg=dict(  # DetrTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=embed_dim,
                    num_heads=8,
                    attn_type='SA',
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=embed_dim,
                    num_heads=8,
                    attn_type='CA',
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=embed_dim,
                    feedforward_channels=2048,
                    num_fcs=2,
                    add_identity=True)),
            init_cfg=None,
        ),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=20.0),
        loss_dice=dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=1.0),
        train_cfg=dict(
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet.ClassificationCost',
                         weight=1.0),
                    dict(
                        type='mmdet.FocalLossCost',
                        weight=20.0,
                        binary_input=True),
                    dict(
                        type='mmdet.DiceCost',
                        weight=1.0,
                        pred_act=True,
                        eps=1.0)
                ]),
            sampler=dict(type='mmdet.MaskPseudoSampler'))),
    # training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

# optimizer
optimizer = dict(type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.005)
# NOTE: ADD
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),  # 0.1 -> 0.01 1.0 -> 0.5
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi,
}

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(custom_keys=custom_keys))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]
# dataset config
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomChoiceResize',
        scales=[int(1024 * x * 0.1) for x in range(5, 21)],
        resize_type='ResizeShortestEdge',
        max_size=4096),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
# In MaskFormer implementation we use batch size 2 per GPU as default
train_dataloader = dict(batch_size=8, num_workers=16, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=1, num_workers=16)
test_dataloader = val_dataloader

lr_config = dict(warmup_iters=1500)
# training schedule for 160k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=90000, val_interval=2500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=10000,
        save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=48)
