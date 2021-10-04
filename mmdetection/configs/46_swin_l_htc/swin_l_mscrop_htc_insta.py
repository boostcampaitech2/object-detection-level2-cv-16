_base_ = [
    './htc_without_semantic_r50_fpn_1x_coco.py'
]

pretrained = '/opt/ml/detection/mmdetection/configs/swin/swin_large_patch4_window12_384_22k.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
         mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768, 1536])
)

img_norm_cfg = dict(
    mean=[127.49413776397705, 127.43779182434082, 127.46098327636719], 
    std=[73.86627551077616, 73.88234865304638, 73.8944344154546], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='InstaBoost',
        action_candidate=('normal', 'horizontal', 'skip'),
        action_prob=(1, 0, 0),
        scale=(0.8, 1.2),
        dx=15,
        dy=15,
        theta=(-1, 1),
        color_prob=0.5,
        hflag=False,
        aug_ratio=0.5),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(614, 1024), (655, 1024), (696, 1024), (737, 1024),
                           (778, 1024), (819, 1024), (860, 1024), (901, 1024), 
                           (942, 1024), (983, 1024), (1024, 1024)],
                multiscale_mode='value',
                keep_ratio=True)
        ],
                  [
                      dict(
                          type='Resize',
                          img_scale=[(512, 1024), (640, 1024), (768, 1024)],
                          multiscale_mode='value',
                          keep_ratio=True),
                      dict(
                          type='RandomCrop',
                          crop_type='absolute_range',
                          crop_size=(384, 600),
                          allow_negative_crop=True),
                      dict(
                          type='Resize',
                          img_scale=[(614, 1024), (655, 1024), (696, 1024), 
                                     (737, 1024), (778, 1024), (819, 1024), 
                                     (860, 1024), (901, 1024), (942, 1024), 
                                     (983, 1024), (1024, 1024)],
                          multiscale_mode='value',
                          override=True,
                          keep_ratio=True)
                  ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
data = dict(train=dict(pipeline=train_pipeline))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
lr_config = dict(warmup_iters=1000, step=[27, 33])
runner = dict(max_epochs=36)
