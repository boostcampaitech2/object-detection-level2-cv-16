_base_ = [
    'cascade_rcnn_r50_fpn.py',
    'dataset.py',
    'default_runtime.py',
    'schedule_1x.py'
]

pretrained =  '/opt/ml/detection/mmdetection/configs/swin/swin_base_patch4_window12_384_22k.pth'


model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
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
    neck=dict(in_channels=[128, 256, 512, 1024]))
