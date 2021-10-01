_base_ = [
    'cascade_rcnn_r50_fpn.py',
    'dataset.py',
    'default_runtime.py',
    'schedule_1x.py'
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
model = dict(
    #type='MaskRCNN', #cascade_rcnn을 쓸 것 이므로
    backbone=dict(
        _delete_=True,  #기존의 Resnet을 지우고 SwinTransformer를 사용하겠다는 의미. Backbone 뿐만 아니라 다른 config에도 사용 가능하다.
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
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
    neck=dict(in_channels=[96, 192, 384, 768]) #neck의 channel 수가 다른 경우가 많으므로 유의해서 수정한다.
    #[256, 512, 1024, 2048]
)