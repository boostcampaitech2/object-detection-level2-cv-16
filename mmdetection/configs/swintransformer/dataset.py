# dataset settings
# from mmdetection/configs/_base_/datasets/coco_detection.py
dataset_type = 'CocoDataset'
data_root = '../dataset/'
classes = ['General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

img_norm_cfg = dict(
    mean= [127.49413776397705, 127.43779182434082, 127.46098327636719], std= [73.86627551077616, 73.88234865304638, 73.8944344154546], to_rgb=True
    )
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'split_train.json',
        img_prefix=data_root,
        pipeline=train_pipeline,
        classes=classes,
        ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'split_valid.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=classes,
        ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=classes,
        ))
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP_50')
