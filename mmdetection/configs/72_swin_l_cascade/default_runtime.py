checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        dict(
        type='WandbLoggerHook',
        init_kwargs=dict(
            project='mmdetection',
            name='72-Swin_L_cosineannealing_train_v3_f3',
            entity = 'passion-ate'
        ))
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
