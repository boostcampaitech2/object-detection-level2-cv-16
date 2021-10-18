import os
from mmcv import Config


from .args_parser import args_parser

def config_parser():
    # get parsed args
    args = args_parser()

    # get default config
    # default_config_path is setted as ralated path from exp_train.py
    default_config_path = "./exp_config/default_config.py"
    cfg = Config.fromfile(default_config_path)
    
    # set resume_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    
    # GPU_id for single gpu
    cfg.gpu_ids = [0]
    
    # random seed
    cfg.seed = args.seed


    #============================================================
    #-                       ------------------------------------
    #= Sweep hyperparameters ====================================
    #-                       ------------------------------------

    # max_epochs
    cfg.runner.max_epochs = args.max_epochs
    
    # batch size
    cfg.data.samples_per_gpu = args.samples_per_gpu
    if cfg.data.samples_per_gpu < cfg.data.workers_per_gpu:
        cfg.data.workers_per_gpu = cfg.data.samples_per_gpu


    # optimizer
    cfg.optimizer.lr = args.lr_default
    cfg.lr_config.step = [args.lr_step_1, args.lr_step_2]
    
    # dataset
    data_root = '/opt/ml/detection/dataset/'
    cfg.data.train.img_prefix = data_root
    cfg.data.val.img_prefix = data_root
    
    if args.data_version.lower() == 'v1':
        train_ann_file, valid_ann_file = ('split_train.json', 'split_valid.json')
    elif args.data_version.lower() == 'v2':
        train_ann_file, valid_ann_file = ('split_train_v2.json', 'split_valid_v2.json')
    elif args.data_version.lower() == 'debug':
        train_ann_file, valid_ann_file = ('simple_train.json', 'simple_valid.json')
    else:
        raise ValueError("Invalid data_version")
    cfg.data.train.ann_file = data_root+train_ann_file
    cfg.data.val.ann_file = data_root+valid_ann_file


    # generate a new_config_name with costum hyperparameters
    new_config_name = f"{args.model_name}"
    new_config_name += f"_{cfg.data.samples_per_gpu}"
    new_config_name += f"_{cfg.optimizer.lr:.4f}"
    new_config_name += f"_{cfg.lr_config.step}"

    # workflow
    cfg.workflow = [('train',1),('val',1)]

    # log config setting
    cfg.log_config.max_keep_ckpts=args.max_keep_ckpts
    cfg.log_config.hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='mmdetection',
                name=new_config_name
            )
        )
    ]


    # work-dir setting
    cfg.work_dir = "./work-dir/" + new_config_name
    os.makedirs(cfg.work_dir, exist_ok=True)
    cfg.dump(os.path.join(cfg.work_dir, new_config_name+".py"))

    return new_config_name, cfg
