import os
from mmcv import Config


from .args_parser import args_parser

def config_parser():
    # get parsed args
    args = args_parser()

    new_config_name = f"{args.model_name}"

    # get default config
    # default_config_path is setted as ralated path from exp_train.py
    default_config_path = "./exp_config/default_config.py"
    cfg = Config.fromfile(default_config_path)
    
    # GPU_id for single gpu
    cfg.gpu_ids = [0]
    
    # batch size
    cfg.data.samples_per_gpu = args.samples_per_gpu
    cfg.data.workers_per_gpu = cfg.data.samples_per_gpu

    # model config
    cfg.roi_feat_size = args.roi_feat_size
    cfg.class_loss_weight = args.class_loss_weight
    cfg.bbox_loss_weight = args.bbox_loss_weight
    cfg.rpn_iou_threshold = args.rpn_iou_threshold

    # optimizer
    cfg.optimizer.lr = args.lr_default
    cfg.lr_config.step = [args.lr_step_1,args.lr_step_2]
    cfg.optimizer_config.grad_clip = dict(max_norm=args.max_norm, norm_type=2)
    cfg.optimizer.weight_decay = args.weight_decay

    # max_epochs
    cfg.runner.max_epochs = args.max_epochs
    
    # dataset
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    data_root = '/opt/ml/detection/dataset/'
    if args.data_version.lower() == 'v1':
        train_ann_file, valid_ann_file = ('split_train.json', 'split_valid.json')
    elif args.data_version.lower() == 'v2':
        train_ann_file, valid_ann_file = ('split_train_v2.json', 'split_valid_v2.json')
    elif args.data_version.lower() == 'debug':
        train_ann_file, valid_ann_file = ('simple_train.json', 'simple_valid.json')
    else:
        raise ValueError("Invalid data_version")
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = data_root
    cfg.data.train.ann_file = data_root+train_ann_file
    
    cfg.data.val.classes = classes
    cfg.data.val.img_prefix = data_root
    cfg.data.val.ann_file = data_root+valid_ann_file

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = data_root
    cfg.data.test.ann_file = data_root + 'test.json' # test json 정보
    cfg.data.test.pipeline[1]['img_scale'] = (1024,1024) # Resize
    
    if args.multi_scale == 'multi':
        scale_1 = [
            (614, 1024), (655, 1024), (696, 1024), (1024, 1024),
            (737, 1024), (778, 1024), (819, 1024), (983, 1024),
            (860, 1024), (901, 1024), (942, 1024), 
        ]
        scale_2 = [
            (512, 1024), (640, 1024), (768, 1024)
        ]
        scale_3 = [
            (614, 1024), (655, 1024), (696, 1024), (1024, 1024),
            (737, 1024), (778, 1024), (819, 1024), (983, 1024),
            (860, 1024), (901, 1024), (942, 1024), 
        ]
    else:
        scale_1, scale_2, scale_3 = [(1024,1024)], [(1024,1024)], [(1024,1024)]
    cfg.data.train.pipeline[3].policies[0][0].img_scale = scale_1
    cfg.data.train.pipeline[3].policies[1][0].img_scale = scale_2
    cfg.data.train.pipeline[3].policies[1][2].img_scale = scale_3

    # random seed
    cfg.seed = args.seed


    # generate a new_config_name with costum hyperparameters
    new_config_name += f"_{cfg.data.samples_per_gpu}"
    new_config_name += f"_{args.multi_scale}"
    new_config_name += f"_{cfg.optimizer.lr:.4f}"
    new_config_name += f"_{cfg.lr_config.step}"

    # log config setting
    cfg.log_config.hooks.append(
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='mmdetection',
                name=new_config_name,
                #entity = 'passion-ate'
            )
        )
    )
    # ckpt setting
    cfg.checkpoint_config = dict(max_keep_ckpts=args.max_keep_ckpts, interval=1)
    
    cfg.model.roi_head.bbox_head[0].num_classes = 10
    cfg.model.roi_head.bbox_head[1].num_classes = 10
    cfg.model.roi_head.bbox_head[2].num_classes = 10
    

    # work-dir setting
    cfg.work_dir = "./work-dir/" + new_config_name
    os.makedirs(cfg.work_dir, exist_ok=True)
    cfg.dump(os.path.join(cfg.work_dir, new_config_name+".py"))
    cfg.pretrained = ('/opt/ml/detection/mmdetection/'+\
        'configs/swin/swin_base_patch4_window12_384_22k.pth')
    return new_config_name, cfg
