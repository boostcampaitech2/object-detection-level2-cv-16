from mmcv import Config

# default_config_path is setted as ralated path from exp_train.py
config_file_path = \
    "/opt/ml/CBNetV2/configs/cbnet/htc_cbv2_swin_base_patch4_window7_mstrain_400-1400_adamw_20e_coco.py"
cfg = Config.fromfile(config_file_path)

cfg.dump("../exp_config/default_config.py")