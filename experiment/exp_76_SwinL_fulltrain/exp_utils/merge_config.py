from mmcv import Config

# default_config_path is setted as ralated path from exp_train.py
config_file_path = "/opt/ml/object-detection-level2-cv-16/exp_70_swin_squeeze/exp_config/55_swin_l_cascade/55_swin.py"
cfg = Config.fromfile(config_file_path)

cfg.dump("../exp_config/default_config.py")