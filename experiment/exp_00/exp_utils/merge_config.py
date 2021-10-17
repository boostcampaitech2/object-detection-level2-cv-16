from mmcv import Config

# default_config_path is setted as ralated path from exp_train.py
config_file_path = "./exp_config/faster_rcnn_r50_fpn_1x_coco.py"
cfg = Config.fromfile(config_file_path)

cfg.dump("./exp_config/default_config.py")