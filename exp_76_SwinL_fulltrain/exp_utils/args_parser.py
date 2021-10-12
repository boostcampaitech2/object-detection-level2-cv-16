import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument(
        '--model_name',
        type=str, help='train config file basename'
    )
    parser.add_argument(
        '--samples_per_gpu',
        type=int, help='samples_per_gpu'
    )
    parser.add_argument(
        '--roi_feat_size',
        type=int, help='lr_default'
    )
    parser.add_argument(
        '--class_loss_weight',
        type=float, help='lr_default'
    )
    parser.add_argument(
        '--bbox_loss_weight',
        type=float, help='lr_default'
    )
    parser.add_argument(
        '--rpn_iou_threshold',
        type=float, help='lr_default'
    )
    parser.add_argument(
        '--lr_default',
        type=float, help='lr_default'
    )
    parser.add_argument(
        '--lr_step_1',
        type=int, help='lr_step'
    )
    parser.add_argument(
        '--weight_decay',
        type=float, help='max_epochs'
    )
    parser.add_argument(
        '--max_norm',
        type=float, help='max_epochs'
    )
    parser.add_argument(
        '--max_epochs',
        type=int, help='max_epochs'
    )
    parser.add_argument(
        '--multi_scale',
        type=str, help='multi_scale'
    )
    parser.add_argument(
        '--data_version',
        type=str, help='data_version'
    )
    parser.add_argument(
        '--seed',
        type=int, help='data_version'
    )
    parser.add_argument(
        '--max_keep_ckpts',
        type=int, help='max_keep_ckpts'
    )
    
    args = parser.parse_args()
    return args