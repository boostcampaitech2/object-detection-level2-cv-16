import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument(
        '--model_name',
        type=str, help='train config file basename'
    )
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from'
    )
    parser.add_argument(
        '--samples_per_gpu',
        type=int, help='samples_per_gpu'
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
        '--max_epochs',
        type=int, help='max_epochs'
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
    parser.add_argument(
        '--max_norm',
        type=float, help='max_nrom'
    )
    
    args = parser.parse_args()
    return args