import yaml
import os
import argparse
from scripts import run


def load_config(args):
    config_path = os.path.join('../config', f"{args.config}.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"404 FILE NOT FOUND: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='VinCXR', help='Dataset name')
    parser.add_argument('--mode', type=str, default='train', help='Mode')
    args = parser.parse_args()

    config = load_config(args)
    if args.mode == 'train':
        run.train(config, args)
    elif args.mode == 'inference':
        run.inference(config, args)



