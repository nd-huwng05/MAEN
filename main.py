from types import SimpleNamespace

import yaml
import os
import argparse
from scripts.train import train
from scripts.inference import inference


def load_config(args):
    config_path = os.path.join('./config', f"{args.config}.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"404 FILE NOT FOUND: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='VinCXR', help='Dataset name')
    parser.add_argument('--mode', type=str, default='train', help='Mode')
    parser.add_argument('--model', type=str, default="MAE", help='Model name')
    args = parser.parse_args()
    str_model = args.model
    config = load_config(args)

    if args.mode == 'train':
        args = argparse.Namespace()
        args.train = SimpleNamespace(**config["train"][f"{str_model}"])
        train(config, args.train, str_model)
        inference(config, args.train, str_model)
    elif args.mode == 'inference':
        args = argparse.Namespace()
        args.train = SimpleNamespace(**config["train"][f"{str_model}"])
        inference(config, args.train, str_model)



