import yaml
import os
import argparse

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str, default='VinCXR', required=True, help='Dataset name')

    args = parser.parse_args()
    config_path = os.path.join('../config', f"{args.config}.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"404 FILE NOT FOUND: {config_path}")

    with open(config_path,'r') as f:
        config = yaml.safe_load(f)

    return config

if __name__=="__main__":
    config = load_config()