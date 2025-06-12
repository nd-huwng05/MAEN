import os.path
import torch
from pathlib import Path

def save_model(args, epoch, models, optimizers, loss_scalers, auc_best, auc_old=None, best=False):
    output = Path(os.path.join(args.output, "best/")) if best else Path(os.path.join(args.output, "latest/"))
    output.mkdir(parents=True, exist_ok=True)
    for i in range(args.num_model):
        if loss_scalers[i] is not None:
            path_latest = f"checkpoint-{'best' if best else 'latest'}-{i}.pth"
            checkpoint_path = output/path_latest
            if not best:
                to_save = {
                    'model': models[i].state_dict(),
                    'optimizer': optimizers[i].state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scalers[i].state_dict(),
                    'AUC': auc_best[i].item(),
                }
                torch.save(to_save, checkpoint_path)
            elif best and auc_best[i].item() > auc_old[i].item():
                to_save = {
                    'model': models[i].state_dict(),
                    'optimizer': optimizers[i].state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scalers[i].state_dict(),
                    'AUC': auc_best[i].item(),
                }
                torch.save(to_save, checkpoint_path)

def load_model(args, models, optimizers, loss_scalers, best=False):
    epoch = 0
    auc_best = []
    output = Path(os.path.join(args.output, "best/")) if best else Path(os.path.join(args.output, "latest/"))
    for i in range(args.num_model):
        checkpoint_file = output / f"checkpoint-{'best' if best else 'latest'}-{i}.pth"
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"404 FILE NOT FOUND: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        models[i].load_state_dict(checkpoint['model'])
        optimizers[i].load_state_dict(checkpoint['optimizer'])
        loss_scalers[i].load_state_dict(checkpoint['scaler'])
        auc_best.append(checkpoint['AUC'])
        epoch = checkpoint['epoch']
    return None if best else epoch, auc_best


