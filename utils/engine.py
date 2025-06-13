import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import logger
import torch
from utils.losses import MAENLoss

def train_one_epoch(models, dataloader_train, optimizers, loss_scalers, args, epoch, log_writer):
    for model in models:
        model.train()

    metric_logger = logger.MetricLogger(delimiter="  ")
    metric_logger.add_meter('L_sim', logger.SmoothedValue(window_size=20, fmt='{median:.4f} ({global_avg:.4f})'))
    metric_logger.add_meter('L_org', logger.SmoothedValue(window_size=20, fmt='{median:.4f} ({global_avg:.4f})'))
    metric_logger.add_meter('L_total', logger.SmoothedValue(window_size=20, fmt='{median:.4f} ({global_avg:.4f})'))
    header = 'Epoch: [{}]'.format(epoch)

    for images, _ in metric_logger.log_every(dataloader_train, args.print_freq, header):
        for opt in optimizers:
            opt.zero_grad()

        features = []
        x_recons = []
        log_vars = []

        latents = images
        latents = latents.to(args.device)
        latents.requires_grad = True

        for i in range(args.num_model):
            with torch.amp.autocast(device_type='cuda', enabled=loss_scalers[i] is not None):
                output = models[i](latents)
                features.append(output['features'])
                x_recons.append(output['x_hat'])
                log_vars.append(output['log_var'])

        latents = latents.to(dtype=torch.float32)
        log_vars = [lv.to(dtype=torch.float32) for lv in log_vars]
        x_recons = [xr.to(dtype=torch.float32) for xr in x_recons]
        features = [f.to(dtype=torch.float32) for f in features]

        loss = MAENLoss(alpha=1, beta=1, gamma=1)
        L_sim, L_org, L_total = loss(x_recons, features, latents, log_vars)
        for opt in optimizers:
            opt.zero_grad()
        metric_logger.update(L_sim=L_sim.item(), L_org=L_org.item(), L_total=L_total.item())

        L_total.backward()
        for opt in optimizers:
            opt.step()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def test_one_epoch(models, dataloader_test, args, epoch, log_writer):
    for i in range(args.num_model):
        models[i].eval()
    metric_logger = logger.MetricLogger(delimiter="  ")
    header = 'Testing epoch: [{}]'.format(epoch)


    y_scores = [[] for _ in range(args.num_model)]
    y_trues = [[] for _ in range(args.num_model)]
    with torch.no_grad():
        for images, label in metric_logger.log_every(dataloader_test, args.print_freq, header):
            x = images.to(args.device)

            for i in range(args.num_model):
                output  = models[i](x)
                x_hat, log_var = output['x_hat'], output['log_var']
                rec_err = (x_hat - x)**2
                res = torch.exp(-log_var)*rec_err
                res = res.mean(dim=(1,2,3))

                y_trues[i].append(label.detach().cpu())
                y_scores[i].append(res.detach().cpu().view(-1))

        y_trues = [np.concatenate(y_true) for y_true in y_trues]
        y_scores = [np.concatenate(y_score) for y_score in y_scores]
        aucs = [roc_auc_score(y_trues[i], y_scores[i]) for i in range(args.num_model)]
        aps = [average_precision_score(y_trues[i], y_scores[i]) for i in range(args.num_model)]

        auc_dict = {f"AUC_model_{i}": aucs[i] for i in range(args.num_model)}
        ap_dict = {f"AP_model_{i}": aps[i] for i in range(args.num_model)}
        metric_logger.update(**auc_dict, **ap_dict)
        metric_logger.synchronize_between_processes()

        print("Averaged stats:", metric_logger)
        return {
            **{k: meter.global_avg for k, meter in metric_logger.meters.items()},
            "AUC": aucs,
            "AP": aps,
        }
