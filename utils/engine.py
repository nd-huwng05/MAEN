from sklearn.metrics import roc_auc_score, average_precision_score
from utils import logger
import torch
import torch.nn.functional as F
from utils.losses import SimilarityLoss

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

        attn_maps = []
        feature_maps = []
        x_recons = []
        latents = []

        for i in range(args.num_model):
            latent = images
            latent = latent.to(args.device)

            models[i] = models[i].float()
            with torch.amp.autocast(device_type='cuda', enabled=loss_scalers[i] is not None):
                attn_map, x_recon, feature_map = models[i](latent)

                attn_maps.append(attn_map)
                x_recons.append(x_recon)
                feature_maps.append(feature_map)

        feature_pooled = [fm.mean(dim=(2, 3)) for fm in feature_maps]

        L_sim = 0.0
        for i in range(args.num_model):
            sim_loss_fn = SimilarityLoss(device=args.device, idx=i)
            L_sim += sim_loss_fn(
                features=feature_pooled,
                attentions=attn_maps
            )

        L_org = 0.0
        for i in range(args.num_model):
            latent = latents[i]
            recon = x_recons[i].to(latent.device)
            L_org += F.mse_loss(recon, latent.to(recon.dtype))

        L_total = L_sim + L_org

        metric_logger.update(
            L_sim=L_sim.item(),
            L_org=L_org.item(),
            L_total=L_total.item()
        )

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

    all_targets = []
    all_scores = []
    for images, targets in metric_logger.log_every(dataloader_test, args.print_freq, header):
        images = images.to(args.device)
        targets = targets.to(args.device)

        batch_scores = []

        for model in models:
            images.requires_grad = True
            attn_map, x_recon, feature_map = model(images)

            rec_error = F.mse_loss(x_recon, images, reduction='none')
            rec_error_pixel = rec_error.mean(dim=1, keepdim=True)

            loss = rec_error_pixel.mean()
            grads = torch.autograd.grad(outputs=loss, inputs=images, retain_graph=True, create_graph=False, only_inputs=True)[0]
            grads_pixel = grads.abs().mean(dim=1, keepdim=True)

            if attn_map.shape[-2:] != rec_error_pixel.shape[-2:]:
                attn_map = F.interpolate(attn_map, size=rec_error_pixel.shape[-2:], mode='bilinear')

            attn_map = attn_map.to(args.device)
            grads_pixel = grads_pixel.to(args.device)
            rec_error_pixel = rec_error_pixel.to(args.device)

            anomaly_map = attn_map * grads_pixel * rec_error_pixel
            score_per_image = anomaly_map.flatten(1).mean(dim=1)
            batch_scores.append(score_per_image.detach().cpu())

            del attn_map, x_recon, feature_map, rec_error, rec_error_pixel, loss
            del grads, grads_pixel, anomaly_map, score_per_image
            torch.cuda.empty_cache()

        final_score = torch.stack(batch_scores, dim=0).mean(dim=0)
        all_targets.append(targets.detach().cpu())
        all_scores.append(final_score.detach().cpu())

        del images, targets, batch_scores, final_score
        torch.cuda.empty_cache()

    all_targets = torch.cat(all_targets).cpu().numpy()
    all_scores = torch.cat(all_scores).cpu().numpy()

    auc = roc_auc_score(all_targets, all_scores)
    ap = average_precision_score(all_targets, all_scores)

    metric_logger.update(AUC=auc, AP=ap)
    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger)
    return {
        **{k: meter.global_avg for k, meter in metric_logger.meters.items()},
        "AUC": auc,
        "AP": ap
    }
