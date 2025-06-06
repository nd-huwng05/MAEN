import math
import random
import os
import time
import datetime

import torch.utils.data
from timm.optim import optim_factory
from timm.utils import NativeScaler
from torch.utils.tensorboard import SummaryWriter
from utils.losses import SimilarityLoss
from data.dataset import MedicalImageDataset
from utils import mics
from utils import logger
from utils.logger import SmoothedValue
from models.model_test import VisionTransformer
import torch.nn.functional as F


def train(config,args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    tensorboard_dir = os.path.join(args.output, "tensorboard/")
    os.makedirs(tensorboard_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=tensorboard_dir)

    selected = max(1, math.ceil(args.num_model * args.ratio_abnormal))
    selected = random.sample(range(args.num_model),selected)

    dataloader_train = []
    for k in range(args.num_model):
        print(f"Load dataset train for junior {k}")
        abnormal_ratio = 0.2 if k in selected else 0
        dataset = MedicalImageDataset(config, mode="train", abnormal_ratio=abnormal_ratio)
        sampler_train = torch.utils.data.RandomSampler(dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size= args.batch_size,
            sampler=sampler_train,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        dataloader_train.append(dataloader)
    num_batches = len(dataloader_train[0])

    print(f"Load dataset test...")
    dataset_test = MedicalImageDataset(config, mode="test")
    print(dataset_test)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    models = [VisionTransformer().to(device=args.device) for _ in range(args.num_model)]

    print("actual lr: %.2e" % args.lr)
    param_groups = [optim_factory.param_groups_weight_decay(model, args.weight_decay) for model in models]
    optimizers = [torch.optim.AdamW(param_group, lr=args.lr, betas=(0.9, 0.95)) for param_group in param_groups]
    print(optimizers)
    loss_scalers = [NativeScaler for _ in range(args.num_model)]
    best_auc = 0
    if args.resume:
        start, auc_best = mics.load_model(args, models, optimizers, loss_scalers, best=False)
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.epochs):
        for model in models:
            model.train()

        metric_logger = logger.MetricLogger(delimiter="  ")
        metric_logger.add_meter('L_sim', SmoothedValue(window_size=20, fmt='{median:.4f} ({global_avg:.4f})'))
        metric_logger.add_meter('L_org', SmoothedValue(window_size=20, fmt='{median:.4f} ({global_avg:.4f})'))
        metric_logger.add_meter('L_total', SmoothedValue(window_size=20, fmt='{median:.4f} ({global_avg:.4f})'))
        header = 'Epoch: [{}]'.format(epoch)

        iters = [iter(dl) for dl in dataloader_train]
        for batch_idx, _ in enumerate(metric_logger.log_every(range(num_batches), args.print_freq, header)):

            images = [next(it) for it in iters]
            for opt in optimizers:
                opt.zero_grad()

            attn_maps = []
            feature_maps = []
            x_recons = []
            latents = []

            for i in range(args.num_model):
                latent, _ = images[i]
                latent = latent.to(args.device)
                latents.append(latent)

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



    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
