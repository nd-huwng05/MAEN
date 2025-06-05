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
        header = 'Epoch: [{}]'.format(epoch)

        for images in zip(*dataloader_train):
            for optimizer in optimizers:
                optimizer.zero_grad()

            cls_tokens = []
            attn_maps = []
            feature_maps = []
            x_recons = []

            for i in range(args.num_model):
                latent, _ = images[i]
                latent = latent.to(args.device)
                models[i] = models[i].float()

                with torch.cuda.amp.autocast(enabled=loss_scalers[i] is not None):
                    cls_token, attn_map, x_recon, feature_map = models[i](latent)
                    cls_tokens.append(cls_token)
                    attn_maps.append(attn_map)
                    x_recons.append(x_recon)
                    feature_maps.append(feature_map)

            for i in range(args.num_model):
                latent, labels = images[i]
                SimLoss = SimilarityLoss()
                features_tuples = [[feature_map[i], feature_map[j]] for j in range(args.num_model) if j != i]
                cls_token_tuples = [[cls_tokens[i], cls_tokens[j]] for j in range(args.num_model) if j != i]
                attn_maps_tuples = [[attn_maps[i], attn_maps[j]] for j in range(args.num_model) if j != i]
                loss = [sum(SimLoss(features=features_tuples[idx],
                                    cls_tokens=cls_token_tuples[idx],
                                    attentions=attn_maps_tuples[idx])) for idx in range(args.num_model - 1)]
                recon_loss = F.mse_loss(x_recons[i],latent)
                L_total = sum(loss) + recon_loss

                optimizers[i].zero_grad()
                L_total.backward()
                optimizers[i].step()


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
