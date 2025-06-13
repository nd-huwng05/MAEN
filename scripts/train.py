import os
import time
import datetime
import torch.utils.data
from timm.optim import optim_factory
from timm.utils import NativeScaler
from torch.utils.tensorboard import SummaryWriter
from data.dataset import MedicalImageDataset
from utils import misc
from models.aeu import AEU
from utils.engine import train_one_epoch,test_one_epoch


def train(config,args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    tensorboard_dir = os.path.join(args.output, "tensorboard/")
    os.makedirs(tensorboard_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=tensorboard_dir)

    print(f"Load dataset train...")
    dataset = MedicalImageDataset(config, mode="train")
    sampler_train = torch.utils.data.RandomSampler(dataset)
    dataloader_train = torch.utils.data.DataLoader(
        dataset,
        batch_size= args.batch_size,
        sampler=sampler_train,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    print(f"Load dataset test...")
    dataset_test = MedicalImageDataset(config, mode="test")
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    models = [AEU(latent_size=args.ls, expansion=args.mp, input_size=args.image_size, layer=args.layer).to(device=args.device) for _ in range(args.num_model)]
    print("actual lr: %.2e" % args.lr)
    param_groups = [optim_factory.param_groups_weight_decay(model, args.weight_decay) for model in models]
    optimizers = [torch.optim.AdamW(param_group, lr=args.lr, betas=(0.5, 0.999)) for param_group in param_groups]
    print(optimizers)
    loss_scalers = [NativeScaler() for _ in range(args.num_model)]
    auc_best = []
    for i in  range(args.num_model):
        auc_best.append([0])
    start = 0
    if args.resume:
        start, auc_best = misc.load_model(args, models, optimizers, loss_scalers, best=False)
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(start, args.epochs):
        train_stats = train_one_epoch(models, dataloader_train, optimizers, loss_scalers, args, epoch, log_writer)
        log_writer.add_scalars("Train/Losses", {
            "L_sim": train_stats["L_sim"],
            "L_org": train_stats["L_org"],
            "L_total": train_stats["L_total"],
        }, epoch)

        test_stats = test_one_epoch(models, dataloader_test, args, epoch, log_writer)
        log_writer.add_scalars("Test/AUC", {f"Model_{i}": auc for i, auc in enumerate(test_stats["AUC"])}, epoch)
        log_writer.add_scalars("Test/AP", {f"Model_{i}": ap for i, ap in enumerate(test_stats["AP"])}, epoch)

        auc_old = auc_best
        auc_best = [max(auc_best[i], test_stats['AUC'][i]) for i in range(args.num_model)]
        if auc_old != auc_best:
            misc.save_model(args, epoch, models, optimizers, loss_scalers, auc_best=auc_best, auc_old=auc_old, best=True)
        misc.save_model(args,epoch,models,optimizers,loss_scalers, auc_best, best=False)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
