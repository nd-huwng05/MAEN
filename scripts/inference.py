import os
import time

import torch
import tqdm
import numpy as np
from utils import misc
from sklearn import metrics
from torch.autograd import Variable
from timm.optim import optim_factory
from timm.utils import NativeScaler
from models.model_factory import AEU_Net, MAE_Net
from data.dataset import MedicalImageDataset

def inference(config, args, name):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    dataset_test = MedicalImageDataset(config, mode="test")
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    if name == "AEU":
        models = [AEU_Net(args).to(device=args.device) for _ in range(args.num_model)]
    else:
        models = [MAE_Net(args).to(device=args.device) for _ in range(args.num_model)]
    print("actual lr: %.2e" % args.lr)
    param_groups = [optim_factory.param_groups_weight_decay(model, args.weight_decay) for model in models]
    optimizers = [torch.optim.AdamW(param_group, lr=args.lr, betas=(0.5, 0.999)) for param_group in param_groups]
    print(optimizers)
    loss_scalers = [NativeScaler() for _ in range(args.num_model)]
    misc.load_model(args, models, optimizers, loss_scalers, best=True)

    print("=> Evaluating ... ")
    time.sleep(1)
    for model in models:
        model.eval()
    y_true, unc_dis_l  = [],[]
    for image, label in tqdm.tqdm(dataloader_test, ncols=120):
        x = Variable(image, requires_grad=True).to(args.device)
        grad_recs, unc = [], []
        for model in models:
            out = model(x)
            mean, logvar = out["x_hat"], out["log_var"]
            rec_err = (x - mean) ** 2
            loss = torch.mean(torch.exp(-logvar) * rec_err)
            gradient = torch.autograd.grad(torch.mean(loss), x)[0].squeeze(0)
            grad_rec = torch.abs(mean - x) * gradient
            unc.append(torch.exp(logvar).squeeze(0))
            grad_recs.append(grad_rec)
        grad_recs = torch.cat(grad_recs)
        unc = torch.cat(unc)

        var = torch.mean(unc, dim=0)
        unc_dis = torch.std(grad_recs / torch.sqrt(var), dim=0)
        unc_dis_l.append(unc_dis.detach().mean().cpu())
        y_true.append(label.cpu().item())

    unc_dis_l = np.array(unc_dis_l)
    y_true = np.array(y_true)

    unc_auc = metrics.roc_auc_score(y_true, unc_dis_l)
    unc_ap = metrics.average_precision_score(y_true, unc_dis_l)

    unc_str = "Dual-space Uncertainty      AUC:{:.3f}  AP:{:.3f}".format(
        unc_auc, unc_ap
    )
    print(unc_str)