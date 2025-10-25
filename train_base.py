"""
Train MattingBase

You can download pretrained DeepLabV3 weights from <https://github.com/VainF/DeepLabV3Plus-Pytorch>

Example:

    CUDA_VISIBLE_DEVICES=0 python train_base.py \
        --dataset-name videomatte240k \
        --model-backbone resnet50 \
        --model-name mattingbase-resnet50-videomatte240k \
        --model-pretrain-initialization "pretraining/best_deeplabv3_resnet50_voc_os16.pth" \
        --epoch-end 8

"""

import argparse
import kornia
import torch
import os
import random

from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.utils import make_grid
from tqdm import tqdm
from torchvision import transforms as T
from PIL import Image
import kornia.geometry.transform as K_transform
import kornia.filters as K_filters
import kornia.enhance as K_enhance

from data_path import DATA_PATH
from dataset import ImagesDataset, ZipDataset, VideoDataset, SampleDataset
from dataset import augmentation as A
from model import MattingBase
from model.utils import load_matched_state_dict


# --------------- Arguments ---------------


parser = argparse.ArgumentParser()

parser.add_argument('--dataset-name', type=str, required=True, choices=DATA_PATH.keys())

parser.add_argument('--model-backbone', type=str, required=True, choices=['resnet101', 'resnet50', 'mobilenetv2'])
parser.add_argument('--model-name', type=str, required=True)
parser.add_argument('--model-pretrain-initialization', type=str, default=None)
parser.add_argument('--model-last-checkpoint', type=str, default=None)

parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--num-workers', type=int, default=16)
parser.add_argument('--epoch-start', type=int, default=0)
parser.add_argument('--epoch-end', type=int, required=True)

parser.add_argument('--log-train-loss-interval', type=int, default=10)
parser.add_argument('--log-train-images-interval', type=int, default=2000)
parser.add_argument('--log-valid-interval', type=int, default=5000)

parser.add_argument('--checkpoint-interval', type=int, default=5000)

parser.add_argument('--learning-rate-backbone', type=float, default=1e-5)
parser.add_argument('--learning-rate-aspp', type=float, default=5e-5)
parser.add_argument('--learning-rate-decoder', type=float, default=5e-5)
parser.add_argument('--weight-decay', type=float, default=1e-4)

args = parser.parse_args()


# --------------- Loading ---------------


def train():
    
    # Training DataLoader
    dataset_train = ZipDataset([
        ZipDataset([
            ImagesDataset(DATA_PATH[args.dataset_name]['train']['pha'], mode='L'),
            ImagesDataset(DATA_PATH[args.dataset_name]['train']['fgr'], mode='RGB'),
        ], transforms=A.PairCompose([
            A.PairRandomAffineAndResize((512, 512), degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.4, 1), shear=(-5, 5)),
            A.PairRandomHorizontalFlip(),
            A.PairRandomBoxBlur(0.1, 5),
            A.PairRandomSharpen(0.1),
            A.PairApplyOnlyAtIndices([1], T.ColorJitter(0.15, 0.15, 0.15, 0.05)),
            A.PairApply(T.ToTensor())
        ]), assert_equal_length=True),
        ImagesDataset(DATA_PATH['backgrounds']['train'], mode='RGB', transforms=T.Compose([
            A.RandomAffineAndResize((512, 512), degrees=(-5, 5), translate=(0.1, 0.1), scale=(1, 2), shear=(-5, 5)),
            T.RandomHorizontalFlip(),
            A.RandomBoxBlur(0.1, 5),
            A.RandomSharpen(0.1),
            T.ColorJitter(0.15, 0.15, 0.15, 0.05),
            T.ToTensor()
        ])),
    ])
    dataloader_train = DataLoader(dataset_train,
                                  shuffle=True,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
    
    # Validation DataLoader
    dataset_valid = ZipDataset([
        ZipDataset([
            ImagesDataset(DATA_PATH[args.dataset_name]['valid']['pha'], mode='L'),
            ImagesDataset(DATA_PATH[args.dataset_name]['valid']['fgr'], mode='RGB')
        ], transforms=A.PairCompose([
            A.PairRandomAffineAndResize((512, 512), degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.3, 1), shear=(-5, 5)),
            A.PairApply(T.ToTensor())
        ]), assert_equal_length=True),
        ImagesDataset(DATA_PATH['backgrounds']['valid'], mode='RGB', transforms=T.Compose([
            A.RandomAffineAndResize((512, 512), degrees=(-5, 5), translate=(0.1, 0.1), scale=(1, 1.2), shear=(-5, 5)),
            T.ToTensor()
        ])),
    ])
    dataset_valid = SampleDataset(dataset_valid, 50)
    dataloader_valid = DataLoader(dataset_valid,
                                  pin_memory=True,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    # Model
    model = MattingBase(args.model_backbone).cuda()

    if args.model_last_checkpoint is not None:
        load_matched_state_dict(model, torch.load(args.model_last_checkpoint))
    elif args.model_pretrain_initialization is not None:
        checkpoint = torch.load(args.model_pretrain_initialization)
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        else:
            # Assume checkpoint is already the state dict
            state_dict = checkpoint
        model.load_pretrained_deeplabv3_state_dict(state_dict)
        # Keep BatchNorm layers in evaluation mode to avoid "Expected more than 1 value per channel"
        # errors when batch size becomes 1 in a minibatch.
        try:
            from torch import nn
            for m in model.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        except Exception:
            pass

    optimizer = AdamW([
        {'params': model.backbone.parameters(), 'lr': args.learning_rate_backbone},
        {'params': model.aspp.parameters(), 'lr': args.learning_rate_aspp},
        {'params': model.decoder.parameters(), 'lr': args.learning_rate_decoder}
    ], weight_decay=args.weight_decay)
    scaler = GradScaler()

    # Logging and checkpoints
    if not os.path.exists(f'checkpoint/{args.model_name}'):
        os.makedirs(f'checkpoint/{args.model_name}')
    writer = SummaryWriter(f'log/{args.model_name}')
    
    # Run loop
    for epoch in range(args.epoch_start, args.epoch_end):
        for i, ((true_pha, true_fgr), true_bgr) in enumerate(tqdm(dataloader_train)):
            step = epoch * len(dataloader_train) + i

            true_pha = true_pha.cuda(non_blocking=True)
            true_fgr = true_fgr.cuda(non_blocking=True)
            true_bgr = true_bgr.cuda(non_blocking=True)
            true_pha, true_fgr, true_bgr = random_crop(true_pha, true_fgr, true_bgr)
            
            true_src = true_bgr.clone()
            
            # Augment with shadow
            aug_shadow_idx = torch.rand(len(true_src)) < 0.3
            if aug_shadow_idx.any():
                aug_shadow = true_pha[aug_shadow_idx].mul(0.3 * random.random())
                aug_shadow = T.RandomAffine(degrees=(-5, 5), translate=(0.2, 0.2), scale=(0.5, 1.5), shear=(-5, 5))(aug_shadow)
                kernel_size = random.choice(range(20, 40))
                aug_shadow = K_filters.box_blur(aug_shadow, (kernel_size, kernel_size))
                true_src[aug_shadow_idx] = true_src[aug_shadow_idx].sub_(aug_shadow).clamp_(0, 1)
                del aug_shadow
            del aug_shadow_idx
            
            # Composite foreground onto source
            true_src = true_fgr * true_pha + true_src * (1 - true_pha)

            # Augment with noise
            aug_noise_idx = torch.rand(len(true_src)) < 0.4
            if aug_noise_idx.any():
                true_src[aug_noise_idx] = true_src[aug_noise_idx].add_(torch.randn_like(true_src[aug_noise_idx]).mul_(0.03 * random.random())).clamp_(0, 1)
                true_bgr[aug_noise_idx] = true_bgr[aug_noise_idx].add_(torch.randn_like(true_bgr[aug_noise_idx]).mul_(0.03 * random.random())).clamp_(0, 1)
            del aug_noise_idx
            
            # Augment background with jitter
            aug_jitter_idx = torch.rand(len(true_src)) < 0.8
            if aug_jitter_idx.any():
                true_bgr[aug_jitter_idx] = kornia.augmentation.ColorJitter(0.18, 0.18, 0.18, 0.1)(true_bgr[aug_jitter_idx])
            del aug_jitter_idx
            
            # Augment background with affine
            aug_affine_idx = torch.rand(len(true_bgr)) < 0.3
            if aug_affine_idx.any():
                true_bgr[aug_affine_idx] = T.RandomAffine(degrees=(-1, 1), translate=(0.01, 0.01))(true_bgr[aug_affine_idx])
            del aug_affine_idx

            with autocast():
                pred_pha, pred_fgr, pred_err = model(true_src, true_bgr)[:3]
                loss = compute_loss(pred_pha, pred_fgr, pred_err, true_pha, true_fgr)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if (i + 1) % args.log_train_loss_interval == 0:
                writer.add_scalar('loss', loss, step)

            if (i + 1) % args.log_train_images_interval == 0:
                writer.add_image('train_pred_pha', make_grid(pred_pha, nrow=5), step)
                writer.add_image('train_pred_fgr', make_grid(pred_fgr, nrow=5), step)
                writer.add_image('train_pred_com', make_grid(pred_fgr * pred_pha, nrow=5), step)
                writer.add_image('train_pred_err', make_grid(pred_err, nrow=5), step)
                writer.add_image('train_true_src', make_grid(true_src, nrow=5), step)
                writer.add_image('train_true_bgr', make_grid(true_bgr, nrow=5), step)
                
            del true_pha, true_fgr, true_bgr
            del pred_pha, pred_fgr, pred_err

            if (i + 1) % args.log_valid_interval == 0:
                valid(model, dataloader_valid, writer, step)

            if (step + 1) % args.checkpoint_interval == 0:
                torch.save(model.state_dict(), f'checkpoint/{args.model_name}/epoch-{epoch}-iter-{step}.pth')

        torch.save(model.state_dict(), f'checkpoint/{args.model_name}/epoch-{epoch}.pth')


# --------------- Utils ---------------


def compute_loss(pred_pha, pred_fgr, pred_err, true_pha, true_fgr):
    true_err = torch.abs(pred_pha.detach() - true_pha)
    true_msk = true_pha != 0
    return F.l1_loss(pred_pha, true_pha) + \
           F.l1_loss(K_filters.sobel(pred_pha), K_filters.sobel(true_pha)) + \
           F.l1_loss(pred_fgr * true_msk, true_fgr * true_msk) + \
           F.mse_loss(pred_err, true_err)


def set_bn_eval(module):
    """Set all BatchNorm layers inside `module` to eval() to avoid batch-stat errors when batch size is 1.

    This keeps running stats and uses them during forward, while still allowing gradients to flow for weights.
    """
    from torch import nn
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def random_crop(*imgs):
    w = random.choice(range(256, 512))
    h = random.choice(range(256, 512))
    results = []
    for img in imgs:
        img = K_transform.resize(img, (max(h, w), max(h, w)))
        img = K_transform.center_crop(img, (h, w))
        results.append(img)
    return results


def valid(model, dataloader, writer, step):
    model.eval()
    loss_total = 0
    loss_count = 0
    with torch.no_grad():
        for (true_pha, true_fgr), true_bgr in dataloader:
            batch_size = true_pha.size(0)
            
            true_pha = true_pha.cuda(non_blocking=True)
            true_fgr = true_fgr.cuda(non_blocking=True)
            true_bgr = true_bgr.cuda(non_blocking=True)
            true_src = true_pha * true_fgr + (1 - true_pha) * true_bgr

            pred_pha, pred_fgr, pred_err = model(true_src, true_bgr)[:3]
            loss = compute_loss(pred_pha, pred_fgr, pred_err, true_pha, true_fgr)
            loss_total += loss.cpu().item() * batch_size
            loss_count += batch_size

    writer.add_scalar('valid_loss', loss_total / loss_count, step)
    model.train()
    # Re-apply BatchNorm eval mode after switching model back to train()
    try:
        set_bn_eval(model)
    except Exception:
        pass


# --------------- Start ---------------


if __name__ == '__main__':
    train()
