from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.v2 import Compose
import os, sys
from argparse import ArgumentParser
from typing import Union, Tuple
import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import datasets


def get_dataloader(args: ArgumentParser, split: str = "train", ddp: bool = False) -> Union[Tuple[DataLoader, Union[DistributedSampler, None]], DataLoader]:
    # Check if using UKB dataset
    if hasattr(args, 'dataset_type') and args.dataset_type == 'ukb':
        return get_ukb_dataloader(args, split, ddp)
    
    # Original crowd counting dataset logic
    if split == "train":  # train, strong augmentation
        transforms = Compose([
            datasets.RandomResizedCrop((args.input_size, args.input_size), scale=(args.min_scale, args.max_scale)),
            datasets.RandomHorizontalFlip(),
            datasets.RandomApply([
                datasets.ColorJitter(brightness=args.brightness, contrast=args.contrast, saturation=args.saturation, hue=args.hue),
                datasets.GaussianBlur(kernel_size=args.kernel_size, sigma=(0.1, 5.0)),
                datasets.PepperSaltNoise(saltiness=args.saltiness, spiciness=args.spiciness),
            ], p=(args.jitter_prob, args.blur_prob, args.noise_prob)),
        ])

    elif args.sliding_window:
        if args.resize_to_multiple:
            transforms = datasets.Resize2Multiple(args.window_size, stride=args.stride)
        elif args.zero_pad_to_multiple:
            transforms = datasets.ZeroPad2Multiple(args.window_size, stride=args.stride)
        else:
            transforms = None

    else:
        transforms = None

    dataset = datasets.Crowd(
        dataset=args.dataset,
        split=split,
        transforms=transforms,
        sigma=None,
        return_filename=False,
        num_crops=args.num_crops if split == "train" else 1,
    )

    if ddp and split == "train":  # data_loader for training in DDP
        sampler = DistributedSampler(dataset)
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=datasets.collate_fn,
        )
        return data_loader, sampler

    elif split == "train":  # data_loader for training
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=datasets.collate_fn,
        )
        return data_loader, None

    else:  # data_loader for evaluation
        data_loader = DataLoader(
            dataset,
            batch_size=1,  # Use batch size 1 for evaluation
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=datasets.collate_fn,
        )
        return data_loader


def get_ukb_dataloader(args: ArgumentParser, split: str = "train", ddp: bool = False) -> Union[Tuple[DataLoader, Union[DistributedSampler, None]], DataLoader]:
    """Get dataloader for UKB dataset"""
    
    # Define transforms for UKB dataset
    if split == "train":  # train, strong augmentation
        transforms = Compose([
            datasets.RandomResizedCrop((args.input_size, args.input_size), scale=(args.min_scale, args.max_scale)),
            datasets.RandomHorizontalFlip(),
            datasets.RandomApply([
                datasets.ColorJitter(brightness=args.brightness, contrast=args.contrast, saturation=args.saturation, hue=args.hue),
                datasets.GaussianBlur(kernel_size=args.kernel_size, sigma=(0.1, 5.0)),
            ], p=(args.jitter_prob, args.blur_prob)),
        ])
    else:
        transforms = None

    # 在 get_ukb_dataloader 函数中
    dataset = datasets.UKBDataset(
        excel_path=args.excel_path,
        data_root=args.data_path,
        target_column=args.target_column,
        split=split,
        transforms=transforms,
        return_filename=False,
        num_crops=args.num_crops if split == "train" else 1,
    )

    # Custom collate function for UKB dataset
    def ukb_collate_fn(batch):
        if len(batch[0]) == 3:  # images, targets, filenames
            images, targets, filenames = zip(*batch)
            images = torch.cat(images, 0)
            targets = torch.cat(targets, 0)
            return images, targets, filenames
        else:  # images, targets
            images, targets = zip(*batch)
            images = torch.cat(images, 0)
            targets = torch.cat(targets, 0)
            return images, targets

    if ddp and split == "train":  # data_loader for training in DDP
        sampler = DistributedSampler(dataset)
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=ukb_collate_fn,
        )
        return data_loader, sampler

    elif split == "train":  # data_loader for training
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=ukb_collate_fn,
        )
        return data_loader, None

    else:  # data_loader for evaluation
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size if hasattr(args, 'eval_batch_size') and args.eval_batch_size else 1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=ukb_collate_fn,
        )
        return data_loader
