# datasets/seq_nct.py

import os
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from typing import Tuple
from pathlib import Path

from datasets.utils import set_default_from_args
from utils import smart_joint
from utils.conf import base_path
from datasets.utils.continual_dataset import ContinualDataset, fix_class_names_order, store_masked_loaders
from datasets.transforms.denormalization import DeNormalize
from torchvision.transforms.functional import InterpolationMode


class MyNCT(Dataset):
    """
    NCT-CRC-HE-100K dataset loader.
    Reads from folder structure: root/train/0_ADI/*.tif, root/train/1_BACK/*.tif, etc.
    """
    
    N_CLASSES = 9
    
    LABELS = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.not_aug_transform = transforms.ToTensor()


        split = 'train' if train else 'test' 
        split_dir = Path(root) / split
        
        self.data = []
        self.targets = []
        class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        
        for class_dir in class_dirs:
            class_idx = int(class_dir.name.split('_')[0])
            image_files = sorted([f for f in class_dir.iterdir() if f.suffix.lower() == '.tif'])
            
            for img_file in image_files:
                self.data.append(str(img_file))
                self.targets.append(class_idx)
        
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target, not_aug_img) where target is index of the target class.
        """
        img_path, target = self.data[index], self.targets[index]
        img = Image.open(img_path).convert('RGB')
        original_img = img.copy()
        
        not_aug_img = self.not_aug_transform(original_img)


        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.train:
            return img, target

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class SequentialNCT(ContinualDataset):
    """Sequential NCT-CRC-HE-100K Dataset."""

    NAME = 'seq-nct-224'
    SETTING = 'class-il'
    N_TASKS = 3  
    N_CLASSES_PER_TASK = 3 
    N_CLASSES = 9
    SIZE = (224, 224)
    MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    TRANSFORM = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomCrop(SIZE[0]),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(SIZE[0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    def get_data_loaders(self):
        train_dataset = MyNCT(
            base_path() + 'NCT-CRC-HE-100K-split', 
            train=True,
            download=False, 
            transform=self.TRANSFORM
        )
        test_dataset = MyNCT(
            base_path() + 'NCT-CRC-HE-100K-split', 
            train=False,
            download=False, 
            transform=self.TEST_TRANSFORM
        )

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = fix_class_names_order(MyNCT.LABELS, self.args)
        self.class_names = classes
        return self.class_names

    @staticmethod
    def get_transform():
        transform = transforms.Compose([
            transforms.ToPILImage(), 
            SequentialNCT.TRANSFORM
        ])
        return transform

    @set_default_from_args("backbone")
    def get_backbone():
        return "resnet18"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(mean=SequentialNCT.MEAN, std=SequentialNCT.STD)

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialNCT.MEAN, SequentialNCT.STD)
        return transform

    @set_default_from_args('n_epochs')
    def get_epochs(self):
        return 50

    @set_default_from_args('batch_size')
    def get_batch_size(self):
        return 32
