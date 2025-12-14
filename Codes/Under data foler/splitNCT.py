#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 09:38:23 2025

@author: yuanyuanwu
"""

import os
import random
import shutil
from pathlib import Path

src_dir = Path("NCT-CRC-HE-100K")
out_dir = Path("NCT-CRC-HE-100K-split")

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
random_seed = 42

#make sure ratios must sum to 1
assert abs(train_ratio + val_ratio + test_ratio -1) < 1e-6

random.seed(random_seed)

classes = [d.name for d in src_dir.iterdir() if d.is_dir()]
classes = sorted(classes)


"""
how to manipulate list in python to select the last two elements?
it has two ways, one is [8:]; second is [-2:]
imgs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'] # 10 images
n_train = 8
train_imgs = imgs[:8] #→ [a,b,c,d,e,f,g,h]
test_imgs  = imgs[8:] #→ [i,j]
test1=imgs[len(train_imgs):]
test = imgs[-(len(imgs)-len(train_imgs)):

imgs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'] # 10 images
n_train = 8
train_imgs = imgs[:8] #→ [a,b,c,d,e,f,g,h]
val_imgs = imgs[8:8+1]
test_imgs  = imgs[8+1:] #→ [i,j]

test1=imgs[len(train_imgs):]
test = imgs[-(len(imgs)-len(train_imgs)):]

"""



for idx,cls in enumerate(classes):
    
    src_cls_dir = src_dir/cls
    imgs = [p for p in src_cls_dir.iterdir() if p.is_file()]
    print(f"imgs type: {type(imgs)}")
    random.shuffle(imgs)
    
    n_total = len(imgs)
    #print(f"n_total:{n_total}")
    n_train = int(n_total * train_ratio)
    #print(f"n_train:{n_train}")
    n_val = int(n_total * val_ratio)
    #print("n_val:",n_val)
    n_test = n_total - n_train - n_val
    #print("n_test:",n_test)
    #print("sum",n_train+n_val+n_test)
    train_imgs = imgs[:n_train] #take the first 80% of the shuffled list and assign them to the training set
    val_imgs = imgs[n_train:n_train+n_val]
    test_imgs = imgs[n_train+n_val:]
    
    print(f"{cls}: total={n_total}, train={len(train_imgs)}, val={(len(val_imgs))},test={(len(test_imgs))}")
    
    #create output dirs
    #actually, I should convert class names to class orders
  
    print(f'{idx}_{cls}')
    cls_order = f"{idx}_{cls}"
    train_out = out_dir/"train"/cls_order
    val_out = out_dir/"val"/cls_order
    test_out = out_dir/"test"/cls_order
    
    train_out.mkdir(parents=True, exist_ok=True)
    val_out.mkdir(parents=True, exist_ok=True)
    test_out.mkdir(parents=True, exist_ok=True)
    
    #copy files
    for img in train_imgs:
        shutil.copy2(img, train_out/img.name)
    for img in val_imgs:
        shutil.copy2(img, val_out/img.name)
    for img in test_imgs:
        shutil.copy2(img, test_out/img.name)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
