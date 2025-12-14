# preprocess_nct_macenko.py
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from datasets.transforms.macenko_normalizer import MacenkoStainNormalize


def preprocess_nct_with_macenko(
    src_root: str,
    dst_root: str,
    ref_image_relpath: str,
):

    src_root = Path(src_root)
    dst_root = Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    ref_image_path = src_root / ref_image_relpath
    print("reference image:", ref_image_path)

    macenko = MacenkoStainNormalize(str(ref_image_path))

    for split in ["train", "test"]:
        src_split_dir = src_root / split
        dst_split_dir = dst_root / split

        print(f"split: {split}")
        class_dirs = sorted([d for d in src_split_dir.iterdir() if d.is_dir()])

        for class_dir in class_dirs:
            rel_class = class_dir.relative_to(src_root)  # e.g. "train/0_ADI"
            dst_class_dir = dst_root / rel_class
            dst_class_dir.mkdir(parents=True, exist_ok=True)

            image_files = sorted(
                [f for f in class_dir.iterdir() if f.suffix.lower() == ".tif"]
            )

            for img_path in tqdm(image_files, desc=str(rel_class)):
                dst_path = dst_class_dir / img_path.name
                if dst_path.exists():
                    continue

                img = Image.open(img_path).convert("RGB")
                img_norm = macenko(img)
                img_norm.save(dst_path)


if __name__ == "__main__":
    SRC_ROOT = "./data/NCT-CRC-HE-100K-split"
    DST_ROOT = "./data/NCT-CRC-HE-100K-macenko-split"
    REF_IMAGE_RELPATH = "train/0_ADI/ADI-AAAMHQMK.tif"

    preprocess_nct_with_macenko(SRC_ROOT, DST_ROOT, REF_IMAGE_RELPATH)
