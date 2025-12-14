import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def compute_mean_std_nct(root):
    train_dir = Path(root) / "train"

    image_paths = sorted([p for p in train_dir.rglob("*.tif")])

    if len(image_paths) == 0:
        raise RuntimeError(f" {train_dir} ")

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    n_pixels = 0


    for p in tqdm(image_paths):
        img = Image.open(p).convert("RGB")
        img = np.array(img, dtype=np.float32) / 255.0  

        h, w, c = img.shape
        n_pixels += h * w
        channel_sum += img.sum(axis=(0, 1))
        channel_sum_sq += (img ** 2).sum(axis=(0, 1))

    mean = channel_sum / n_pixels
    std = np.sqrt(channel_sum_sq / n_pixels - mean ** 2)

    return mean, std


if __name__ == "__main__":
    root = "./NCT-CRC-HE-100K-split"

    mean, std = compute_mean_std_nct(root)

    print("NCT  mean:", mean.tolist())
    print("NCT  std :", std.tolist())

    np.savez(Path(root) / "nct_mean_std.npz", mean=mean, std=std)
    print("saved", Path(root) / "nct_mean_std.npz")