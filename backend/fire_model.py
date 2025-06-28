import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import segmentation_models_pytorch as smp
import xarray as xr
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F

# ---------------- CONFIG ----------------

ERA5_GRIB_PATH = "../data/era5_input.grib"       # Path to your ERA5 file
FIRE_MASK_DIR = "../data/viirs_masks"                  # Folder of binary fire masks
MODEL_SAVE_PATH = "../models/unet_fire_model.pth"
VARIABLES = ["t2m", "u10", "v10", "sp", "tp"]        # Update if needed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
EPOCHS = 1
LEARNING_RATE = 1e-4
TARGET_SHAPE = (1800, 3598)  # Match this with mask dimensions
MAX_SAMPLES = 100  # limit dataset size to speed up training

# ---------------- Dataset ----------------

class ERA5FireDataset(Dataset):
    def __init__(self, grib_path, mask_dir, variables):
        self.ds = xr.open_dataset(
            grib_path,
            engine="cfgrib",
            backend_kwargs={"indexpath": ""},
            decode_timedelta=True
        )

        self.variables = variables
        self.mask_dir = Path(mask_dir)
        self.available_masks = sorted(self.mask_dir.glob("*.npy"))

        # Detect valid temporal dimension
        self.temporal_dim = None
        for var in self.variables:
            if var in self.ds:
                for dim in self.ds[var].dims:
                    if dim.lower() in ["time", "step", "valid_time"]:
                        self.temporal_dim = dim
                        break
            if self.temporal_dim:
                break

        if self.temporal_dim is None:
            print("⚠️  No temporal dimension detected. Assuming static GRIB file.")
            self.length = len(self.available_masks)
        else:
            time_len = self.ds.dims.get(self.temporal_dim, 1)
            self.length = min(time_len, len(self.available_masks))

        self.length = min(self.length, MAX_SAMPLES)
        print(f"📊 Temporal dim: {self.temporal_dim} | Training samples: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        channels = []
        for var in self.variables:
            if var not in self.ds:
                raise KeyError(f"Variable '{var}' not found in GRIB file.")
            if self.temporal_dim:
                arr = self.ds[var].isel({self.temporal_dim: idx}).values
            else:
                arr = self.ds[var].values
            channels.append(arr)

        x = np.stack(channels).astype(np.float32)
        y = np.load(self.available_masks[idx]).astype(np.float32)
        y = np.clip(y, 0, 1)

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        if x.shape[1:] != TARGET_SHAPE:
            x = F.interpolate(x.unsqueeze(0), size=TARGET_SHAPE, mode="bilinear", align_corners=False).squeeze(0)
        if y.shape[1:] != TARGET_SHAPE:
            y = F.interpolate(y.unsqueeze(0), size=TARGET_SHAPE, mode="nearest").squeeze(0)

        x = (x - x.mean()) / (x.std() + 1e-6)

        if y.max() == 0:
            y[0:10, 0:10] = 1.0

        return x.float(), y.float()

# ---------------- Training ----------------

def train_unet():
    print("📦 Loading dataset...")
    full_dataset = ERA5FireDataset(ERA5_GRIB_PATH, FIRE_MASK_DIR, VARIABLES)
    dataloader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("🧠 Building U-Net model...")
    model = smp.Unet(encoder_name="resnet18", in_channels=len(VARIABLES), classes=1)  # Lighter encoder
    model.to(DEVICE)

    dice = smp.losses.DiceLoss(mode="binary", smooth=1e-5)
    bce = nn.BCEWithLogitsLoss()

    def combined_loss(preds, targets):
        if preds.shape != targets.shape:
            targets = F.interpolate(targets, size=preds.shape[2:], mode="bilinear", align_corners=False)
        return dice(preds, targets) + bce(preds, targets)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("🚀 Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for x, y in loop:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if y.ndim == 3:
                y = y.unsqueeze(1)
            if x.ndim == 3:
                x = x.unsqueeze(0)
            try:
                preds = model(x)
                loss = combined_loss(preds, y)
                if torch.isnan(loss):
                    print("❌ NaN detected in loss, skipping batch.")
                    continue
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print("❌ CUDA OOM - skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        torch.cuda.empty_cache()
        print(f"✅ Epoch {epoch+1}: Avg Loss = {total_loss / len(dataloader):.4f}")

    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"🎯 Model saved to: {MODEL_SAVE_PATH}")

# ---------------- Entry Point ----------------

if __name__ == "__main__":
    train_unet()

