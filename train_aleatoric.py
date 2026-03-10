import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model_aleatoric import get_model_aleatoric
from dataset import FaceDataset

def aleatoric_loss(mu, log_var, target, dist='laplace'):
    precision = torch.exp(-log_var)
    if dist == 'gaussian':
        return (0.5 * precision * (target - mu)**2 + 0.5 * log_var).mean()
    # Laplace est souvent plus stable pour l'âge
    return (precision * torch.abs(target - mu) + log_var).mean()

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoint_aleatoric")
    parser.add_argument("--dist", type=str, default="laplace")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.checkpoint).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.checkpoint)

    # Utilisation stricte de ta classe FaceDataset
    train_dataset = FaceDataset(args.data_dir, data_type="train", augment=True)
    val_dataset = FaceDataset(args.data_dir, data_type="valid", augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    model = get_model_aleatoric().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_mae = float('inf')

    for epoch in range(args.epochs):
        model.train()
        t_loss, t_mae = 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs, labels = imgs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            
            output = model(imgs)
            mu, log_var = output[:, 0], output[:, 1]
            
            loss = aleatoric_loss(mu, log_var, labels, dist=args.dist)
            loss.backward()
            optimizer.step()
            
            t_loss += loss.item()
            t_mae += torch.abs(mu - labels).mean().item()

        # Validation
        model.eval()
        v_mae = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device).float()
                output = model(imgs)
                v_mae += torch.abs(output[:, 0] - labels).mean().item()
        
        v_mae /= len(val_loader)
        writer.add_scalar("MAE/val", v_mae, epoch)
        print(f"Epoch {epoch} - Val MAE: {v_mae:.2f}")

        if v_mae < best_mae:
            best_mae = v_mae
            torch.save({'state_dict': model.state_dict()}, f"{args.checkpoint}/best.pth")

if __name__ == "__main__":
    train()