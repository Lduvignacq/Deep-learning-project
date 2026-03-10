import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model_aleatoric import get_model_aleatoric
from dataset import FaceDataset

def get_args():
    parser = argparse.ArgumentParser(description="Age Estimation with Aleatoric Uncertainty")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoint_aleatoric")
    parser.add_argument("--dist", type=str, default="laplace", choices=['gaussian', 'laplace'])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()

def aleatoric_loss(mu, log_var, target, dist='laplace'):
    precision = torch.exp(-log_var)
    if dist == 'gaussian':
        # Loss = 0.5 * exp(-s) * (y-mu)^2 + 0.5 * s
        return (0.5 * precision * (target - mu)**2 + 0.5 * log_var).mean()
    else: 
        # Laplace: Loss = exp(-s) * |y-mu| + s
        return (precision * torch.abs(target - mu) + log_var).mean()

def train():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init Tensorboard
    writer = SummaryWriter(log_dir=args.checkpoint)
    
    # Dataset & Model
    train_dataset = FaceDataset(args.data_dir, mode="train")
    val_dataset = FaceDataset(args.data_dir, mode="val")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    model = get_model_aleatoric().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_mae = float('inf')

    for epoch in range(args.epochs):
        model.train()
        train_loss, train_mae = 0, 0
        
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images, targets = images.to(device), targets.to(device).float()
            
            optimizer.zero_grad()
            mu, log_var = model(images)
            
            loss = aleatoric_loss(mu, log_var, targets, dist=args.dist)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += torch.abs(mu - targets).mean().item()

        # Validation logic (simplifiée)
        model.eval()
        val_mae = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device).float()
                mu, _ = model(images)
                val_mae += torch.abs(mu - targets).mean().item()
        
        val_mae /= len(val_loader)
        print(f"Epoch {epoch}: Val MAE = {val_mae:.2f}")
        
        # Logs
        writer.add_scalar("MAE/train", train_mae/len(train_loader), epoch)
        writer.add_scalar("MAE/val", val_mae, epoch)

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), f"{args.checkpoint}/best_model.pth")

if __name__ == "__main__":
    train()