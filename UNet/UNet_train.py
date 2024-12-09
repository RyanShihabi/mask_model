import segmentation_models_pytorch as smp
import torch
from dataset import segDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import shutil
import argparse

def dice_loss(pred, target, smooth=1.):
    intersection = (pred * target).sum(dim=(2, 3))  # Intersection
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))  # Union
    dice = (2.0 * intersection + smooth) / (union + smooth)  # Dice coefficient
    return 1 - dice.mean()

def update_model_endpoint(model_dir, weights):    
    shutil.copy(weights, f"{model_dir}/street_unet.pth")

    os.system(f"~/cvat/serverless/deploy_gpu.sh {model_dir}")

# parser = argparse.ArgumentParser()

# parser.add_argument("--model_dir", type=str, required=True)

# args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=3,
    classes=1
).to(device)

train_dataset = segDataset("./train_street_dataset.json")
val_dataset = segDataset("./val_street_dataset.json")

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

criterion = torch.nn.CrossEntropyLoss()

num_epochs = 150

best_val = 2

for epoch in tqdm(range(num_epochs)):
    model.train()

    train_loss = 0.0

    for batch_idx, (imgs, labels) in enumerate(train_dataloader):
        inputs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs).sigmoid()

        loss = dice_loss(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_dataloader)
    print(f"Training Loss: {avg_train_loss:.4f}")

    model.eval()
    val_loss = 0.0

    for batch_idx, (imgs, labels) in enumerate(val_dataloader):
        inputs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(inputs).sigmoid()

        loss = dice_loss(outputs, labels)

        val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    
    scheduler.step(avg_val_loss)
    print(scheduler.get_last_lr())

    if avg_val_loss < best_val:
        print("saving model...")
        best_val = avg_val_loss

        torch.save(model.state_dict(), "./street_unet.pth")

# update_model_endpoint(args.model_dir, "./street_unet.pth")