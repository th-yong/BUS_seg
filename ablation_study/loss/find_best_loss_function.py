import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))


import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse

from data.transforms import get_transform
from data.dataset import BreastUltrasoundDataset, split_dataset
from utils.loss_function import *
from models.unet import UNet
from utils.visualize import save_validation_images

def train_model(model, train_loader, val_loader, loss_function, optimizer, scheduler, device, num_epochs):
    best_val_loss = float('inf')
    best_val_dice = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0        
        all_images, all_masks, all_preds = [], [], []

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = loss_function(outputs, masks)
                val_loss += loss.item()
                
                # Collect all validation samples
                all_images.append(images)
                all_masks.append(masks)
                all_preds.append((outputs > 0.5).float())

                val_dice += dice_score(outputs, masks)

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        scheduler.step(val_loss)

        # Concatenate all validation data for visualization
        all_images = torch.cat(all_images, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        all_preds = torch.cat(all_preds, dim=0)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"./ablation_study/loss/results/best_model_{loss_function.__class__.__name__}.pth")
            print(f"Best model saved with Val Loss: {best_val_loss:.4f}")

            save_validation_images(all_images, all_masks, all_preds, output_file=f"./ablation_study/loss/results/best_model_{loss_function.__class__.__name__}.png")

    return model, best_val_loss

def test_model(model, test_loader, device):
    model.eval()
    test_dice = 0.0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            test_dice += dice_score(outputs, masks)

    test_dice /= len(test_loader)
    print(f"Test Dice Score: {test_dice:.4f}")
    return test_dice

def plot_evaluation_results(loss_names, test_scores, output_file="evaluation_results.png"):
    """
    Plot evaluation results as a bar chart.

    Args:
        loss_names (list): List of loss function names.
        test_scores (list): Corresponding test scores for each loss function.
        output_file (str): Path to save the output plot.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(loss_names, test_scores, color='skyblue')
    plt.xlabel('Loss Functions', fontsize=14)
    plt.ylabel('Test Dice Score', fontsize=14)
    plt.title('Comparison of Loss Functions for Segmentation', fontsize=16)
    plt.ylim(0.6, 0.8)  # Assuming Dice scores are between 0 and 1
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()

if __name__ == "__main__" :
    
    # Argument parsing
    parser = argparse.ArgumentParser(description="Train or Test a segmentation model.")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True,
                        help="Mode: 'train' for training, 'test' for testing.")
    parser.add_argument("--model_dir", type=str, default="./ablation_study/loss/results",
                        help="Directory to save/load models.")
    args = parser.parse_args()

    # Common setup
    data_dir = r"D:\DEV\BUS_SEG\MT_SMALL_DATASET"
    original_benign_path = os.path.join(data_dir, "Benign", "Original_Benign")
    ground_truth_path = os.path.join(data_dir, "Benign", "Ground_Truth_Benign")
    transform = get_transform()
    original_benign_dataset = BreastUltrasoundDataset(original_benign_path, ground_truth_path, transform=transform)

    train_dataset, val_dataset, test_dataset = split_dataset(original_benign_dataset, test_ratio=0.1, val_ratio=0.2, random_seed=42)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    
    if args.mode == "train":
        loss_functions = [CrossEntropyLoss(), DiceLoss(), ContourLoss(), TverskyLoss()]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        for loss_function in loss_functions:
            print(f"Training with {loss_function.__class__.__name__}")
            train_model(
                model, train_loader, val_loader, loss_function, optimizer, scheduler, device, num_epochs=100
            )

    elif args.mode == "test":
        loss_function_names = ['Cross Entropy', 'Dice', 'Contour', 'Tversky']
        test_dice_scores = []

        for loss_name in loss_function_names:
            model_path = os.path.join(args.model_dir, f"best_model_{loss_name.replace(' ', '')}Loss.pth")
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Testing model loaded from {model_path}")

            test_dice = test_model(model, test_loader, device)
            test_dice_scores.append(test_dice)

        # Plot results
        plot_evaluation_results(loss_function_names, test_dice_scores, output_file=os.path.join(args.model_dir, "evaluation_results_test.png"))
    