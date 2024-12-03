import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from data.dataset import BreastUltrasoundDataset, split_dataset
from data.transforms import get_transform
from models.unet import UNet
from utils.dice_loss import DiceLoss, dice_score
from utils.visualize import print_training_info, visualize_with_mask

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Train or test segmentation model on breast ultrasound dataset.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], 
                        help="Specify whether to train or test the model: 'train' or 'test'")
    parser.add_argument('--dataset', type=str, required=True, choices=['original', 'fuzzy'], 
                        help="Specify the dataset to use: 'original' or 'fuzzy'")
    parser.add_argument('--model_path', type=str, required=False, default="./results/best_model.pth",
                        help="Path to the model file for testing (only used in 'test' mode).")
    args = parser.parse_args()

    # Paths to data directories
    data_dir = r"D:\DEV\BREAST-ULTRASOUND\MT_SMALL_DATASET"
    fuzzy_benign_path = os.path.join(data_dir, "Benign", "Fuzzy_Benign")
    original_benign_path = os.path.join(data_dir, "Benign", "Original_Benign")
    ground_truth_path = os.path.join(data_dir, "Benign", "Ground_Truth_Benign")
    
    # Transformations
    transform = get_transform()

    # Dataset selection based on input argument
    if args.dataset == "original":
        dataset_name = "original"
        selected_dataset = BreastUltrasoundDataset(original_benign_path, ground_truth_path, transform=transform)
    elif args.dataset == "fuzzy":
        dataset_name = "fuzzy"
        selected_dataset = BreastUltrasoundDataset(fuzzy_benign_path, ground_truth_path, transform=transform)
    
    dataset_size = len(selected_dataset)

    train_size = int(0.7 * dataset_size)
    val_size = int(0.2 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = split_dataset(selected_dataset, test_ratio=0.1, val_ratio=0.2, random_seed=42)

    # visualize_with_mask(dataset_name, test_dataset, output_file=f'./config/{dataset_name}_Benign_Overlay.png')

    
    # Initialize batch size, num_epochs and patience
    batch_size = 8
    num_epochs = 100
    patience = 5

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    criterion = DiceLoss()

    if args.mode == "train":
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Print training information
        print_training_info(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=num_epochs,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            batch_size=batch_size,
        )

        # Training loop
        best_val_dice = 0.0
        early_stopping_counter = 0

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0

            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            val_dice = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)

                    # Calculate Validation Loss
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()

                    # Calculate Validation Dice Score
                    val_dice += dice_score(outputs, masks)

            val_loss /= len(val_loader)
            val_dice /= len(val_loader)

            scheduler.step(val_dice)

            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

            if val_dice > best_val_dice:
                # Remove previous best model file if it exists
                if os.path.exists(f"./results/best_model_{dataset_name}_valdice_{best_val_dice:.2f}.pth"):
                    os.remove(f"./results/best_model_{dataset_name}_valdice_{best_val_dice:.2f}.pth")
                
                # Update best_val_dice and save the new best model
                best_val_dice = val_dice
                early_stopping_counter = 0
                model_save_path = f"./results/best_model_{dataset_name}_valdice_{val_dice:.2f}.pth"
                torch.save(model.state_dict(), model_save_path)
                print(f"Best model saved as {model_save_path} with Val Dice: {best_val_dice:.4f}")
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                break

    elif args.mode == "test":
        # Test logic
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Load model
        model.load_state_dict(torch.load(args.model_path))
        model.eval()

        print(f"Testing model: {args.model_path}")
        test_dice = 0.0
        with torch.no_grad():
            for images, masks in tqdm(test_loader, desc="Testing"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)

                # Calculate Dice Score
                test_dice += dice_score(outputs, masks)

        test_dice /= len(test_loader)
        print(f"Test Dice Score: {test_dice:.4f}")
