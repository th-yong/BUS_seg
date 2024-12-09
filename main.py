import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from data.dataset import BreastUltrasoundDataset, split_dataset
from data.transforms import get_transform
from models.unet import UNet
from models.attention_unet import AttentionUNet
from models.unet_plus import NestedUNet
from models.CMUNeXt import CMUNeXt
from models.SegResNet import SegResNet
from utils.loss_function import *
from utils.visualize import print_training_info, visualize_with_mask, save_validation_images, save_test_images

# Map network names to model classes
NETWORKS = {
    "unet": UNet,
    "attention_unet": AttentionUNet,
    "unet_plus" : NestedUNet,
    "CMUNeXt" : CMUNeXt,
    "SegResNet" : SegResNet,
}

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Train or test segmentation model on breast ultrasound dataset.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], 
                        help="Specify whether to train or test the model: 'train' or 'test'")
    parser.add_argument('--dataset', type=str, required=True, choices=['original', 'fuzzy'], 
                        help="Specify the dataset to use: 'original' or 'fuzzy'")
    parser.add_argument('--network', type=str, required=True, choices=NETWORKS.keys(),
                        help="Specify the network architecture to use: " + ", ".join(NETWORKS.keys()))
    parser.add_argument('--model_path', type=str, required=False, default="./results/best_model.pth",
                        help="Path to the model file for testing (only used in 'test' mode).")
    parser.add_argument('--num_epochs', type=int, required=False, default=100,
                        help='number of total epochs to run')
    parser.add_argument('--lr', type=float, required=False, default=1e-4,
                        help='segmentation network learning rate')
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

    train_dataset, val_dataset, test_dataset = split_dataset(selected_dataset, test_ratio=0.1, val_ratio=0.2, random_seed=42)

    # visualize_with_mask(dataset_name, test_dataset, output_file=f'./config/{dataset_name}_Benign_Overlay.png')

    
    # Initialize batch size, num_epochs and patience
    batch_size = 2
    num_epochs = args.num_epochs
    patience = 10
    lr = args.lr
    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NETWORKS[args.network]().to(device)
    criterion = DiceLoss()

    if args.mode == "train":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Print training information
        print_training_info(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=num_epochs,
            train_size=len(train_loader)*batch_size,
            val_size=len(val_loader)*batch_size,
            batch_size=batch_size,
        )

        # Training loop
        best_val_dice = 0.0
        best_val_loss = float('inf')
        
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
            all_images, all_masks, all_preds = [], [], []

            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
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
                if os.path.exists(f"./results/best_model_{args.network}_{dataset_name}_valloss_{best_val_loss:.2f}.pth"):
                    os.remove(f"./results/best_model_{args.network}_{dataset_name}_valloss_{best_val_loss:.2f}.pth")
                    os.remove(f"./results/{args.network}_validation_{dataset_name}_valloss_{best_val_loss:.2f}.png")

                best_val_loss = val_loss
                model_save_path = f"./results/best_model_{args.network}_{dataset_name}_valloss_{best_val_loss:.2f}.pth"
                torch.save(model.state_dict(), model_save_path)
                print(f"Best model saved as {model_save_path} with Val Loss: {best_val_loss:.4f}")
                
                save_validation_images(all_images, all_masks, all_preds, output_file=f"./results/{args.network}_validation_{dataset_name}_valloss_{best_val_loss:.2f}.png")

    elif args.mode == "test":
        # Test logic
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Load model
        model.load_state_dict(torch.load(args.model_path))
        model.eval()

        print(f"Testing model: {args.model_path}")
        
        # Calculate model parameters in millions
        total_params = sum(p.numel() for p in model.parameters())
        total_params_in_m = total_params / 1e6  # Convert to millions
        print(f"Model Parameters: {total_params_in_m:.2f}M")

        # Initialize inference time tracker
        total_images = 0
        inference_times = []
        
        # Initialize lists to store batch-wise metrics
        dice_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        all_images, all_masks, all_preds = [], [], []
        
        with torch.no_grad():
            for images, masks in tqdm(test_loader, desc="Testing"):
                images, masks = images.to(device), masks.to(device)
                batch_size = images.size(0)  # Get the number of images in the current batch
                total_images += batch_size
                
                # Measure inference time
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                start_time.record()

                outputs = model(images)

                end_time.record()
                torch.cuda.synchronize()  # Ensure time measurement is accurate
                inference_times.append(start_time.elapsed_time(end_time))  # Time in milliseconds

                # Collect all validation samples
                all_images.append(images)
                all_masks.append(masks)
                all_preds.append((outputs > 0.5).float())

                # Calculate batch metrics
                dice = dice_score(outputs, masks)
                precision = precision_score(outputs, masks)
                recall = recall_score(outputs, masks)
                f1 = f1_score(outputs, masks)

                # Append metrics to lists
                dice_scores.append(dice)
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)

        # Calculate average inference time per image
        total_inference_time = sum(inference_times)  # Total time for all batches
        avg_inference_time_per_image = total_inference_time / total_images  # Average time per image

        # Calculate average and standard deviation
        test_dice_mean, test_dice_std = torch.tensor(dice_scores).mean().item(), torch.tensor(dice_scores).std().item()
        precision_mean, precision_std = torch.tensor(precision_scores).mean().item(), torch.tensor(precision_scores).std().item()
        recall_mean, recall_std = torch.tensor(recall_scores).mean().item(), torch.tensor(recall_scores).std().item()
        f1_mean, f1_std = torch.tensor(f1_scores).mean().item(), torch.tensor(f1_scores).std().item()

        # Print results
        print(f"Test Dice Score: {test_dice_mean:.4f} ± {test_dice_std:.4f}")
        print(f"Test Precision: {precision_mean:.4f} ± {precision_std:.4f}")
        print(f"Test Recall: {recall_mean:.4f} ± {recall_std:.4f}")
        print(f"Test F1 Score: {f1_mean:.4f} ± {f1_std:.4f}")
        print(f"Average Inference Time per Image: {avg_inference_time_per_image:.2f} ms")

        # Concatenate all test data for visualization
        all_images = torch.cat(all_images, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        all_preds = torch.cat(all_preds, dim=0)
        
        save_test_images(all_images, all_masks, all_preds, output_file=f"./results/{args.network}_test_{dataset_name}.png")
