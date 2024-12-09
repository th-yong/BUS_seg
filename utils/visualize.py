import random
import matplotlib.pyplot as plt
import numpy as np

# Function to visualize dataset with masks
def visualize_with_mask(dataset_name, dataset, output_file="visualization.png", figsize=(15, 6)):
    """
    Visualizes the dataset images with overlayed masks.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'Original', 'Fuzzy').
        dataset: Dataset object containing images and masks.
        output_file (str): File path to save the visualization.
        figsize (tuple): Size of the visualization figure.
    """
    num_samples = 10  # Total number of samples to display
    _, axes = plt.subplots(2, 5, figsize=figsize)

    for i in range(num_samples):
        # Load image and mask
        image, mask = dataset[i]

        # Prepare image and mask for visualization
        image_np = image.permute(1, 2, 0).numpy() * 0.5 + 0.5  # De-normalize
        mask_np = mask.squeeze(0).numpy()

        # Overlay mask on the image
        overlay = image_np.copy()
        overlay[..., 0] = image_np[..., 0] * (1 - mask_np) + mask_np  # Add mask to red channel

        # Determine subplot position
        row = i // 5
        col = i % 5

        # Display the image
        axes[row, col].imshow(image_np)
        axes[row, col].set_title(f"{dataset_name} {i+1}")
        axes[row, col].axis("off")

        # Overlay mask with transparency
        overlay_ax = axes[row, col].twinx()
        overlay_ax.imshow(overlay, alpha=0.7)
        overlay_ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    # plt.show()
    
# Function to print training information
def print_training_info(model, optimizer, scheduler, device, num_epochs, train_size, val_size, batch_size):
    """
    Prints key information about the training setup.

    Args:
        model: PyTorch model being trained.
        optimizer: Optimizer used for training.
        scheduler: Learning rate scheduler.
        device: Device (CPU or GPU).
        num_epochs (int): Number of epochs for training.
        train_size (int): Size of the training dataset.
        val_size (int): Size of the validation dataset.
        batch_size (int): Batch size for training.
    """
    print("\n=== Training Information ===")
    print(f"Device: {device}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Scheduler: {scheduler.__class__.__name__}")
    print(f"Num Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Training Dataset Size: {train_size}")
    print(f"Validation Dataset Size: {val_size}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")
    print("\n============================\n")

# Function to save validation results
def save_validation_images(images, masks, preds, output_file="validation_overlay.png", alpha=0.7):
    """
    Saves validation images with overlaid ground truth and predictions.

    Args:
        images: Batch of validation images.
        masks: Corresponding ground truth masks.
        preds: Model predictions.
        output_file (str): File path to save the visualization.
        alpha (float): Transparency level for overlays.
    """
    images = images.permute(0, 2, 3, 1).cpu().numpy()  # Convert [B, C, H, W] -> [B, H, W, C]
    masks = masks.squeeze(1).cpu().numpy()
    preds = preds.squeeze(1).cpu().numpy()

    num_samples = images.shape[0]
    grid_rows, grid_cols = 4, 10  # Define the grid size for 40 samples

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(20, 8))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
            if i < num_samples:
                # Create overlay
                overlay = images[i].copy()
                overlay[..., 0] = overlay[..., 0] * (1 - masks[i]) + masks[i]  # Mask in red
                overlay[..., 1] = overlay[..., 1] * (1 - preds[i]) + preds[i]  # Prediction in green

                # Display the original image with overlay
                ax.imshow(images[i], cmap="gray")
                ax.axis("off")
                ax.imshow(overlay, alpha=0.7)
                ax.set_title(f"Val {i+1}", fontsize=8)
            else:
                ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.cla()   
    plt.clf()   
    plt.close() 

# Function to save test results
def save_test_images(images, masks, preds, output_file="test_overlay.png"):
    """
    Saves test images with overlaid ground truth and predictions.

    Args:
        images: Batch of test images.
        masks: Corresponding ground truth masks.
        preds: Model predictions.
        output_file (str): File path to save the visualization.
    """
    images = images.permute(0, 2, 3, 1).cpu().numpy()  # Convert [B, C, H, W] -> [B, H, W, C]
    masks = masks.squeeze(1).cpu().numpy()
    preds = preds.squeeze(1).cpu().numpy()

    num_samples = images.shape[0]
    grid_rows, grid_cols = 4, 5  # Define the grid size for 20 samples

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(15, 10))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
            if i < num_samples:
                # Create overlay
                overlay = images[i].copy()
                overlay[..., 0] = overlay[..., 0] * (1 - masks[i]) + masks[i]  # Mask in red
                overlay[..., 1] = overlay[..., 1] * (1 - preds[i]) + preds[i]  ## Prediction in green

                # Display the original image with overlay
                ax.imshow(images[i], cmap="gray")
                ax.axis("off")
                ax.imshow(overlay, alpha=0.7)
                ax.set_title(f"Test {i+1}", fontsize=8)
            else:
                ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close() 