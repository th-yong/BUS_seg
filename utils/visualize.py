import random
import matplotlib.pyplot as plt

# Visualize random samples from the dataset with mask overlay and save as one image
def visualize_with_mask(dataset, output_file="visualization.png", figsize=(15, 6)):
    """
    Visualize samples with masks overlayed in a 2x5 grid using fixed index multiples of 5.

    Args:
        dataset: PyTorch Dataset containing images and masks.
        output_file (str): Path to save the output image.
        figsize (tuple): Figure size for the plot.
    """
    num_samples = 10  # 2x5 grid
    indices = [i * 5 for i in range(num_samples)]  # Fixed indices (5's multiple)

    fig, axes = plt.subplots(2, 5, figsize=figsize)

    for idx, ax in zip(indices, axes.flat):
        # Load the image and mask
        image, mask = dataset[idx]

        # Prepare image and mask for visualization
        image_np = image.permute(1, 2, 0).numpy() * 0.5 + 0.5  # De-normalize
        mask_np = mask.squeeze(0).numpy()

        # Overlay mask on the image
        overlay = image_np.copy()
        overlay[..., 0] = image_np[..., 0] * (1 - mask_np) + mask_np  # Add mask to red channel

        # Display the image with overlay
        ax.imshow(overlay)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    # plt.show()
    
# Function to print training information
def print_training_info(model, optimizer, scheduler, device, num_epochs, train_size, val_size, test_size, batch_size):
    """
    Print training parameters and environment details.

    Args:
        model: PyTorch model.
        optimizer: Optimizer used for training.
        scheduler: Learning rate scheduler.
        device: Device (CPU or GPU).
        num_epochs (int): Number of epochs.
        train_size (int): Size of the training dataset.
        val_size (int): Size of the validation dataset.
        test_size (int): Size of the test dataset.
        batch_size (int): Batch size.
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
    print(f"Test Dataset Size: {test_size}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")
    print("\n============================\n")