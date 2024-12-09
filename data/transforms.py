from torchvision import transforms

def get_transform():
    """
    Returns a transformation pipeline for preprocessing images.

    Transformation Steps:
        1. Converts a PIL Image or numpy.ndarray to a PyTorch tensor.
        2. Scales the pixel values to the [0, 1] range automatically.

    Returns:
        torchvision.transforms.Compose: Transformation pipeline.
    """
    return transforms.Compose([
        transforms.ToTensor() # Converts image to PyTorch tensor and scales to [0, 1] range
    ])
