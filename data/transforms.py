from torchvision import transforms

def get_transform():
    return transforms.Compose([
        transforms.ToTensor()  # Converts image to [0, 1] range automatically
    ])
