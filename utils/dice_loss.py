import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1.0
        pred = pred.contiguous()
        target = target.contiguous()

        intersection = (pred * target).sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)

        return 1 - dice.mean()

# Evaluate Dice Score
def dice_score(pred, target):
    smooth = 1.0
    pred = (pred > 0.5).float()  # Binarize predictions
    intersection = (pred * target).sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
    return dice.mean().item()