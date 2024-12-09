import torch
import torch.nn as nn
import torch.nn.functional as F

# Dice Loss: Measures the overlap between predicted and ground truth masks
class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation tasks.
    
    Formula:
        Dice = 2 * (|P ∩ G| + smooth) / (|P| + |G| + smooth)
        Loss = 1 - Dice

    Args:
        smooth (float): Smoothing factor to avoid division by zero.
    """
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc

# Cross-Entropy Loss: Pixel-wise classification loss
class CrossEntropyLoss(nn.Module):
    """
    Cross-Entropy Loss for binary segmentation tasks.
    
    Formula:
        Loss = -[y * log(y_pred) + (1 - y) * log(1 - y_pred)]
    """
    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        return F.binary_cross_entropy_with_logits(input=y_pred, target=y_true)

# Contour Loss: Penalizes differences in edge gradients
class ContourLoss(nn.Module):
    """
    Contour Loss to refine object boundaries by comparing gradients of predictions and ground truth.
    
    Formula:
        Loss = MSE(pred_grad_x, true_grad_x) + MSE(pred_grad_y, true_grad_y)
    """
    def __init__(self):
        super(ContourLoss, self).__init__()
        # Sobel filters for gradient computation
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size(), "y_pred and y_true must have the same size"
        
        # Compute gradients using Sobel filters
        pred_grad_x = F.conv2d(y_pred, self.sobel_x.to(y_pred.device), padding=1)
        pred_grad_y = F.conv2d(y_pred, self.sobel_y.to(y_pred.device), padding=1)
        true_grad_x = F.conv2d(y_true, self.sobel_x.to(y_true.device), padding=1)
        true_grad_y = F.conv2d(y_true, self.sobel_y.to(y_true.device), padding=1)

        # Compute mean squared error between gradients
        loss_x = F.mse_loss(pred_grad_x, true_grad_x)
        loss_y = F.mse_loss(pred_grad_y, true_grad_y)
        contour_loss = loss_x + loss_y

        return contour_loss

# Tversky Loss: Generalized Dice Loss with false positive and false negative penalties
class TverskyLoss(nn.Module):
    """
    Tversky Loss for handling imbalanced datasets.
    
    Formula:
        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        Loss = 1 - Tversky

    Args:
        smooth (float): Smoothing factor to avoid division by zero.
    """
    def __init__(self):
        super(TverskyLoss, self).__init__()
        self.smooth = 1.0
        
    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        
        # True Positives, False Positives, False Negatives
        TP = (y_pred * y_true).sum()    
        FP = ((1-y_true) * y_pred).sum()
        FN = (y_true * (1-y_pred)).sum()
        
        Tversky = (TP + self.smooth) / (TP + 0.5*FP + 0.5*FN + self.smooth)
        
        return 1 - Tversky

# Evaluation Metrics
def dice_score(y_pred, y_true, smooth=1.0):
    """
    Calculates the Dice Coefficient.
    """
    y_pred = (y_pred > 0.5).float() 
    y_pred = y_pred.contiguous().view(-1)
    y_true = y_true.contiguous().view(-1)

    intersection = (y_pred * y_true).sum()
    dice = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
    return dice.item()


def precision_score(y_pred, y_true):
    """
    Calculates precision: TP / (TP + FP).
    """
    y_pred = (y_pred > 0.5).float()
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return precision


def recall_score(y_pred, y_true):
    """
    Calculates recall: TP / (TP + FN).
    """
    y_pred = (y_pred > 0.5).float()
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return recall


def f1_score(y_pred, y_true):
    """
    Calculates F1 score: Harmonic mean of precision and recall.
    """
    precision = precision_score(y_pred, y_true)
    recall = recall_score(y_pred, y_true)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1