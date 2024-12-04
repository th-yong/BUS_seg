import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
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

class CrossEntropyLoss(nn.Module):
    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        return F.binary_cross_entropy_with_logits(input=y_pred, target=y_true)

class ContourLoss(nn.Module):
    def __init__(self):
        super(ContourLoss, self).__init__()
        # Define Sobel filters for x and y gradients
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

class TverskyLoss(nn.Module):
    def __init__(self):
        super(TverskyLoss, self).__init__()
        self.smooth = 1.0
        
    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (y_pred * y_true).sum()    
        FP = ((1-y_true) * y_pred).sum()
        FN = (y_true * (1-y_pred)).sum()
        
        Tversky = (TP + self.smooth) / (TP + 0.5*FP + 0.5*FN + self.smooth)
        
        return 1 - Tversky

def dice_score(y_pred, y_true, smooth=1.0):
    y_pred = (y_pred > 0.5).float() 
    y_pred = y_pred.contiguous().view(-1)
    y_true = y_true.contiguous().view(-1)

    intersection = (y_pred * y_true).sum()
    dice = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
    return dice.item()


def precision_score(y_pred, y_true):
    y_pred = (y_pred > 0.5).float()
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return precision


def recall_score(y_pred, y_true):
    y_pred = (y_pred > 0.5).float()
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return recall


def f1_score(y_pred, y_true):
    precision = precision_score(y_pred, y_true)
    recall = recall_score(y_pred, y_true)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1