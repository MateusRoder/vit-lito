import cv2
import torch
import numpy as np
import torchvision.transforms as T

from pytorch_wavelets import DWTForward, DWTInverse

class PrintShape(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, img):
        print("SHAPE IN", img.shape, img.type(), img.max())
        return img
    
class MinMaxScale(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.min=0
        self.max=1

    def forward(self, img):
        X_std = (img - img.min()) / (img.max() - img.min())
        img = X_std * (self.max - self.min) + self.min
        
        return img
    
class MeanCenter(torch.nn.Module):

    def __init__(self, mean):
        super().__init__()
        self.mean=mean

    def forward(self, img):
        
        return img-self.mean
    
class Five_Crop(torch.nn.Module):
    """Crops a given image into 5 images.    

    Args:
        size (sequence or int): Desired output size of the crop.
    """

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            Tensor: Cropped, Normalized [0,1] and stacked images.
        """

        (top_left, top_right, bottom_left, bottom_right, center) = T.FiveCrop(size=self.size)(img)
        top_left = T.PILToTensor()(top_left).type(torch.float).div(255)
        top_right = T.PILToTensor()(top_right).type(torch.float).div(255)
        bottom_left = T.PILToTensor()(bottom_left).type(torch.float).div(255)
        bottom_right = T.PILToTensor()(bottom_right).type(torch.float).div(255)
        center = T.PILToTensor()(center).type(torch.float).div(255)
        final = torch.stack([top_left, top_right, bottom_left, bottom_right, center])
        
        return final
    

class Wavelet(torch.nn.Module):
    """Applies the Wavelet function to a given image.

    Args:
        size (sequence or int): Desired output size of the crop.
    """

    def __init__(self):
        super().__init__()
        self.wave = 'db2'
        self.mode = 'zero'
        self.J = 1

    def forward(self, img):
        """
        Args:
            img (torch Tensor): Tensor to be decomposed.

        Returns:
            Tensor: 
        """
        xfm = DWTForward(J=self.J, mode=self.mode, wave=self.wave).requires_grad_(False)
        #x = torch.tensor(img, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        while len(img.shape)<4:
            img = img.unsqueeze(0)

        yl, yh = xfm(img)
        
        return yl.squeeze(0)


class median_filter:
    """Remove skull by applying morphological operations."""

    def __init__(self):
        super().__init__()        
        self.kernel = np.array([
                        [0,1,0],
                        [1,1,1],
                        [0,1,1]
                        ], np.uint8)

    def __call__(self, x):
        copy = np.array(x.squeeze()*255, np.uint8)
        median_img = cv2.medianBlur(copy, 3)

        return median_img

class exapand_channel(torch.nn.Module):
    """."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        
        return x.repeat(1, 3, 1, 1)