import cv2
import numpy as np

class median_filter:
    """median filter"""

    def __init__(self):
        super().__init__()        
        self.kernel = np.array([
                        [0,1,0],
                        [1,1,1],
                        [0,1,1]
                        ], np.uint8)

    def __call__(self, x):
        copy = np.array(x.squeeze()*255, np.uint8)
        median_img = cv2.medianBlur(copy, 5)
                
        return median_img