from PIL import Image
from PIL import  ImagePalette
import  numpy as np
import cv2
import torch
x = torch.randn(1,21,100,100)
print x.shape
print (x.max(1))[1]
print np.random.random((x.shape[0], x.shape[1], 3)) * 255

