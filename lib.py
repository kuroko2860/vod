import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import cv2
from albumentations.pytorch import ToTensorV2
import albumentations as A
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
