import torch
import cv2
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n, m = 256, 256
n0, m0 = 260, 346

eps = 1e-16
pol_list = [1.0, -1.0]

def create_of_img(flow, mask):
    flow_map = flow.squeeze().cpu().detach().numpy()
    mask_map = torch.cat((mask.unsqueeze(0), mask.unsqueeze(0)), dim=0).squeeze().cpu().detach().numpy()
    flow_map_mask = flow_map * mask_map
    flow_magnitude_mask, flow_angle_mask = cv2.cartToPolar(flow_map_mask[0, :, :], flow_map_mask[1, :, :])

    hsv_mask = np.zeros((flow_map_mask.shape[1], flow_map_mask.shape[2], 3), dtype=np.uint8)
    hsv_mask[..., 0] = flow_angle_mask * 180 / np.pi / 2 
    hsv_mask[..., 1] = 255 
    hsv_mask[..., 2] = cv2.normalize(flow_magnitude_mask, None, 0, 255, cv2.NORM_MINMAX)

    bgr_mask = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

    flow_magnitude, flow_angle = cv2.cartToPolar(flow_map[0, :, :], flow_map[1, :, :])

    hsv = np.zeros((flow_map.shape[1], flow_map.shape[2], 3), dtype=np.uint8)
    hsv[..., 0] = flow_angle * 180 / np.pi / 2 
    hsv[..., 1] = 255 
    hsv[..., 2] = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr_mask, bgr