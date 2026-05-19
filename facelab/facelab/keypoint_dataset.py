import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class KeypointDataset(Dataset):
    def __init__(self, images, keypoints, transform=None):
        self.images = images
        self.keypoints=[list(map(tuple, keypoint)) for keypoint in keypoints]
        self.transform = transform

    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):

        image = self.images[idx]
        keypoints = self.keypoints[idx]
        keypoints = np.array(keypoints).reshape(-1, 2)  # Reshape into (num_keypoints, 2)
        heatmaps = self.generate_heatmap(keypoints, 256, 256)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize
        heatmaps = torch.tensor(heatmaps, dtype=torch.float32)

        return {"image": image, "heatmaps": heatmaps, "keypoints": keypoints}


    def generate_heatmap(self,keypoints, height, width, sigma=2):
        num_keypoints = len(keypoints)
        heatmaps = np.zeros((num_keypoints, height, width), dtype=np.float32)


        for i, (x, y) in enumerate(keypoints):
            if x < 0 or y < 0 or x >= width or y >= height:
                continue  # Ignore invalid keypoints
            heatmap = np.zeros((height, width), dtype=np.float32)
            cv2.circle(heatmap, (int(x), int(y)), radius=2, color=1, thickness=-1)
            heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigma)
            # plt.imshow(heatmap)
            # plt.title(f"{i,x,y}")
            # plt.show()
            heatmaps[i] = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap

        return heatmaps

