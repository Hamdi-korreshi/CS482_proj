import torch
from torchvision.io import read_video
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.utils import save_image
import numpy as np
from pathlib import Path
import os

v = 'Videos/Questions Remain Over Explosion At Gaza Hospital  NPR News Now.mp4'
folder = "Frames"
if not os.path.exists(folder):
    os.mkdir(folder)
print("Go into the mother load")
print(Path(v))
directory = 'Videos'
for filename in os.listdir(directory):
    if filename.endswith(".mp4"):  # specify the type of files, you can use a tuple of extensions
        frames,_,_ = read_video(str(Path(v)), output_format="TCHW", start_pts=0, end_pts=1, pts_unit = 'sec')
        frames = frames.float()
        mean = torch.mean(frames)
        std = torch.std(frames)
        norm = T.Normalize(mean, std)
        resized_1 = F.resize(frames, size=(224, 224))
        resized_1 -= mean
        resized_1 /= std
        r_and_n = norm(resized_1)
        arr = r_and_n.numpy()
        np.save(f'Frames/{filename}_frame.npy', arr)
