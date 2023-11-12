import torch
from torchvision.io import read_video
import torchvision.transforms.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams["savefig.bbox"] = "tight"

def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()
    plt.show()

v = 'Videos/Questions Remain Over Explosion At Gaza Hospital  NPR News Now.mp4'
print("Go into the mother load")
print(Path(v))
frames,_,_ = read_video(str(Path(v)), output_format="TCHW", start_pts=0, end_pts=2, pts_unit = 'sec')
img1_batch = torch.stack([frames[0], frames[1]])
plot(img1_batch)
