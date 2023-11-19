import matplotlib.pyplot as plt
from pgvector.psycopg import register_vector
import psycopg
import tempfile
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import Adam
from tqdm import tqdm
import cv2

class VideoFrameDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.frame_files = [f for f in os.listdir(root) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx):
        frame_file = self.frame_files[idx]
        frame_path = os.path.join(self.root, frame_file)
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Convert to tensor
        frame = transforms.ToTensor()(frame)

        # Normalize
        mean = torch.mean(frame)
        std = torch.std(frame)
        frame = transforms.Normalize(mean=[mean.item()]*3, std=[std.item()]*3)(frame)
        frame = frame/255
        return frame

video_folder = "/workspaces/CS482_proj/Videos"
preprocessed_folder = "Frames"
dataset = VideoFrameDataset(root=preprocessed_folder)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

seed = True

conn = psycopg.connect(dbname='postgres', host="localhost", port=5432)
conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)