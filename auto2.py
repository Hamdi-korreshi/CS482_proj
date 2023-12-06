import matplotlib.pyplot as plt
from pgvector.psycopg import register_vector
import psycopg
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
import matplotlib
import os
from PIL import Image
import numpy as np
matplotlib.use("TkAgg")

class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image = Image.open(image_file).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

def generate_embeddings(inputs):
    inputs = inputs.view(-1, 3, 224, 224)
    return model(inputs.to(device)).detach().cpu().numpy()

def show_images(dataset_images):
    grid = torchvision.utils.make_grid(dataset_images)
    grid = grid.permute(1, 2, 0).numpy()
    img = (grid / 2 + 0.5)
    plt.imshow(img)
    plt.draw()
    plt.waitforbuttonpress(timeout=3)

video_folder = "/workspaces/CS482_proj/Videos"
preprocessed_folder = "Frames"
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.Lambda(lambda x: (x / 255).unsqueeze(0)),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# dataset = VideoFrameDataset(root=preprocessed_folder)
dataset = ImageFolderDataset(preprocessed_folder, transform=transform)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = torchvision.models.resnet18(weights='DEFAULT')
model.fc = torch.nn.Identity()
model.to(device)
model.eval()
seed = True

conn = psycopg.connect(dbname='postgres', host="localhost", port=5432)
conn.execute('CREATE EXTENSION IF NOT EXISTS vector;')
register_vector(conn)

curr = conn.cursor()

# generate, save, and index embeddings
if seed:
    conn.execute('DROP TABLE IF EXISTS image;')
    conn.execute('CREATE TABLE image (id bigserial PRIMARY KEY, embedding vector(512));')
    count = 1
    print('Generating embeddings')
    for data in tqdm(data_loader):
        embeddings = generate_embeddings(data[0])
        np.save(f'img_embeddings/embeddings{count}.npy', embeddings)
        sql = 'INSERT INTO image (embedding) VALUES ' + ','.join(['(%s)' for _ in embeddings])
        params = [embedding for embedding in embeddings]
        conn.execute(sql, params)
        count+=1

images = next(iter(data_loader))[0]

embeddings = generate_embeddings(images)
for image, embedding in zip(images, embeddings):
    result = conn.execute('SELECT id FROM image ORDER BY embedding <=> %s LIMIT 15;', (embedding,)).fetchall()
    print("Found: ", result)