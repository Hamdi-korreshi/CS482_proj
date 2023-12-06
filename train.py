import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
import matplotlib.pyplot as plt

# Define the vision encoder (image model)
class VisionModel(nn.Module):
    def __init__(self):
        super(VisionModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, 512)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

vision_model = VisionModel()

# Define the text encoder (language model)
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LanguageModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 512)

    def forward(self, x):
        x = self.embed(x)
        lstm_out, _ = self.lstm(x)
        x = self.linear(lstm_out[:, -1, :])
        return x

language_model = LanguageModel(10000,256,512,2)

class ImageCaptionDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.captions = self._load_captions(captions_file)

    def _load_captions(self, captions_file):
        with open("flickr8k/captions.txt", 'r') as f:
            captions = [line.strip().split(',', 1) for line in f.readlines()]
        return captions


    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_id, caption = self.captions[idx]
        image_file = os.path.join(self.root_dir, image_id)
        image = Image.open(image_file).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, caption


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1):
        super().__init__()
        self.temperature = temperature

    def forward(self, img_emb, text_emb):
        img_emb = F.normalize(img_emb, dim=1)
        text_emb = F.normalize(text_emb, dim=1)
        return 2 - 2 * (img_emb * text_emb).sum(dim=1)

class DualEncoder(nn.Module):
    def __init__(self, text_model, img_model, temperature=1):
        super().__init__()
        self.text_model = text_model
        self.img_model = img_model
        self.temperature = temperature

    def forward(self, image, caption):
        img_emb = self.img_model(image)
        text_emb = self.text_model(caption)
        return img_emb, text_emb

# Define your transformations here
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create the dataset
dataset = ImageCaptionDataset(root_dir='flickr8k/Images', captions_file='flickr8k/captions.txt', transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
data_train, data_val = random_split(dataset, [train_size, val_size])

# Create the data loaders
batch_size = 64
train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=data_val, batch_size=batch_size, shuffle=False)

# Create the model
model = DualEncoder(language_model, vision_model)

# Define the loss function and the optimizer
criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters())

# Training loop
num_epochs = 10
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    train_loss = 0.0
    val_loss = 0.0

    model.train()
    for images, captions in train_loader:
        optimizer.zero_grad()
        img_emb, text_emb = model(images, captions)
        loss = criterion(img_emb, text_emb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    with torch.no_grad():
        for images, captions in val_loader:
            img_emb, text_emb = model(images, captions)
            loss = criterion(img_emb, text_emb)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}')

# Plot the training and validation loss
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()