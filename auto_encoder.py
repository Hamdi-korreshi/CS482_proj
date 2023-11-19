import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms

# Define the Autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Use Sigmoid to get values between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Initialize the autoencoder and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
model.load_state_dict(torch.load('autoencoder.pth'))  # Load pretrained model
model.eval()

folder = "/home/hamdi/CS482_proj/Videos"
prep = "/home/hamdi/CS482_proj/Frames"
all_count = 0
if not os.path.exists(prep):
    os.mkdir(prep)
    print("MADE")
for video in os.listdir(folder):
    if video.endswith('.mp4'):
        print(f"Preprocessing video file: {video}")
        video_path = os.path.join(folder, video)
        cap = cv2.VideoCapture(video_path)
        count = 0
        embeddings = []
        # read frame loop
        while True:
            ret, frame = cap.read()
        # Break the loop if no more frames are available
            if not ret:
                break
            # every frame 90th frame because its gonna be way too big othewise and now we are grabbing ever 3 seconds or so
            # int is needed so my frames arent 257.9888887877678
            if count % 90 == 0:
                frame_filename = os.path.join(prep, f"frame{int(all_count / 90)}.jpg")
                resized_frame = cv2.resize(frame, (224, 224))
                cv2.imwrite(frame_filename, frame)
                # Convert frame to tensor and normalize
                frame_tensor = transforms.ToTensor()(resized_frame).unsqueeze(0).to(device)
                frame_tensor = frame_tensor / 255.0
                # Pass frame through autoencoder
                embedding, _ = model(frame_tensor)
                embeddings.append(embedding.squeeze().detach().cpu())
            count += 1
            all_count += 1
        # Combine embeddings
        combined_embedding = torch.stack(embeddings).mean(dim=0)
        torch.save(combined_embedding, os.path.join(prep, f"{video}_embedding.pt"))
        # you the release otherwise it courrpts you're docker and or whole ipynb fil :)
        cap.release()
        print(count, "frames saved ")
print("Done")
