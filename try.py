import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

class ProjectEmbeddings(nn.Module):
    def __init__(self, input_dim, num_projection_layers, projection_dims, dropout_rate):
        super(ProjectEmbeddings, self).__init__()
        self.fc1 = nn.Linear(input_dim, projection_dims)
        self.dropout = nn.Dropout(dropout_rate)
        self.projection_layers = nn.ModuleList([
            nn.Linear(projection_dims, projection_dims) for _ in range(num_projection_layers)
        ])
        self.layer_norm = nn.LayerNorm(projection_dims)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        for layer in self.projection_layers:
            residual = x
            x = layer(x)
            x = self.dropout(x)
            x = residual + x
            x = self.layer_norm(x)
        return x

class DualEncoder(nn.Module):
    def __init__(self, text_encoder, image_encoder, temperature=1.0):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.temperature = temperature
        self.loss_tracker = nn.MSELoss()

    def forward(self, features):
        # the keras model had these encoders here but i have the embeddings since I can grab them directly from files
        caption_embeddings = torch.load('caption_embeddings.pth')
        image_embeddings = torch.load('image_embeddings.pth')
        return caption_embeddings, image_embeddings

    def compute_loss(self, caption_embeddings, image_embeddings):
        # logits[i][j] is the dot_similarity(caption_i, image_j).
        logits = torch.mm(caption_embeddings, image_embeddings.t()) / self.temperature
        # images_similarity[i][j] is the dot_similarity(image_i, image_j).
        images_similarity = torch.mm(image_embeddings, image_embeddings.t())
        # captions_similarity[i][j] is the dot_similarity(caption_i, caption_j).
        captions_similarity = torch.mm(caption_embeddings, caption_embeddings.t())
        # targets[i][j] = avarage dot_similarity(caption_i, caption_j) and dot_similarity(image_i, image_j).
        targets = F.softmax((captions_similarity + images_similarity) / (2 * self.temperature), dim=-1)
        # Compute the loss for the captions using crossentropy
        captions_loss = F.cross_entropy(logits, targets.argmax(dim=-1))
        # Compute the loss for the images using crossentropy
        images_loss = F.cross_entropy(logits.t(), targets.argmax(dim=-1))
        # Return the mean of the loss over the batch.
        return (captions_loss + images_loss) / 2

    def training_step(self, features):
        # Forward pass
        caption_embeddings, image_embeddings = self(features)
        loss = self.compute_loss(caption_embeddings, image_embeddings)
        return loss

    def validation_step(self, features):
        caption_embeddings, image_embeddings = self(features)
        loss = self.compute_loss(caption_embeddings, image_embeddings)
        return loss

# for reference later on
# project_model = ProjectEmbeddings(input_dim=300, num_projection_layers=2, projection_dims=50, dropout_rate=0.1)

# folder_path = "/workspaces/CS482_proj/text-embeddings"
# save_path = "/workspaces/CS482_proj/Projected_text"
# folder_img_path = "/workspaces/CS482_proj/text-embeddings"
# save_path2 = "/workspaces/CS482_proj/Projected_img"
# if not os.path.exists(save_path):
#             os.mkdir(save_path)
# if not os.path.exists(save_path2):
#     os.mkdir(save_path2)
# # makes the text projections 
# for filename in os.listdir(folder_path):
#     # Check if the file is a .npy file
#     if filename.endswith('.npy'):
#         # load the numpy array then the embedding
#         embedding = np.load(os.path.join(folder_path, filename))
#         embedding = torch.from_numpy(embedding)
        
#         # Apply the model to the embedding
#         projected_embedding = project_model(embedding)

#         # Save the projected embedding
#         torch.save(projected_embedding, os.path.join(save_path, 'projected_' + filename))

# img_model = ProjectEmbeddings(input_dim=512, num_projection_layers=2, projection_dims=50, dropout_rate=0.1)

# #makes the image projections
# for filename in os.listdir(folder_img_path):
#     # Check if the file is a .npy file
#     if filename.endswith('.npy'):
#         # load the numpy array then the embedding
#         embedding = np.load(os.path.join(folder_path, filename))
#         embedding = torch.from_numpy(embedding)
        
#         # Apply the model to the embedding
#         projected_embedding = project_model(embedding)

#         # Save the projected embedding
#         torch.save(projected_embedding, os.path.join(save_path2, 'projected_' + filename))
