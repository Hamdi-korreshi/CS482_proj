import os
import re
import numpy as np
from gensim.models import KeyedVectors

# Directory containing .txt files
dir_path = 'Videos'

# Load the downloaded model
model = KeyedVectors.load('word2vec-google-news-300.model')
print("Model loaded successfully.")

# Read each .txt file and create a list of sentences
sentences = []
for filename in os.listdir(dir_path):
    if filename.endswith('.txt'):
        with open(os.path.join(dir_path, filename), 'r') as f:
            lines = f.readlines()
            for line in lines:
                # Ignore lines with timestamps or empty lines
                if not re.match(r'^\d+.*-->.*$', line.strip()) and line.strip():
                    sentences.append(line.split())
print(f"Finished reading {len(sentences)} sentences from the text files.")

# Create embeddings for each sentence
embeddings = []
for i, sentence in enumerate(sentences):
    # Filter out words that are not in the model's vocabulary
    sentence = [word for word in sentence if word in model.key_to_index]
    if sentence:
        # Average the vectors of the individual words in the sentence
        embeddings.append(np.mean([model[word] for word in sentence], axis=0))
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1} sentences.")
print("Finished creating embeddings for all sentences.")

# Create a directory to save the embeddings
os.makedirs('text-embeddings', exist_ok=True)
print("Created directory for embeddings.")

# Save each embedding to a .npy file
for i, embedding in enumerate(embeddings):
    np.save(f'text-embeddings/embedding_{i}.npy', embedding)
print("Finished saving all embeddings.")