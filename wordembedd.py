# %%
# Read data.txt

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Read data.txt line by line
with open('data.txt', 'r') as f:
    lines = f.readlines()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# Create a dataclass from torch
from torch.utils.data import Dataset, DataLoader

# Create a class for the dataset
class WordEmbeddingDataset(Dataset):
    def __init__(self, lines, context_size=2):
        # Read data.txt line by line
        with open('data.txt', 'r') as f:
            self.lines = f.readlines()

        # Process lines
        self.process_lines()

        # Concatenate all lines
        self.text = ' '.join(self.lines)

        # Create a list of words
        self.words = self.text.split()
        self.vocab_size = len(set(self.words))


        # Create a dictionary of words
        # for one-hot encoding
        self.word2idx = {word: idx for idx, word in enumerate(set(self.words))}
        self.idx2word = {idx: word for idx, word in enumerate(set(self.words))}

        # Convert words to vectors with one-hot encoding
        self.words = [self.one_hot_encode(word) for word in tqdm(self.words, desc='One-hot encoding')]
        
        data = []
        target = []
        # Create a list of tuples
        # (next_word, [context_words])
        for i in tqdm(range(context_size, len(self.words) - context_size), desc='Preparing data'):
            context = []
            for j in range(i - context_size, i):
                context.append(self.words[j])
            data.append(context)
            target.append(self.words[i])

        # Convert to numpy array, this makes it faster aparently
        data = np.array(data)
        target = np.array(target)

        # Here data has the shape of (n_samples, context_size, vocab_size)
        # Target has the shape of (n_samples, vocab_size)
        # Let's reshape data to (n_samples, context_size * vocab_size)
        data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
            
        # Convert to torch tensor
        self.data = torch.tensor(data, dtype=torch.float32).to(device)
        self.target = torch.tensor(target, dtype=torch.float32).to(device)


    def process_lines(self):
        self.lines = [line.lower() for line in self.lines]
        self.lines = [line.replace('\n', '') for line in self.lines]
        self.lines = [''.join([c for c in line if c.isalnum() or c == ' ']) for line in self.lines]
            
    # One-hot encoding
    def one_hot_encode(self, word):
        x = np.zeros(len(self.word2idx))
        x[self.word2idx[word]] = 1
        return x

    def one_hot_decode(self, x):
        return self.idx2word[np.argmax(x)]

    def __len__(self):
        return len(self.words) -1
    
    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

    def get_all(self):
        return self.data, self.target

# %%
# Create an instance of the dataset
dataset = WordEmbeddingDataset(lines, context_size=5)

# %%
# Inicialize our model
from torch import nn

# Create a class for the model
class WordEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(WordEmbeddingModel, self).__init__()
        
        # This will be a NN with 3 layers
        # 1. Input layer
        # 2. Hidden layer
        # 3. Output layer

        # Input layer
        self.lay1 = nn.Linear(vocab_size * context_size, 64)
        # Activation function
        self.relu1 = nn.ReLU()

        self.lay2 = nn.Linear(64, embedding_dim)
        # Activation function
        self.relu2 = nn.ReLU()

        self.lay3 = nn.Linear(embedding_dim, vocab_size)
        # Activation function
        self.relu3 = nn.ReLU()

        # Output layer
        self.lay4 = nn.Linear(vocab_size, vocab_size)
        # Activation function
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.lay1(x)
        x = self.relu1(x)

        x = self.lay2(x)
        x = self.relu2(x)

        x = self.lay3(x)
        x = self.relu3(x)

        x = self.lay4(x)
        x = self.softmax(x)

        return x

# Create an instance of the model
# which will try to predict the next word
# given a word
model = WordEmbeddingModel(vocab_size=dataset.vocab_size, embedding_dim=10, context_size=5).to(device)

# Create a loss function
loss_fn = nn.MSELoss()

# Create an optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create a data loader
dataloader = DataLoader(dataset, batch_size=100, shuffle=False)

# Train the model
for epoch in range(1000):
    for i, batch in tqdm(enumerate(dataloader), desc='Training', total=len(dataloader), leave=True):
        data, target = batch
        
        # Forward pass
        y_pred = model(data)

        # Compute loss
        loss = loss_fn(y_pred, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch + 1}, Loss: {loss.item():.4f}')

    # Save
    torch.save(model.state_dict(), f'wordembedding_model_{epoch + 1}.pth')


# %%


# %%



