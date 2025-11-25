import torch
import torch.nn as nn
from torch.optim import Adam
from dataset import create_dataloader, AutoencoderDataset
from model import RecommenderAutoencoder
import scipy.sparse
from datetime import datetime


sparse_matrix_path = "data/ratings_sparse_coo.npz"

batch_size = 256
hidden_dim = 128
learning_rate = 1e-3
num_epochs = 5

dataloader = create_dataloader(
    sparse_matrix_path,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True
)

dataset = AutoencoderDataset(sparse_matrix_path)
num_users, num_anime = dataset.data.shape
print(f"Users: {num_users}, Anime: {num_anime}")

model = RecommenderAutoencoder(num_anime=num_anime, hidden_dim=hidden_dim)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in dataloader:
        inputs = batch
        optimizer.zero_grad()
        outputs = model(inputs)

        mask = inputs > 0
        loss = criterion(outputs[mask], inputs[mask])

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"model_autoencoder_{timestamp}.pth"
torch.save(model.state_dict(), model_filename)
print(f"Model saved to {model_filename}")
