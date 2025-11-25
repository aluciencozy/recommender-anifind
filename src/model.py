import torch
import torch.nn as nn


class RecommenderAutoencoder(nn.Module):
    """
    Simple autoencoder for collaborative filtering.
    Each input is a full user rating vector.
    """

    def __init__(self, num_anime, hidden_dim=128):
        """
        Args:
            num_anime (int): number of anime (input dimension)
            hidden_dim (int): size of bottleneck layer
        """
        super().__init__()

        # Encoder: compresses full rating vector to hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(num_anime, hidden_dim),
            nn.ReLU()
        )

        # Decoder: reconstructs full rating vector
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, num_anime)
            # Note: no activation here; we'll use MSELoss directly
        )

    def forward(self, x):
        """
        Forward pass through autoencoder
        """
        z = self.encoder(x)  # compress
        out = self.decoder(z)  # reconstruct
        return out
