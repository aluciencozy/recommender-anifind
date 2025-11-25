import torch
from torch.utils.data import Dataset, DataLoader
import scipy.sparse


class AutoencoderDataset(Dataset):
    """
    PyTorch Dataset for autoencoder-based recommender.
    Each item is a user's full rating vector.
    """

    def __init__(self, sparse_matrix_path):
        """
        Args:
            sparse_matrix_path (str): Path to the .npz sparse rating matrix
        """
        # Load the sparse COO matrix
        coo = scipy.sparse.load_npz(sparse_matrix_path)

        # Convert to dense PyTorch tensor (FloatTensor)
        # Each row = one user's ratings
        self.data = torch.FloatTensor(coo.toarray())

    def __len__(self):
        """
        Returns number of users in dataset
        """
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        Returns full rating vector for one user
        """
        return self.data[idx]


def create_dataloader(
    sparse_matrix_path, batch_size=256, shuffle=True, pin_memory=True
):
    """
    Builds a PyTorch DataLoader for the autoencoder.

    Args:
        sparse_matrix_path (str): path to sparse matrix .npz
        batch_size (int): number of users per batch
        shuffle (bool): shuffle users each epoch
        pin_memory (bool): optimize memory transfer to GPU (ignored if using CPU)

    Returns:
        DataLoader
    """
    dataset = AutoencoderDataset(sparse_matrix_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
    )
    return loader
