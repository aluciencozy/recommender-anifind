import os
import pandas as pd
from scipy.sparse import coo_matrix, save_npz


DATA_PATH = os.path.join("data", "users-score-2023.csv")
SPARSE_OUT = os.path.join("data", "ratings_sparse_coo.npz")
USER_MAP_OUT = os.path.join("data", "user_mapping.csv")
ANIME_MAP_OUT = os.path.join("data", "anime_mapping.csv")


def load_dataframe(path):
    print(f"Loading cleaned df from: {path}")
    df = pd.read_csv(path)

    df = df[["user_id", "anime_id", "rating"]]

    return df


def build_mapping(df):
    """
    Create dense integer indices:
        user_id  -> user_idx
        anime_id -> anime_idx

    We return:
        df with new dense columns
        and two mapping DataFrames.
    """
    
    # Adds user_idx and anime_idx columns to df
    # Also returns unique user and anime IDs e.g., ["100", "200", ...]
    df["user_idx"], user_unique = pd.factorize(df["user_id"])
    df["anime_idx"], anime_unique = pd.factorize(df["anime_id"])
    
    # Create mapping DataFrames. See in data folder for examples
    user_map = pd.DataFrame({
        "user_idx": range(len(user_unique)),
        "user_id": user_unique,
    })

    anime_map = pd.DataFrame({
        "anime_idx": range(len(anime_unique)),
        "anime_id": anime_unique,
    })
    
    print(f"Users: {len(user_unique)}, Anime: {len(anime_unique)}")
    return df, user_map, anime_map


def build_sparse_matrix(df):
    """
    Convert df[user_idx, anime_idx, rating] into a SciPy COO sparse matrix.
    """

    print("Building sparse matrix...")

    rows = df["user_idx"].to_numpy()
    cols = df["anime_idx"].to_numpy()
    data = df["rating"].to_numpy()

    num_users = df["user_idx"].max() + 1
    num_anime = df["anime_idx"].max() + 1

    sparse_matrix = coo_matrix((data, (rows, cols)), shape=(num_users, num_anime))

    print(f"Matrix shape: {sparse_matrix.shape}")
    print(f"Nonzero ratings: {sparse_matrix.nnz}")

    return sparse_matrix


def main():
    df = load_dataframe(DATA_PATH)
    
    df, user_map, anime_map = build_mapping(df)
    
    sparse = build_sparse_matrix(df)

    print(f"Saving sparse matrix to: {SPARSE_OUT}")
    save_npz(SPARSE_OUT, sparse)

    print(f"Saving user mapping to: {USER_MAP_OUT}")
    user_map.to_csv(USER_MAP_OUT, index=False)

    print(f"Saving anime mapping to: {ANIME_MAP_OUT}")
    anime_map.to_csv(ANIME_MAP_OUT, index=False)


if __name__ == "__main__":
    main()