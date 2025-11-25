import torch 
import pandas as pd
import scipy.sparse

from model import RecommenderAutoencoder


anime_map = pd.read_csv('data/anime_mapping.csv')
user_map = pd.read_csv('data/user_mapping.csv')
anime_full = pd.read_csv('data/anime-dataset-2023.csv')

R_sparse = scipy.sparse.load_npz('data/ratings_sparse_coo.npz')
R_dense = torch.FloatTensor(R_sparse.toarray())

anime_counts = R_dense.gt(0).sum(dim=0)
anime_map['num_ratings'] = anime_counts.tolist()

num_users, num_anime = R_dense.shape

model = RecommenderAutoencoder(num_anime, hidden_dim=128)
model.load_state_dict(torch.load('model_autoencoder_20251123_190735.pth', map_location="cpu"))
model.eval()


def recommend(user_idx, N=10, popularity_threshold=1000) -> pd.DataFrame:
    """
    Generate top-N anime recommendations for a given user index.

    Args:
        user_idx (int): index of user to recommend for
        N (int): number of top recommendations to return
        popularity_threshold (int): only recommend anime with at least this many ratings

    Returns:
        pd.DataFrame with columns: anime_id, title, predicted_score
    """

    user_tensor = R_dense[user_idx].unsqueeze(0)
    with torch.no_grad():
        predicted = model(user_tensor)[0]

    true_ratings = R_dense[user_idx]
    popularity_mask = torch.tensor(anime_map['num_ratings'].values) >= popularity_threshold
    mask = (true_ratings == 0) & popularity_mask  # only recommend unrated & popular anime
    filtered = predicted * mask.float()  # zero-out anime already rated

    top_indices = torch.argsort(filtered, descending=True)[:N]

    records = []
    for idx in top_indices.tolist():
        anime_id = anime_map.iloc[idx]["anime_id"]

        row = anime_full[anime_full["anime_id"] == anime_id]
        if not row.empty:
            title = row.iloc[0]["Name"]
        else:
            title = "(Title not found)"

        score = float(predicted[idx].item())

        records.append({
            "anime_id": anime_id,
            "title": title,
            "predicted_score": score
        })

    return pd.DataFrame(records)


def main():
    print("=== Anime Recommender CLI ===")
    print(f"Total users: {num_users}, Total anime: {num_anime}")

    while True:
        user_input = input("\nEnter a user index (or 'q' to quit): ")

        if user_input.lower() == "q":
            print("Goodbye!")
            break

        if not user_input.isdigit():
            print("Please enter a valid integer user index.")
            continue

        user_idx = int(user_input)
        if user_idx < 0 or user_idx >= num_users:
            print(f"User index must be between 0 and {num_users-1}.")
            continue

        N = input("How many recommendations do you want? (default 10): ")
        N = int(N) if N.isdigit() else 10

        threshold = input("Minimum number of ratings for popularity filter? (default 1000): ")
        threshold = int(threshold) if threshold.isdigit() else 1000

        df = recommend(user_idx=user_idx, N=N, popularity_threshold=threshold)
        print("\nTop Recommendations:")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
