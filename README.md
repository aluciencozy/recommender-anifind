# Recommender-Anifind

A collaborative filtering anime recommender system using a PyTorch Autoencoder. This project suggests Top-N anime to users based on historical ratings, using popularity filters to ensure high-quality recommendations.

## Features
- **PyTorch Autoencoder:** Efficient collaborative filtering model.
- **Smart Filtering:** Removes niche/unrated noise by focusing on popular titles.
- **Dual Interface:** Includes a simple CLI and a Jupyter Notebook demo.

## Installation & Setup

1. **Clone and Install**
   ```bash
   git clone <repo_url>
   cd recommender-anifind
   
   # Create and activate virtual environment
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Download Required Files**
   - **Model:** Place `model_autoencoder_20251123_190735.pth` in the project root.
   - **Data:** Download the [Kaggle Anime Dataset] https://www.kaggle.com/datasets/dbdmobile/myanimelist-dataset?resource=download and place `anime-dataset-2023.csv` and `users-score-2023.csv` into the `data/` folder.

3. **Build Data Mappings**
   Run the build script to generate the sparse matrix and mapping files:
   ```bash
   python src/build_sparse.py
   ```

## Usage

### CLI
Generate recommendations for a specific user.
```bash
# Usage: python src/cli.py --user_idx <ID> --num_recommendations <N>
python src/cli.py --user_idx 0 --num_recommendations 10
```

### Notebook
For an interactive demo, open the notebook:
```bash
jupyter notebook notebooks/demo_recommender_anifind.ipynb
```

## Requirements
- Python 3.10+
- PyTorch
- pandas, scipy