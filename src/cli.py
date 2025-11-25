import argparse
from predict_recommendations import recommend


def main():
    parser = argparse.ArgumentParser(description="Product Recommendation CLI")
    
    parser.add_argument("--user_idx", type=int, help="User ID to get recommendations for", required=True)
    parser.add_argument("--num_recommendations", type=int, default=10, help="Number of recommendations to return")
    
    args = parser.parse_args()
    
    recommendations = recommend(user_idx=args.user_idx, N=args.num_recommendations)
    print("Recommendations:", recommendations)
    
if __name__ == "__main__":
    main()