import os
import sys
import pandas as pd
from src.model_training import train_and_compare_models

DATA_PATH = os.path.join("data", "tweets.csv")


def main():
    print("==================================================")
    print("  TweetSense - Model Training & Evaluation Pipeline")
    print("==================================================")

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Training dataset not found at '{DATA_PATH}'.")
        sys.exit(1)

    print(f"\n[1/5] Loading training dataset from '{DATA_PATH}'...")
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"      Successfully loaded {len(df)} total dataset rows.")
    except Exception as e:
        print(f"[ERROR] Failed to read CSV dataset: {e}")
        sys.exit(1)

    print("\n[2/5] Validating dataset columns and labels...")
    if "tweet" not in df.columns or "sentiment" not in df.columns:
        print("[ERROR] Dataset must contain 'tweet' and 'sentiment' columns.")
        sys.exit(1)

    print("\n[3/5] Preprocessing text and performing stratified train/test split...")
    print("[4/5] Extracting TF-IDF features (unigrams & bigrams)...")
    print("[5/5] Training candidate models (Logistic Regression, Naive Bayes, Linear SVM)...")

    try:
        primary_name, metrics, comparison, metadata = train_and_compare_models(df, text_col="tweet", label_col="sentiment")
        
        print("\n==================================================")
        print("           MODEL EVALUATION SUMMARY               ")
        print("==================================================")
        print(f" Selected Model : {primary_name}")
        print(f" Total Samples  : {metadata['total_samples']} (Train: {metadata['train_samples']}, Test: {metadata['test_samples']})")
        print(f" Accuracy       : {metrics['accuracy']:.4f}")
        print(f" Precision      : {metrics['precision']:.4f}")
        print(f" Recall         : {metrics['recall']:.4f}")
        print(f" F1-Score       : {metrics['f1_score']:.4f}")
        print("--------------------------------------------------")
        print(" Multi-Model Comparison Table:")
        for model_name, res in comparison.items():
            print(f"  - {model_name:<25} | Acc: {res['accuracy']:.4f} | F1: {res['f1_score']:.4f}")
        print("--------------------------------------------------")
        print(" [SUCCESS] Trained model and vectorizer saved in 'models/' directory.")
        print("==================================================\n")

    except Exception as e:
        print(f"[ERROR] Model training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
