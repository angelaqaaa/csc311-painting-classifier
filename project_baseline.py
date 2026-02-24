"""
This Python file provides some useful code for reading the training file
"clean_dataset.csv". You may adapt this code as you see fit. However,
keep in mind that the code provided does only basic feature transformations
to build a rudimentary kNN model in sklearn. Not all features are considered
in this code, and you should consider those features! Use this code
where appropriate, but don't stop here!
"""
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Helper function to parse ratings (e.g., "4 - Agree" -> 4)
def extract_rating(response):
    if pd.isna(response):
        return None
    match = re.match(r'^(\d+)', str(response))
    return int(match.group(1)) if match else None

def main():
    # 1. Load Data
    df = pd.read_csv("clean_dataset.csv")

    # 2. Define Features and Target
    target_col = "Painting"
    feature_cols = [
        "On a scale of 1–10, how intense is the emotion conveyed by the artwork?",
        "This art piece makes me feel sombre.",
        "This art piece makes me feel content.",
        "This art piece makes me feel calm.",
        "This art piece makes me feel uneasy.",
        "How many prominent colours do you notice in this painting?",
        "How many objects caught your eye in the painting?"
    ]

    # 3. Preprocessing
    # Apply extract_rating to the Likert scale columns
    likert_cols = [
        "This art piece makes me feel sombre.",
        "This art piece makes me feel content.",
        "This art piece makes me feel calm.",
        "This art piece makes me feel uneasy."
    ]
    
    df_processed = df.copy()
    for col in likert_cols:
        df_processed[col] = df_processed[col].apply(extract_rating)

    # Select necessary columns and drop rows with missing values
    df_model = df_processed[[target_col] + feature_cols].dropna()

    X = df_model[feature_cols].values
    y = df_model[target_col].values

    # 4. Split Data (50% Train, 20% Val, 30% Test)
    # First, split off the 30% Test set
    X_remaining, X_test, y_remaining, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Then split the remaining 70% into Train (50% total) and Val (20% total)
    # 20% is approx 28.57% of the remaining 70%
    val_split_ratio = 0.20 / 0.70
    X_train, X_val, y_train, y_val = train_test_split(
        X_remaining, y_remaining, test_size=val_split_ratio, random_state=42, stratify=y_remaining
    )

    print(f"Data Splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 5. Find Best k using Validation Set
    best_k = 1
    best_val_acc = 0
    
    for k in range(1, 31):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        val_acc = knn.score(X_val, y_val)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_k = k
            
    print(f"Best k found: {best_k}")

    # 6. Apply to Test Set and Report Accuracies
    final_knn = KNeighborsClassifier(n_neighbors=best_k)
    final_knn.fit(X_train, y_train)

    train_acc = final_knn.score(X_train, y_train)
    val_acc = final_knn.score(X_val, y_val)
    test_acc = final_knn.score(X_test, y_test)

    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()