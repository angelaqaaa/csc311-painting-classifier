"""
Softmax regression painting classifier, inference only.
Loads saved weights, intercept, and preprocessing parameters from the
development notebook, preprocesses new test data identically,
and computes softmax predictions using numpy.
"""
import os
import re
import json
import numpy as np
import pandas as pd


# ---- Paths (relative to this script, not the working directory) ----
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARAMS_PATH = os.path.join(_SCRIPT_DIR, "softmax_params.json")


# ---- Load artifacts once at module level ----
with open(_PARAMS_PATH) as _f:
    _PARAMS = json.load(_f)

_WEIGHTS = np.load(os.path.join(_SCRIPT_DIR, "softmax_weights.npy"))   # (3, D)
_INTERCEPT = np.load(os.path.join(_SCRIPT_DIR, "softmax_intercept.npy"))  # (3,)


# ---- Helper functions ----

def extract_rating(response):
    """Parse a Likert-scale response string into an integer."""
    if pd.isna(response):
        return None
    match = re.match(r'^(\d+)', str(response))
    return int(match.group(1)) if match else None


def clean_pay_value(val):
    """Parse the willingness-to-pay field into a float."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    s = re.sub(r'[$]', '', s)
    s = re.sub(r'\b(cad|dollars?|bucks?)\b', '', s)
    s = s.replace(',', '').strip()
    s = re.sub(r'\.$', '', s)
    try:
        return float(s)
    except ValueError:
        return np.nan


# ---- Preprocessing functions ----

def preprocess_numeric(df, params):
    """Preprocess numeric and Likert features using saved parameters."""
    num_cols = params["num_cols"]
    likert_cols = params["likert_cols"]
    pay_col = params["pay_col"]
    eps = params["epsilon"]
    numeric_params = params["preprocessing"]["numeric"]
    all_cols = num_cols + likert_cols

    result = pd.DataFrame(index=df.index)

    for col in num_cols:
        if col == pay_col:
            cleaned = df[col].apply(clean_pay_value)
            # Clip negatives to 0, then log1p to compress extreme outliers
            cleaned = cleaned.clip(lower=0)
            result[col] = np.log1p(cleaned)
        else:
            result[col] = pd.to_numeric(df[col], errors="coerce")

    for col in likert_cols:
        result[col] = df[col].apply(extract_rating).astype(float)

    for col in all_cols:
        p = numeric_params[col]
        result[col] = result[col].fillna(p["median"])
        result[col] = (result[col] - p["mean"]) / (p["std"] + eps)

    return result[all_cols].values


def preprocess_categorical(df, params):
    """Multi-hot encode categorical features using saved category lists."""
    cat_cols = params["cat_cols"]
    cat_params = params["preprocessing"]["categorical"]

    arrays = []
    for col in cat_cols:
        cats = cat_params[col]
        cat_to_idx = {c: i for i, c in enumerate(cats)}
        n = len(df)
        d = len(cats)
        hot = np.zeros((n, d))

        for i, val in enumerate(df[col].values):
            if pd.isna(val):
                continue
            for item in str(val).split(","):
                item = item.strip()
                if item in cat_to_idx:
                    hot[i, cat_to_idx[item]] = 1.0

        arrays.append(hot)

    return np.hstack(arrays)


def preprocess_text(df, params):
    """Compute TF-IDF features using saved vocabulary and IDF vectors.

    Stop words (English) were filtered during vocabulary fitting in the
    training notebook via sklearn's TfidfVectorizer(stop_words="english").
    They are excluded here implicitly: any word not in the saved vocabulary
    is simply ignored, so no explicit stop-word list is needed at inference.
    """
    text_cols = params["text_cols"]
    text_params = params["preprocessing"]["text"]

    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    arrays = []

    for col in text_cols:
        vocab = text_params[col]["vocabulary"]
        idf = np.array(text_params[col]["idf"])
        d = len(vocab)
        texts = df[col].fillna("").astype(str).values
        n = len(texts)

        tf_matrix = np.zeros((n, d))
        for i, text in enumerate(texts):
            words = token_pattern.findall(text.lower())
            if len(words) == 0:
                continue
            for word in words:
                if word in vocab:
                    tf_matrix[i, vocab[word]] += 1

        tfidf = tf_matrix * idf
        row_norms = np.sqrt(np.sum(tfidf ** 2, axis=1, keepdims=True))
        row_norms = np.maximum(row_norms, 1e-10)
        tfidf = tfidf / row_norms

        arrays.append(tfidf)

    return np.hstack(arrays)


def build_test_features(df, params):
    """Build the complete feature matrix for test data."""
    X_num = preprocess_numeric(df, params)
    X_cat = preprocess_categorical(df, params)
    X_text = preprocess_text(df, params)
    return np.hstack([X_num, X_cat, X_text])


# ---- Softmax prediction ----

def softmax_predict(X, weights, intercept):
    """
    Predict classes using the trained softmax regression model.

    Parameters:
        X         - feature matrix, shape (N, D)
        weights   - weight matrix, shape (K, D) where K is the number of classes
        intercept - intercept vector, shape (K,)

    Returns: predicted class indices, shape (N,)
    """
    logits = X @ weights.T + intercept
    # softmax(logits) is monotonic, so argmax(logits) == argmax(softmax(logits));
    # we skip the exp/normalize step since only the predicted class is needed.
    return np.argmax(logits, axis=1)


# ---- Main prediction function ----

def predict_all(filename):
    """
    Make predictions for the data in filename.

    Parameters:
        filename - path to a CSV file with survey responses

    Returns: list of str, predicted painting names, one per row
    """
    df = pd.read_csv(filename)

    X_test = build_test_features(df, _PARAMS)

    y_pred = softmax_predict(X_test, _WEIGHTS, _INTERCEPT)

    label_map = _PARAMS["label_map"]
    predictions = [label_map[str(y)] for y in y_pred]

    return predictions
