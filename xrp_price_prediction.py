"""
XRP Minute‑Level Price Prediction
================================

This script implements the core logic used in my CIS 661 term project, “A Ripple
Through Time: Predicting the Tides of XRP.”  It loads minute‑level historical
XRP/USD data, performs feature engineering, trains a Decision Tree regression
model to forecast the next minute’s closing price and augments the model with
K‑Means clustering to identify market regimes.  A simple live bot is included
that pulls fresh data from Binance’s public API and makes predictions in
real time.

Usage:
    python xrp_price_prediction.py

Requirements:
    - pandas
    - numpy
    - scikit‑learn
    - requests

The CSV file should contain at least the following columns:  'Open time',
 'Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume',
 'Number of trades', 'Taker buy base asset volume',
 'Taker buy quote asset volume'.
"""

from __future__ import annotations
import time
import requests
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def prepare_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Clean the raw DataFrame and engineer features.

    The function drops unneeded columns, converts timestamps,
    derives new features (returns, volatility, volume change, trade change), and
    constructs the target variable (next‑minute close).  Any rows with NaN or
    infinite values are removed.

    Returns
    -------
    X : DataFrame
        Features used for modelling.
    y : Series
        Target variable – next minute's closing price.
    features : list of str
        List of feature column names.
    """
    df = df.copy()
    # Drop any non‑contributing columns
    for col in ("Ignore",):
        if col in df.columns:
            df = df.drop(columns=[col])

    # Convert timestamps
    if 'Open time' in df.columns:
        df['Open time'] = pd.to_datetime(df['Open time'])
    if 'Close time' in df.columns:
        df['Close time'] = pd.to_datetime(df['Close time'])

    # Feature engineering
    df['return'] = df['Close'].pct_change()
    df['volatility'] = df['return'].rolling(window=5).std()
    df['volume_change'] = df['Volume'].pct_change()
    df['trade_change'] = df['Number of trades'].pct_change()

    # Target variable – shift close by -1 to predict next minute
    df['target'] = df['Close'].shift(-1)

    # Drop rows with NaN or infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Define feature list
    features = [
        'Close',
        'Volume',
        'Number of trades',
        'Taker buy base asset volume',
        'Taker buy quote asset volume',
        'volatility',
        'volume_change',
        'trade_change',
    ]

    X = df[features].copy().apply(pd.to_numeric, errors='coerce')
    y = df['target'].copy().astype(float)

    # Remove any remaining NaNs
    combined = pd.concat([X, y], axis=1).dropna()
    X = combined[features]
    y = combined['target']
    return X, y, features


def train_decision_tree(X: pd.DataFrame, y: pd.Series, max_depth: int = 5) -> Tuple[DecisionTreeRegressor, float, float]:
    """Train a Decision Tree regressor and return the model and metrics.

    Splits the data 80/20, trains the model, and computes MSE and R² on the
    test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2


def train_kmeans(X: pd.DataFrame, n_clusters: int = 3) -> Tuple[KMeans, np.ndarray, float, StandardScaler]:
    """Fit K‑Means on selected features and compute the silhouette score.

    Returns the fitted kmeans model, cluster labels, silhouette score, and the
    scaler used for normalising the data.
    """
    # Select only the clustering features
    cluster_features = [
        'Volume',
        'Number of trades',
        'Taker buy base asset volume',
        'Taker buy quote asset volume',
        'volatility',
        'volume_change',
        'trade_change',
    ]
    X_cluster = X[cluster_features].replace([np.inf, -np.inf], np.nan).dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, clusters)
    return kmeans, clusters, score, scaler


def evaluate_hybrid(model: DecisionTreeRegressor, X: pd.DataFrame, y: pd.Series,
                    clusters: np.ndarray, trusted_clusters: List[int]) -> Tuple[float, float]:
    """Evaluate predictions only within trusted clusters.

    Parameters
    ----------
    model : DecisionTreeRegressor
        The trained decision tree model.
    X : DataFrame
        Feature matrix used to train the model (must align with clusters).
    y : Series
        True target values corresponding to X.
    clusters : ndarray
        Cluster assignments for each row in X (must be same length as X).
    trusted_clusters : list
        List of cluster indices for which predictions are considered reliable.

    Returns
    -------
    mse : float
        Mean squared error on the trusted subset.
    r2 : float
        R² score on the trusted subset.
    """
    mask = np.isin(clusters, trusted_clusters)
    if not mask.any():
        raise ValueError("No rows belong to trusted clusters")
    X_trusted = X.loc[mask]
    y_trusted = y.loc[mask]
    preds = model.predict(X_trusted)
    mse = mean_squared_error(y_trusted, preds)
    r2 = r2_score(y_trusted, preds)
    return mse, r2


def fetch_latest_binance_data(symbol: str = 'XRPUSDT', interval: str = '1m', limit: int = 5) -> pd.DataFrame:
    """Fetch recent klines data from Binance.

    Returns a DataFrame with columns matching the training data format.  Note
    that Binance returns timestamps in milliseconds.
    """
    url = 'https://api.binance.com/api/v3/klines'
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data, columns=[
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
    ])
    # Convert numeric columns to floats/ints
    float_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                  'Quote asset volume', 'Taker buy base asset volume',
                  'Taker buy quote asset volume']
    int_cols = ['Number of trades']
    df[float_cols] = df[float_cols].astype(float)
    df[int_cols] = df[int_cols].astype(int)
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')
    return df


def compute_live_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling features on live data for the most recent rows.

    Unlike the batch preparation step, this function only updates derived
    columns for the latest observations so the live bot can operate
    incrementally.
    """
    df = df.copy()
    # Ensure time ordering
    df = df.sort_values('Open time')
    df['return'] = df['Close'].pct_change()
    df['volatility'] = df['return'].rolling(window=5).std()
    df['volume_change'] = df['Volume'].pct_change()
    df['trade_change'] = df['Number of trades'].pct_change()
    df['target'] = df['Close'].shift(-1)
    return df


def run_live_bot(model: DecisionTreeRegressor, scaler: StandardScaler, kmeans: KMeans,
                 feature_columns: List[str], trusted_clusters: Optional[List[int]] = None,
                 poll_interval: float = 60.0) -> None:
    """Continuously fetch live data and print price predictions.

    This function loops indefinitely, fetching the two most recent minutes of
    XRP/USDT data from Binance, appending them to an internal window,
    computing features and clusters, and making predictions when the current
    market regime is trusted.
    """
    if trusted_clusters is None:
        trusted_clusters = [1, 2]

    window_data: pd.DataFrame = pd.DataFrame()
    print("Starting live bot – press Ctrl+C to stop.")
    try:
        while True:
            # Get the latest two minutes of data
            latest = fetch_latest_binance_data(symbol='XRPUSDT', interval='1m', limit=2)
            # Append and remove duplicates
            window_data = pd.concat([window_data, latest], ignore_index=True).drop_duplicates(subset=['Open time'])
            # We need at least 6 rows to compute a 5‑period rolling std
            if len(window_data) < 6:
                print("Waiting for enough data...")
                time.sleep(poll_interval)
                continue
            # Compute live features
            window_data = compute_live_features(window_data)
            # Take the most recent row (the latest minute)
            current = window_data.iloc[-1:].copy()
            # Ensure all features are present
            features = current[feature_columns].fillna(0.0)
            # Standardise the clustering features to determine the current regime
            cluster_features = [
                'Volume', 'Number of trades', 'Taker buy base asset volume',
                'Taker buy quote asset volume', 'volatility',
                'volume_change', 'trade_change'
            ]
            X_cluster = features[cluster_features]
            # Some features may be missing due to NaN; replace with zeros
            X_cluster = X_cluster.values.astype(float)
            X_scaled = scaler.transform(X_cluster)
            cluster_idx = kmeans.predict(X_scaled)[0]
            if cluster_idx in trusted_clusters:
                pred = model.predict(features.values)[0]
                print(f"[Cluster {cluster_idx}] Predicted next price: {pred:.5f} | Current: {current['Close'].values[0]:.5f}")
            else:
                print(f"[Cluster {cluster_idx}] Market not predictable. No trade.")
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("Stopped live bot.")


def main(csv_path: str) -> None:
    """Load data, train models and print evaluation results.

    Parameters
    ----------
    csv_path : str
        Path to a CSV file containing minute‑level XRP/USDT data.
    """
    print(f"Loading data from {csv_path}…")
    df = pd.read_csv(csv_path)
    X, y, features = prepare_dataset(df)
    print(f"Prepared dataset with {len(X)} rows and {len(features)} features.")

    # Train Decision Tree
    dt_model, mse, r2 = train_decision_tree(X, y, max_depth=5)
    print(f"Decision Tree – MSE: {mse:.8f}, R²: {r2:.6f}")

    # Train K‑Means
    kmeans, clusters, silhouette, scaler = train_kmeans(X)
    print(f"K‑Means – Silhouette score: {silhouette:.3f}")

    # Evaluate hybrid on trusted clusters (1 and 2 by default)
    hybrid_mse, hybrid_r2 = evaluate_hybrid(dt_model, X, y, clusters, trusted_clusters=[1, 2])
    print(f"Hybrid (clusters 1 & 2) – MSE: {hybrid_mse:.8f}, R²: {hybrid_r2:.6f}")

    # Example of running the live bot (commented out to avoid long‑running process)
    # run_live_bot(model=dt_model, scaler=scaler, kmeans=kmeans, feature_columns=features, trusted_clusters=[1, 2])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train a model to predict the next minute's XRP price and run a live bot.")
    parser.add_argument('csv_path', help='Path to the minute‑level XRP data CSV file')
    args = parser.parse_args()
    main(args.csv_path)