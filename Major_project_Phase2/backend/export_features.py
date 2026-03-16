"""
Export feature engineering data to CSV file.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ALL_TICKERS
from data.fetcher import DataFetcher
from data.feature_engineering import FeatureEngineer

def export_features():
    """Fetch data and export engineered features to CSV."""
    print("Fetching data for all tickers...")
    df = DataFetcher.fetch_prices(ALL_TICKERS, start="2015-01-01")
    
    print("Engineering features...")
    df_features = FeatureEngineer.add_technicals(df)
    df_features = FeatureEngineer.add_calendar_features(df_features)
    df_features = FeatureEngineer.add_labels(df_features)
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), "data", "engineered_features.csv")
    df_features.to_csv(output_path, index=False)
    print(f"Exported {len(df_features)} rows to {output_path}")
    
    # Print summary
    print(f"\nFeature Summary:")
    print(f"  - Total rows: {len(df_features)}")
    print(f"  - Total columns: {len(df_features.columns)}")
    print(f"  - Tickers: {df_features['ticker'].nunique()}")
    print(f"  - Date range: {df_features['date'].min()} to {df_features['date'].max()}")
    print(f"\nColumns: {list(df_features.columns)}")

if __name__ == "__main__":
    export_features()
