# This module handles data storage and synchronization for the Senate trades dataset.
# It checks the local CSV for the most recent trade date, scrapes only new trades from the website, merges them with existing data 
# while avoiding duplicates, and saves the updated dataset back to disk.
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from src.ingestion.capitol_client import CapitolTradesClient
import logging

logger = logging.getLogger(__name__)

# Where we keep the loot
DATA_PATH = Path("data/processed/senate_trades_history.csv")

def load_local_data() -> pd.DataFrame:
    """Reads the CSV."""
    if not DATA_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(DATA_PATH)
        # Fix date types cuz CSVs turn them into strings
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['disclosure_date'] = pd.to_datetime(df['disclosure_date'])
        return df
    except Exception as e:
        print(f"Error loading local data: {e}")
        return pd.DataFrame()


def sync_data(MAX_LOOKBACK_DAYS: int = 365):
    """
    The Master Function.
    1. Checks local DB for last date.
    2. Scrapes only what's new (within lookback limit).
    3. Trims old rows to avoid stale analysis and saves.
    """
    df_local = load_local_data()

    cutoff_date = pd.Timestamp(datetime.now() - timedelta(days=MAX_LOOKBACK_DAYS)).normalize()
    print(f"\nCutoff date: {cutoff_date.date()}\n")

    if not df_local.empty:
        # Limit local records to a rolling 90-day window, then refresh from the newest allowed date.
        old_len = len(df_local)
        df_local = df_local[df_local['transaction_date'] >= cutoff_date].copy()
        if len(df_local) != old_len:
            print(f"Pruned {old_len - len(df_local)} stale rows older than {MAX_LOOKBACK_DAYS} days.")

    start_date = None
    if not df_local.empty:
        last_date = df_local['transaction_date'].max()
        start_date = last_date.strftime('%Y-%m-%d')
        print(f"Local data found up to {start_date}. Checking for new data within {MAX_LOOKBACK_DAYS}-day window...")
    else:
        start_date = cutoff_date.strftime('%Y-%m-%d')
        print(f"🆕 No local data (or all old data pruned). Scraping from {start_date} onwards.")

    # Run the scraper
    client = CapitolTradesClient()
    df_new = client.fetch_trades(start_date=start_date)

    if df_new.empty:
        print(" No new trades found. Up to date.")
        return df_local
    
    # Want to compare df_local to df_new before merge to view how many new records we got. This is just for logging, not required for merge.
    if not df_local.empty:
        new_records = len(df_new)
        print(f"\nFound {new_records} new trades since {start_date}. Updating database...\n")

    # Merge logic (Avoid dupes)
    if not df_local.empty:
        # Combine old and new
        df_combined = pd.concat([df_local, df_new])
        # Dedupe based on key fields (cuz we don't have a unique ID)
        df_combined = df_combined.drop_duplicates(
            subset=['transaction_date', 'senator', 'ticker', 'amount_est', 'type'],
            keep='last'
        )
    else:
        df_combined = df_new

    # Prune combined dataset again to ensure we keep only the latest window
    df_combined = df_combined[df_combined['transaction_date'] >= cutoff_date].copy()

    # Save to disk (Persistence)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(DATA_PATH, index=False)
    print(f"Database updated. Total records: {len(df_combined)}")

    return df_combined