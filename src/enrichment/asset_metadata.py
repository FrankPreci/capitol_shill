import yfinance as yf
import pandas as pd
import logging
import sys
import os
from contextlib import contextmanager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

@contextmanager
def suppress_stdout_stderr():
    """
    A context manager that redirects stdout and stderr to devnull
    to suppress yfinance's verbose 404 errors.
    """
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

class AssetEnricher:
    def __init__(self):
        # Cache to prevent hitting Yahoo Finance API repeatedly for the same ticker
        self._cache = {}

    def get_asset_info(self, ticker: str) -> dict:
        """
        Fetches Sector, Industry, and Name for a given ticker.
        """
        # 1. Validation
        if not isinstance(ticker, str):
            return self._default_metadata()

        ticker = ticker.strip().upper()

        # FIX: Sanitize ticker for Yahoo Finance (BRK/B -> BRK-B)
        ticker = ticker.replace('/', '-').replace('.', '-')

        # Skip tickers starting with $ (Crypto) or digits (Invalid/Bond)
        if ticker.startswith('$') or (len(ticker) > 0 and ticker[0].isdigit()):
            return self._default_metadata()

        # 2. Check Cache
        if ticker in self._cache:
            return self._cache[ticker]

        # 3. Handle Invalid Tickers
        if not ticker or ticker == "---" or "UNKNOWN" in ticker:
            return self._default_metadata()

        # 4. Fetch from yfinance
        try:
            # Suppress the annoying 404 prints from yfinance
            with suppress_stdout_stderr():
                stock = yf.Ticker(ticker)
                info = stock.info

            metadata = {
                "name": info.get('shortName', 'Unknown'),
                "sector": info.get('sector', 'Unknown'),
                "industry": info.get('industry', 'Unknown'),
                "market_cap": info.get('marketCap', 0)
            }

            self._cache[ticker] = metadata
            return metadata

        except Exception as e:
            # Common for delisted stocks or weird symbols
            # logger.debug(f"Metadata fetch failed for {ticker}: {e}")
            return self._default_metadata()

    def _default_metadata(self):
        return {
            "name": "Unknown",
            "sector": "Unknown",
            "industry": "Unknown",
            "market_cap": 0
        }

    def enrich_dataframe(self, df: pd.DataFrame, ticker_col='ticker') -> pd.DataFrame:
        """
        Applies enrichment to a full DataFrame.
        Handles column deduplication to prevent crashes.
        """
        if df.empty: return df

        logger.info(f"Enriching {len(df)} trades with market data...")

        # 1. Get new metadata
        # Use apply to fetch data for every row
        meta_list = df[ticker_col].apply(self.get_asset_info)
        meta_df = pd.DataFrame(meta_list.tolist())

        # 2. Clean up duplicates BEFORE merging
        # If 'sector' or 'industry' already exist (as None/NaN), drop them
        # so the new meta_df versions take precedence.
        cols_to_drop = [c for c in meta_df.columns if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # 3. Merge
        # We reset index to ensure rows align perfectly
        df = df.reset_index(drop=True)
        meta_df = meta_df.reset_index(drop=True)

        result = pd.concat([df, meta_df], axis=1)
        return result