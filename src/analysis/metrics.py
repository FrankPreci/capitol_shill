import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import logging
from datetime import timedelta
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Mapping for problematic tickers to valid Yahoo Finance symbols
TICKER_MAP = {
    "$BTC": "BTC-USD",
    "VWUSX:US": "VWUSX",
    "XSP": "SPY",   # or ^GSPC if you prefer the index
    # Add more mappings here as needed
}

class EventStudy:
    def __init__(self, benchmark_ticker='^GSPC'):
        # ^GSPC is the S&P 500 index
        self.benchmark = benchmark_ticker

    def calculate_car(self, ticker: str, trade_date: pd.Timestamp, window_days=30) -> float:
        """
        Calculate CAR (cumulative abnormal return) for a trade.
        Returns decimal percent or None.
        """
        if pd.isna(trade_date):
            return None

        # Skip invalid tickers
        if not isinstance(ticker, str) or ticker in ['--', 'NaN', '']:
            return None

        # Sanitize ticker for Yahoo Finance
        ticker = ticker.strip().upper().replace('/', '-').replace('.', '-')
        ticker = TICKER_MAP.get(ticker, ticker)

        # Define time windows
        est_start = trade_date - timedelta(days=200)
        est_end = trade_date - timedelta(days=10)
        evt_end = trade_date + timedelta(days=window_days)

        try:
            # Fetch stock + market benchmark data
            data = yf.download(
                [ticker, self.benchmark],
                start=est_start,
                end=evt_end + timedelta(days=5),
                progress=False,
                auto_adjust=False,
            )['Adj Close']

            # Fix for yfinance returning multiindex columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)

            # Calculate daily returns (percent change)
            returns = data.pct_change(fill_method=None).dropna()

            # Require enough data
            if len(returns) < 50:
                return None

            # Estimation window
            est_data = returns.loc[est_start:est_end]

            # Check if both assets exist
            if est_data.empty or self.benchmark not in est_data or ticker not in est_data:
                return None

            X = est_data[self.benchmark].values.reshape(-1, 1)  # market return
            y = est_data[ticker].values  # stock return

            model = LinearRegression()
            model.fit(X, y)

            alpha = model.intercept_
            beta = model.coef_[0]

            # Event window
            evt_data = returns.loc[trade_date:evt_end]
            if evt_data.empty:
                return None

            actual_returns = evt_data[ticker]
            market_returns = evt_data[self.benchmark]

            # Expected return = alpha + (beta * market return)
            expected_returns = alpha + (beta * market_returns)

            # Abnormal return = actual - expected
            abnormal_returns = actual_returns - expected_returns

            car = abnormal_returns.sum()

            return car
        except Exception as e:
            logger.debug(f"Error calculating CAR for {ticker}: {e}")
            return None

    def analyze_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies CAR calculation to a whole dataframe of trades.
        """
        if df.empty:
            return df

        logger.debug(f"Calculating financial metrics (Alpha/Beta) for {len(df)} trades...")

        df['car_30d'] = df.apply(
            lambda row: self.calculate_car(row['ticker'], row['transaction_date']),
            axis=1
        )
        return df
