import yfinance as yf
import pandas as pd


class Ticker:
    def __init__(self, ticker: str):
        self._ticker = yf.Ticker(ticker)

    def __str__(self) -> str:
        return self._ticker.ticker

    def history(self, period: None | str = None) -> pd.DataFrame:
        return self._ticker.history(period=period)

    def __truediv__(self, other: "Ticker") -> "DivTicker":
        return DivTicker(self, other)


class DivTicker(Ticker):
    dividend: Ticker
    divisor: Ticker

    def __init__(self, dividend: Ticker, divisor: Ticker):
        self.dividend = dividend
        self.divisor = divisor

    def __str__(self) -> str:
        return f"{self.dividend}/{self.divisor}"

    def history(self, *args, **kwargs) -> pd.DataFrame:
        dividend_df = self.dividend.history(*args, **kwargs)
        divisor_df = self.divisor.history(*args, **kwargs)

        dividend_df.index = dividend_df.index.tz_localize(None)  # ty:ignore[possibly-missing-attribute]
        divisor_df.index = divisor_df.index.tz_localize(None)  # ty:ignore[possibly-missing-attribute]

        df = pd.DataFrame(
            data={
                col: dividend_df[col] / divisor_df[col]
                for col in {"Open", "High", "Low", "Close"}
            },
            index=dividend_df.index,
        )

        return df.dropna()
