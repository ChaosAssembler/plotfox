from matplotlib.ticker import ScalarFormatter, LogLocator
import numpy
from pandas import DataFrame
from typing import Literal, Sequence
from matplotlib.typing import RcStyleType
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

_zorder_counter = 0


def next_zorder() -> int:
    global _zorder_counter
    _zorder_counter += 10
    return _zorder_counter - 10


def candlestick_chart(hist: DataFrame):
    assert sorted(hist.index)
    bar_width = (hist.index[1:] - hist.index[:-1]).min()

    plt.bar(
        hist.index,
        hist["High"] - hist["Low"],
        bottom=hist["Low"],
        color=[
            "green" if c >= o else "red" for c, o in zip(hist["Close"], hist["Open"])
        ],
        width=bar_width / 10,
        zorder=next_zorder(),
    )
    plt.bar(
        hist.index,
        (hist["Close"] - hist["Open"]).abs(),
        bottom=hist[["Close", "Open"]].min(axis=1),
        color=[
            "green" if c >= o else "red" for c, o in zip(hist["Close"], hist["Open"])
        ],
        width=bar_width,
        zorder=next_zorder(),
    )


ChartTransform = (
    Literal["candlestick", "close", "open"]
    | tuple[Literal["running-avg", "mid-running-avg"], int]
)


def plot(
    ticker: yf.Ticker,
    charts: Sequence[ChartTransform] = ("candlestick",),
    period="max",
    style: RcStyleType = "dark_background",
) -> None:
    hist = ticker.history(period=period)

    hist["CloseT"] = numpy.log2(hist["Close"])
    dispT = numpy.exp2

    plt.style.use(style)
    plt.grid(
        True, which="both", axis="y", zorder=next_zorder(), color=(0.5, 0.5, 0.5, 0.5)
    )

    for chart in charts:
        match chart:
            case "candlestick":
                candlestick_chart(hist)
            case "open":
                plt.plot(
                    hist.index,
                    hist["Open"],
                    label=f"{ticker} - Open",
                    zorder=next_zorder(),
                )
            case "close":
                plt.plot(
                    hist.index,
                    hist["Close"],
                    label=f"{ticker} - Close",
                    zorder=next_zorder(),
                )
            case "running-avg", num_days:
                plt.plot(
                    hist.index,
                    dispT(hist["CloseT"].rolling(window=num_days).mean()),
                    label=f"{num_days}-Day Running Avg.",
                    zorder=next_zorder(),
                )
            case "mid-running-avg", num_days:
                hist["CloseT_ravg"] = (
                    hist["CloseT"].rolling(window=num_days, center=True).mean()
                )
                plt.plot(
                    hist.index,
                    dispT(hist["CloseT_ravg"]),
                    label=f"{num_days}-Day Mid. Running Avg.",
                    zorder=next_zorder(),
                )

                hist["timestamp_numeric"] = hist.index.astype("int64") // 10**9

                lin_pred_factorT = (
                    LinearRegression()
                    .fit(
                        hist["timestamp_numeric"][-num_days:].values.reshape(-1, 1),
                        hist["CloseT"][-num_days:].values.reshape(-1, 1),
                    )
                    .coef_[0, 0]
                )

                ravg_endT = (
                    hist["CloseT"]
                    .rolling(window=num_days, center=True, min_periods=1)
                    .mean()[(-num_days + 1) // 2 :]
                )
                ravg_lin_w = (
                    DataFrame(
                        range(len(ravg_endT) - 1, -1, -1),
                        index=hist.index[(-num_days + 1) // 2 :],
                    ).squeeze()
                    / (len(ravg_endT) - 1)
                    / 2
                    + 0.5
                )

                line_pred_m1p5T = (
                    hist["CloseT_ravg"].iloc[-num_days // 2]
                    + (
                        hist["timestamp_numeric"][-num_days // 2 :]
                        - hist["timestamp_numeric"].iloc[-num_days // 2]
                    )
                    * lin_pred_factorT
                    * 1.5
                )

                hist["ravg_lin_predT"] = ravg_endT * ravg_lin_w + line_pred_m1p5T[
                    -len(ravg_endT) :
                ] * (1 - ravg_lin_w)
                plt.plot(
                    hist.index,
                    dispT(hist["ravg_lin_predT"]),
                    label=f"{num_days}-Day Mid. Running Avg. Pred.",
                    zorder=next_zorder(),
                )

    plt.title(f"{ticker.ticker} over {period}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.yscale("log")
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.gca().yaxis.set_major_locator(LogLocator(subs="all"))
    plt.legend()
    plt.show()


def main():
    dro = yf.Ticker("DRH.MU")
    plot(
        dro,
        period="10y",
        charts=(
            "candlestick",
            # ("running-avg", 50),
            # "close",
            # ("mid-running-avg", 50),
            # ("mid-running-avg", 100),
            # ("mid-running-avg", 150),
            # ("mid-running-avg", 200),
            ("mid-running-avg", 300),
            ("mid-running-avg", 500),
            ("mid-running-avg", 1000),
        ),
    )


# def main():
#     print(DataFrame([3.0, 2.0, 1.0])[:].corrwith(DataFrame([1, 2, 3])))
