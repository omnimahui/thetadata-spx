import pandas as pd
import numpy as np
from arch import arch_model
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import OrderedDict
import pandas_ta as ta
import multiprocessing as mp
import os

base_time = datetime.strptime("00:00:00", "%H:%M:%S")
folder = os.path.dirname(os.path.realpath(__file__))+os.sep

df = pd.read_parquet(folder+"combined1minutedata.parquet")
df["underlying_price"] = df["underlying_price"].astype(float)
df.loc[:, "time"] = df["ms_of_day"].apply(
    lambda ms_of_day: (base_time + timedelta(milliseconds=ms_of_day)).time()
)
df.set_index(["date", "time"], inplace=True)

ROLLING_WINDOW = 30

def garch_calc(rtn):
    am = arch_model(rtn.dropna().values*100)
    res = am.fit(disp="off")
    forecasts = res.forecast(horizon=5)
    # 390 minutes per day
    vol_forecast = (
        forecasts.residual_variance.iloc[-1, :].sum() * 390 / 5
    ) ** 0.5 / 100
    return vol_forecast

def sigma(price, method="garch"):
    if method == "hist":
        return (price.pct_change()*100).std() * (390**0.5)
    elif method == "garch":
        return garch_calc(price.pct_change()*100)
    elif method == "realized":
        return np.sqrt(((price.pct_change()*100)**2).sum()) * np.sqrt(
            390
        )


def find_strike(underlying_price, strike, callput="C", spread=0):
    sign = 1 if callput == "C" else -1
    if len(underlying_price) == 0:
        print("Invalid underlying_price")
        return np.nan
    if len(strike) == 0:
        print("Invalid strike")
        return np.nan

    price = underlying_price.iloc[0]
    while True:
        closest_index = np.argmin(np.abs((strike.values - price))) + sign * spread
        if abs(closest_index) < len(strike):
            closest_value = strike.iloc[closest_index]
            break
        else:
            # closest_value = strike.iloc[closest_index - sign * spread]
            spread = spread - 1

    return closest_value


def iv_on_spread(df,spreads=1):
    df_iv = pd.DataFrame()
    for spread in range(0, spreads):
        atm_iv_col_name = f"atm_plus_{spread}spread_iv"
        atm_col_name = f"atm_plus_{spread}spread"
        df_day_call = df.loc[df["right"] == "C"].reset_index().set_index("time")
        df_day_call.loc[:, atm_col_name] = df_day_call.groupby(["time"]).apply(
            lambda rows: find_strike(
                rows["underlying_price"], rows["strike"], "C", spread=spread
            )
        )
        df_iv_call = df_day_call.loc[
            df_day_call["strike"] == df_day_call[atm_col_name]
        ]["implied_vol"].to_frame()
        df_iv_call.columns = [atm_iv_col_name + "_CALL"]

        df_day_put = df.loc[df["right"] == "P"].reset_index().set_index("time")
        df_day_put.loc[:, atm_col_name] = df_day_put.groupby(["time"]).apply(
            lambda rows: find_strike(
                rows["underlying_price"], rows["strike"], "P", spread=spread
            )
        )
        df_iv_put = df_day_put.loc[df_day_put["strike"] == df_day_put[atm_col_name]][
            "implied_vol"
        ].to_frame()
        df_iv_put.columns = [atm_iv_col_name + "_PUT"]

        df_iv = pd.concat(
            [
                df_iv,
                df_iv_call,
                df_iv_put,
            ],
            axis=1,
        )

    return df_iv


def pnl_on_spread(df, spreads=1):
    df_pnl = pd.DataFrame()
    for spread in range(0, spreads):
        atm_col_name = f"atm_plus_{spread}spread"
        atm_pnl_col_name = f"atm_plus_{spread}spread_pnl"
        df_day_call = df.loc[df["right"] == "C"].reset_index().set_index("time")

        df_day_call.loc[:, atm_col_name] = df_day_call.groupby(["time"]).apply(
            lambda rows: find_strike(
                rows["underlying_price"], rows["strike"], "C", spread=spread
            )
        )

        df_pnl_call = df_day_call.loc[
            df_day_call["strike"] == df_day_call[atm_col_name]
        ]["EOD_PnL"].to_frame()

        df_day_put = df.loc[df["right"] == "P"].reset_index().set_index("time")
        df_day_put.loc[:, atm_col_name] = df_day_put.groupby(["time"]).apply(
            lambda rows: find_strike(
                rows["underlying_price"], rows["strike"], "P", spread=spread
            )
        )

        df_pnl_put = df_day_put.loc[df_day_put["strike"] == df_day_put[atm_col_name]][
            "EOD_PnL"
        ].to_frame()

        df_pnl_sum_spread = df_pnl_call + df_pnl_put
        df_pnl_sum_spread.columns = [atm_pnl_col_name]

        df_pnl = pd.concat(
            [df_pnl, df_pnl_sum_spread],
            axis=1,
        )
    return df_pnl


def underlying_ohlc(df):
    df_price=df.groupby(["date", "time"])["underlying_price"].first().reset_index()
    # Combine date and time into a single datetime column
    df_price["datetime"] = pd.to_datetime(df_price["date"].astype(str) + " " + df_price["time"].astype(str))

    # Set the datetime column as the index
    df_price.set_index("datetime", inplace=True)

    # Drop the original date and time columns (optional)
    df_price.drop(columns=["date", "time"], inplace=True)

    # Resample to daily OHLC
    ohlc_dict = {
        "price": {
            "open": "first",  # First price of the day
            "high": "max",  # Highest price of the day
            "low": "min",  # Lowest price of the day
            "close": "last",  # Last price of the day
        }
    }

    # Resample and aggregate
    df_ohlc = (
        df_price["underlying_price"]
        .resample("D")
        .agg(
            OrderedDict(
                [
                    ("open", "first"),
                    ("high", "max"),
                    ("low", "min"),
                    ("close", "last"),
                ]
            )
        )
    ).dropna(axis=0)

    df_ohlc.index.name = "date"
    # Display the result
    return df_ohlc

def daily_atr(df,window=14):
    df_ohlc = underlying_ohlc(df)
    df_atr = df_ohlc.ta.atr(window, fillna=0)
    return df_atr

def feature_engineering_global(df):
    df_day_price = df.groupby(["date", "time"])["underlying_price"].first()
    df_vlty = (
        df_day_price
        .rolling(ROLLING_WINDOW)
        .apply(lambda x: sigma(x, method="garch"))
        .reset_index()
        .set_index("time")
    )
    df_histvlty = (
        df_day_price
        .rolling(ROLLING_WINDOW)
        .apply(lambda x: sigma(x, method="hist"))
        .reset_index()
        .set_index("time")
    )

    df_price = (
        df_day_price
        .reset_index()
        .set_index("time")
    )

    df_feature = pd.concat(
        [
            df_day_price.reset_index().set_index("time")["date"],
            df_vlty["underlying_price"],
            df_histvlty["underlying_price"],
            df_price["underlying_price"],
        ],
        axis=1,
    )
    df_feature.columns = ["date", "Garch_Vlty", "Hist_Vlty", "Price"]
    df_feature=df_feature.reset_index().set_index(["date", "time"]) 
    return df_feature   

def feature_engineering_by_date(df):
    df_day_price = df.groupby(["time"])["underlying_price"].first()

    df_iv = iv_on_spread(df,spreads=1)
    df_pnl = pnl_on_spread(df,spreads=1)

    df_features = pd.concat(
        [
            df.reset_index().groupby("time")["date"].first(),
            df_iv,
            df_pnl
        ],
        axis=1,
    )
    df_features.columns = ["date"]+df_iv.columns.to_list()+df_pnl.columns.to_list()
    df_features = df_features.reset_index().set_index(["date","time"]).fillna(0)
    return df_features

#df=df.loc[(df.index.get_level_values(0) == '20250211')|(df.index.get_level_values(0) == '20250214')]
df_features_global = feature_engineering_global(df)
# Split DataFrame into groups
groups = [group for _, group in df.groupby('date')]

# Use multiprocessing to process groups in parallel
with mp.Pool(mp.cpu_count()) as pool:
    results = pool.map(feature_engineering_by_date, groups)

# Concatenate results back into a single DataFrame
df_features_date = pd.concat(results)
df_features = pd.concat([df_features_global,df_features_date],axis=1)
df_features.index = df_features.index.set_levels(df.index.levels[1].astype(str), level=1)
df_features.to_parquet(folder+"thetadata-spx-features.parquet")
