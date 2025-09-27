import pandas as pd
import numpy as np

# -------------------------------
# Add Candle Patterns
# -------------------------------
def add_candle_patterns(df, o="open", h="high", l="low", c="close",
                        doji_thresh=0.1, long_body=0.6, small_body=0.25,
                        shadow_ratio=2.0, near_edge=0.25):
    O, H, L, C = df[o].astype(float), df[h].astype(float), df[l].astype(float), df[c].astype(float)
    rng   = (H - L).replace(0, np.nan)
    body  = (C - O).abs()
    upsh  = H - np.maximum(O, C)
    dnsh  = np.minimum(O, C) - L
    bull  = C > O
    bear  = O > C

    new_cols = {}

    # Core patterns
    new_cols["Doji"] = (body <= doji_thresh * rng)
    new_cols["Hammer"] = (dnsh >= shadow_ratio * body) & (upsh <= 0.2 * body) & (np.maximum(O, C) >= H - near_edge * rng)
    new_cols["InvertedHammer"] = (upsh >= shadow_ratio * body) & (dnsh <= 0.2 * body) & (np.minimum(O, C) <= L + near_edge * rng)
    new_cols["ShootingStar"] = (upsh >= shadow_ratio * body) & (dnsh <= 0.2 * body) & (np.minimum(O, C) <= L + near_edge * rng)

    O1, C1 = O.shift(1), C.shift(1)
    new_cols["BullishEngulfing"] = (O1 > C1) & bull & (C >= O1) & (O <= C1)
    new_cols["BearishEngulfing"] = (C1 > O1) & bear & (O >= C1) & (C <= O1)

    new_cols["BullishHarami"] = (O1 > C1) & bull & (np.maximum(O, C) <= np.maximum(O1, C1)) & (np.minimum(O, C) >= np.minimum(O1, C1))
    new_cols["BearishHarami"] = (C1 > O1) & bear & (np.maximum(O, C) <= np.maximum(O1, C1)) & (np.minimum(O, C) >= np.minimum(O1, C1))
    new_cols["HaramiCross"] = new_cols["Doji"] & ((np.maximum(O, C) <= np.maximum(O1, C1)) & (np.minimum(O, C) >= np.minimum(O1, C1)))

    new_cols["PiercingLine"] = (O1 > C1) & bull & (O < C1) & (C > (O1 + C1) / 2) & (C < O1)
    new_cols["DarkCloudCover"] = (C1 > O1) & bear & (O > C1) & (C < (O1 + C1) / 2) & (C > O1)

    O2, C2 = O.shift(2), C.shift(2)
    new_cols["MorningStar"] = (O2 > C2) & (abs(C1 - O1) < abs(C2 - O2) * small_body) & bull & (C >= (O2 + C2) / 2)
    new_cols["EveningStar"] = (C2 > O2) & (abs(C1 - O1) < abs(C2 - O2) * small_body) & bear & (C <= (O2 + C2) / 2)

    bull1, bull2 = bull.shift(1), bull.shift(2)
    bear1, bear2 = bear.shift(1), bear.shift(2)
    new_cols["ThreeWhiteSoldiers"] = bull & bull1 & bull2 & (C > C.shift(1)) & (C.shift(1) > C.shift(2))
    new_cols["ThreeBlackCrows"] = bear & bear1 & bear2 & (C < C.shift(1)) & (C.shift(1) < C.shift(2))

    # Additional patterns
    new_cols["BullishMarubozu"] = (body >= long_body * rng) & (upsh <= 0.05 * rng) & (dnsh <= 0.05 * rng) & bull
    new_cols["BearishMarubozu"] = (body >= long_body * rng) & (upsh <= 0.05 * rng) & (dnsh <= 0.05 * rng) & bear

    new_cols["TweezerTop"] = (H == H.shift(1)) & bear & bull.shift(1)
    new_cols["TweezerBottom"] = (L == L.shift(1)) & bull & bear.shift(1)

    new_cols["InsideBar"] = (H < H.shift(1)) & (L > L.shift(1))
    new_cols["OutsideBar"] = (H > H.shift(1)) & (L < L.shift(1))

    # Diagnostics
    new_cols["SuspiciousCandle"] = (rng <= 0.001) | (body <= 0.001)
    recent_high = H.rolling(5).max()
    recent_low = L.rolling(5).min()
    new_cols["NearHigh"] = H >= recent_high * (1 - near_edge)
    new_cols["NearLow"] = L <= recent_low * (1 + near_edge)

    # Add all pattern columns at once
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # Metadata
    pattern_cols = [col for col in new_cols if df[col].dtype == bool]
    df["PatternCount"] = df[pattern_cols].sum(axis=1)

    df["PatternType"] = np.select(
        [df["BullishEngulfing"], df["MorningStar"], df["ThreeWhiteSoldiers"], df["BullishMarubozu"]],
        ["BullishEngulfing", "MorningStar", "ThreeWhiteSoldiers", "BullishMarubozu"],
        default="None"
    )

    return df.copy().fillna(False)


# -------------------------------
# Add Trends
# -------------------------------
def add_trend_filters(df, c="close"):
    df = df.copy()  # Ensure safe assignment

    profiles = {
        "Short":  {"ma": 2,  "ret": 1,  "vol": 3},
        "Long":   {"ma": 50, "ret": 20, "vol": 30},
        "Swing":  {"ma": 20, "ret": 5,  "vol": 10},
        "Daily":  {"ma": 5,  "ret": 1,  "vol": 5}
    }

    new_cols = {}

    for label, p in profiles.items():
        # Moving Average & Slope
        ma = df[c].rolling(p["ma"]).mean()
        ma_slope = ma.diff(p["ma"])
        new_cols[f"MA_{label}"] = ma
        new_cols[f"MA_slope_{label}"] = ma_slope
        new_cols[f"UpTrend_MA_{label}"] = ma_slope > 0
        new_cols[f"DownTrend_MA_{label}"] = ma_slope < 0

        # Recent Return
        recent_ret = df[c].pct_change(p["ret"])
        new_cols[f"RecentReturn_{label}"] = recent_ret
        new_cols[f"UpTrend_Return_{label}"] = recent_ret > 0
        new_cols[f"DownTrend_Return_{label}"] = recent_ret < 0

        # Volatility
        vol = df[c].rolling(p["vol"]).std()
        vol_median = vol.median()
        new_cols[f"Volatility_{label}"] = vol
        new_cols[f"LowVolatility_{label}"] = vol < vol_median
        new_cols[f"HighVolatility_{label}"] = vol > vol_median

        # Momentum
        roc = df[c].pct_change(p["ma"])
        new_cols[f"ROC_{label}"] = roc
        new_cols[f"MomentumUp_{label}"] = roc > 0.02
        new_cols[f"MomentumDown_{label}"] = roc < -0.02

        # Confirmed Trend
        new_cols[f"ConfirmedUpTrend_{label}"] = (
            new_cols[f"UpTrend_MA_{label}"] &
            new_cols[f"UpTrend_Return_{label}"] &
            new_cols[f"MomentumUp_{label}"]
        )
        new_cols[f"ConfirmedDownTrend_{label}"] = (
            new_cols[f"DownTrend_MA_{label}"] &
            new_cols[f"DownTrend_Return_{label}"] &
            new_cols[f"MomentumDown_{label}"]
        )

    # Add all columns at once
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df.copy()


# -------------------------------
# Confirmed Signals
# -------------------------------
def add_confirmed_signals(df):
    df = df.copy()  # Ensure safe assignment
    timeframes = ['Short', 'Long', 'Swing', 'Daily']
    new_cols = {}

    for tf in timeframes:
        new_cols[f"ValidHammer_{tf}"] = df.get("Hammer", False) & df.get(f"DownTrend_MA_{tf}", False)
        new_cols[f"ValidShootingStar_{tf}"] = df.get("ShootingStar", False) & df.get(f"UpTrend_MA_{tf}", False)
        new_cols[f"ValidBullishEngulfing_{tf}"] = df.get("BullishEngulfing", False) & df.get(f"DownTrend_MA_{tf}", False)
        new_cols[f"ValidBearishEngulfing_{tf}"] = df.get("BearishEngulfing", False) & df.get(f"UpTrend_MA_{tf}", False)
        new_cols[f"ValidPiercingLine_{tf}"] = df.get("PiercingLine", False) & df.get(f"DownTrend_Return_{tf}", False)
        new_cols[f"ValidDarkCloud_{tf}"] = df.get("DarkCloudCover", False) & df.get(f"UpTrend_Return_{tf}", False)
        new_cols[f"ValidMorningStar_{tf}"] = df.get("MorningStar", False) & df.get(f"DownTrend_MA_{tf}", False)
        new_cols[f"ValidEveningStar_{tf}"] = df.get("EveningStar", False) & df.get(f"UpTrend_MA_{tf}", False)
        new_cols[f"ValidThreeWhiteSoldiers_{tf}"] = df.get("ThreeWhiteSoldiers", False) & df.get(f"DownTrend_MA_{tf}", False)
        new_cols[f"ValidThreeBlackCrows_{tf}"] = df.get("ThreeBlackCrows", False) & df.get(f"UpTrend_MA_{tf}", False)

    # Add all columns at once
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df.copy()