CONFIG = {
    # -------------------------
    # Default config (baseline)
    # -------------------------
    "timeframe_map": {
        "Daily": 1,   # daily timeframe = 1 day
        "Short": 3,   # short swing = 3 days
        "Swing": 5,   # medium swing = 5 days
        "Long": 10,   # long-term swing = 10 days
    },

    # Columns we keep for writing signals (not used yet)
    "signal_keep_cols": [
        "UserId","CompanyId","StockDate","Open","High","Low","Close","TomorrowClose","Return","TomorrowReturn",
        "MA","MA_slope","UpTrend_MA","DownTrend_MA","MomentumUp","MomentumDown","ConfirmedUpTrend","ConfirmedDownTrend",
        "Volatility","LowVolatility","HighVolatility","SignalStrength","SignalStrengthHybrid","ActionConfidence",
        "ActionConfidenceNorm","BullishStrengthHybrid","BearishStrengthHybrid","SignalDuration","PatternAction",
        "CandleAction","UpTrend_Return","CandidateAction","Action","TomorrowAction","TimeFrame"
    ],
    "default": {
        "phases": {
            "phase1": {"topN": 20},  # top N stocks to select in phase1
            "phase2": {"topN": 10},  # top N for phase2
            "phase3": {"topN": 5},   # top N for phase3
        },

        "profiles": {
            "Short": {"ma": 2, "ret": 1, "vol": 3, "roc_thresh": 0.02, "slope_horizon": 1},
            "Swing": {"ma": 5, "ret": 5, "vol": 5, "roc_thresh": 0.02, "slope_horizon": 5},
            "Long":  {"ma": 10, "ret": 10, "vol": 10, "roc_thresh": 0.02, "slope_horizon": 10},
            "Daily": {"ma": 7, "ret": 1, "vol": 5, "roc_thresh": 0.02, "slope_horizon": 1},
            # ma = moving average window
            # ret = return window
            # vol = volatility window
            # roc_thresh = minimum rate of change threshold
            # slope_horizon = horizon for trend slope calculation
        },

        "candle_params": {
            "doji_base": 0.01,       # base threshold for Doji candle
            "doji_scale": 0.02,      # sensitivity scaling for Doji detection
            "doji_min": 0.01,        # min Doji proportion
            "doji_max": 0.1,         # max Doji proportion

            "long_body_base": 0.3,   # base threshold for long body candle
            "long_body_scale": 0.3,
            "long_body_min": 0.3,
            "long_body_max": 0.6,

            "small_body_base": 0.15,
            "small_body_scale": 0.1,
            "small_body_min": 0.15,
            "small_body_max": 0.25,

            "shadow_ratio_base": 1.2,   # ratio of candle shadows
            "shadow_ratio_scale": 0.8,
            "shadow_ratio_min": 1.2,
            "shadow_ratio_max": 2.0,

            "near_edge": 0.25,          # proximity to candle edge for certain patterns
            "highvol_spike": 1.5,       # volume spike multiplier
            "lowvol_dip": 0.7,          # volume dip multiplier

            "hammer_base": 0.15,
            "hammer_scale": 0.1,
            "hammer_min": 0.15,
            "hammer_max": 0.25,

            "marubozu_base": 0.03,
            "marubozu_scale": 0.02,
            "marubozu_min": 0.03,
            "marubozu_max": 0.05,

            "rng_base": 1e-5,           # minimal range threshold
            "rng_scale": 1e-4,
            "rng_min": 1e-5,
            "rng_max": 1e-4,
        },

        "signals": {
            "bullish_patterns": [
                "Hammer", "InvertedHammer", "BullishEngulfing", "BullishHarami", "PiercingLine",
                "MorningStar", "ThreeWhiteSoldiers", "BullishMarubozu", "TweezerBottom",
                "DragonflyDoji", "RisingThreeMethods", "GapUp"
            ],
            "bearish_patterns": [
                "HangingMan", "ShootingStar", "BearishEngulfing", "BearishHarami", "DarkCloudCover",
                "EveningStar", "ThreeBlackCrows", "BearishMarubozu", "TweezerTop",
                "GravestoneDoji", "FallingThreeMethods", "GapDown"
            ],
            "timeframes": {
                "Short": {"buy": ["hammer","bullish","piercing","morning","white","marubozu","tweezerbottom"],
                          "sell": ["shooting","bearish","dark","evening","black","marubozu","tweezertop"],
                          "momentum": 0.05},  # default momentum threshold for Short
                "Swing": {"buy": ["hammer","bullish","piercing","morning","white"],
                          "sell": ["shooting","bearish","dark","evening","black"],
                          "momentum": 0.1},
                "Long":  {"buy": ["bullish","morning","white","threewhitesoldiers"],
                          "sell": ["bearish","evening","black","threeblackcrows"],
                          "momentum": 0.2},
                "Daily": {"buy": ["bullish","morning","white"],
                          "sell": ["bearish","evening","black"],
                          "momentum": 0.15}
            },
            "penny_stock_adjustment": {
                "threshold": 1.0,   # price threshold for penny stock adjustment
                "factor": 0.2,      # adjustment factor
                "min_momentum": 0.005
            }
        },

        "signal_strength": {
            "doji_thresh": 0.1,
            "hammer_thresh": 0.25,
            "marubozu_thresh": 0.05,
            "long_body": 0.6,
            "small_body": 0.25,
            "shadow_ratio": 2.0,
            "near_edge": 0.25,
            "rng_thresh": 1e-4,
        },

        "fundamental_weights": {
            "valuation": 0.2,
            "profitability": 0.3,
            "DebtLiquidity": 0.2,
            "Growth": 0.2,
            "Sentiment": 0.1
        },
    },
    # -------------------------
    # User-specific overrides
    # -------------------------
    # These overrides customize candlestick thresholds, topN selection, profiles,
    # and momentum/penny stock adjustments per user strategy.
    "user1": {
        "candle_params": {
            "doji_base": 0.025,   # ↑ Increased from 0.01 to be more sensitive to Doji patterns
            "doji_max": 0.09,     # ↑ Increased max proportion from 0.08 to allow more Doji detection
            "long_body_base": 0.4  # ↑ Increased from 0.35 to prefer longer bodies for stronger bullish/bearish signals
        },
    },
    
    "user2": {
        "profiles": {
            "Swing": {  # Override the Swing profile
                "ma": 7,          # ↑ Increased moving average window from 5 to smooth more
                "ret": 5,         # same as default; retained for reference
                "vol": 5,         # same as default; volatility window unchanged
                "roc_thresh": 0.025, # ↑ Increased rate-of-change threshold slightly to filter noise
                "slope_horizon": 5   # same as default; trend horizon unchanged
            }
        },
        "phases": {
            "phase1": {"topN": 10}  # ↑ Increased topN from 20 to capture more candidate stocks for aggressive strategy
            "phase2": {"topN": 5},  # top N for phase2
            "phase3": {"topN": 2},   # top N for phase3
        },
        }
        "candle_params": {
            "doji_thresh": 0.09   # ↑ Increased Doji detection threshold from 0.08 to be more lenient
        },
        "signals": {
            "timeframes": {
                "Short": {
                    "momentum": 0.08  # ↑ Increased momentum threshold from 0.05 to require stronger trend for Short timeframe
                }
            },
            "penny_stock_adjustment": {
                "factor": 0.25  # ↑ Increased adjustment factor from 0.2 to amplify penny stock momentum adjustments
            }
        }
    },
    
    "user3": {
        "profiles": {
            "Short": {
                "ma": 3,            # ↑ Slightly longer MA than default 2 to reduce noise
                "ret": 1.5,         # ↑ Slightly higher return window to capture stronger short-term moves
                "vol": 3,           # same as default
                "roc_thresh": 0.02, # same as default
                "slope_horizon": 1  # same as default
            },
            "Long": {
                "ma": 12,           # ↑ Increased MA from 10 for smoother long-term trend
                "ret": 12,          # ↑ Longer return window to capture more pronounced moves
                "vol": 10,          # same as default
                "roc_thresh": 0.02, # same as default
                "slope_horizon": 10 # same as default
            }
        },
           "phases": {
                "phase1": {"topN": 10}  # ↑ Increased topN from 20 to capture more candidate stocks for aggressive strategy
                "phase2": {"topN": 5},  # top N for phase2
                "phase3": {"topN": 2},   # top N for phase3
        },
        "candle_params": {
            "hammer_base": 0.18,    # ↑ More sensitive to Hammer candles than default 0.15
            "hammer_max": 0.28,     # ↑ Increased max threshold to allow detection of stronger Hammers
            "marubozu_base": 0.035  # ↑ Slightly higher base for Marubozu detection than default 0.03
        },
        "signals": {
            "timeframes": {
                "Daily": {
                    "momentum": 0.12  # ↑ Increased momentum requirement for Daily timeframe signals
                }
            }
        }
    },
    
    "user4": {
        "phases": {
            "phase2": {"topN": 12},  # ↑ Increase from 10 to allow more candidates in phase2
            "phase3": {"topN": 6}    # ↑ Increase from 5 to allow more candidates in phase3
        },
        "candle_params": {
            "long_body_base": 0.38,     # ↑ Increased long body sensitivity to detect stronger candles
            "small_body_base": 0.18,    # ↑ Slightly higher threshold for small bodies to reduce noise
            "shadow_ratio_base": 1.3    # ↑ Adjusted base shadow ratio to account for longer shadows in dataset
        },
        "signals": {
            "timeframes": {
                "Swing": {"momentum": 0.11},  # ↑ Slightly stronger momentum for Swing timeframe
                "Long": {"momentum": 0.22}    # ↑ Increased momentum requirement for Long timeframe
            },
            "penny_stock_adjustment": {
                "threshold": 1.5,   # ↑ Higher price threshold for penny stock adjustment
                "factor": 0.22      # ↑ Slightly stronger adjustment factor for user strategy
            }
        }
    }
    
    # Overall reason for changes:
    # ---------------------------
    # Each user override is tailored to specific trading strategies:
    # - Adjust candle thresholds (Doji, Hammer, Marubozu, Long/Small bodies) for sensitivity
    # - Modify phase topN to select more/less candidates for aggressive/conservative strategies
    # - Tune moving averages, rate-of-change thresholds, and slope horizons to match user preferences
    # - Modify momentum requirements and penny stock adjustments for timeframe-specific sensitivity


}
