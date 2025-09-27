CONFIG = {
    # -------------------------
    # Default config (baseline)
    # -------------------------
    "timeframe_map": {
            "Daily": 1,
            "Short": 3,
            "Swing": 5,
            "Long": 10,
        },

    
    "default": {

        "phases": {
            "phase1": {"topN": 20},
            "phase2": {"topN": 10},
            "phase3": {"topN": 5},
        },

        "profiles": {
            "Short": {"ma": 2, "ret": 1, "vol": 3, "roc_thresh": 0.02, "slope_horizon": 1},
            "Swing": {"ma": 5, "ret": 5, "vol": 5, "roc_thresh": 0.02, "slope_horizon": 5},
            "Long":  {"ma": 10, "ret": 10, "vol": 10, "roc_thresh": 0.02, "slope_horizon": 10},
            "Daily": {"ma": 7, "ret": 1, "vol": 5, "roc_thresh": 0.02, "slope_horizon": 1},
        },

       "candle_params": {
            "doji_base": 0.01,
            "doji_scale": 0.02,
            "doji_min": 0.01,
            "doji_max": 0.1,

            "long_body_base": 0.3,
            "long_body_scale": 0.3,
            "long_body_min": 0.3,
            "long_body_max": 0.6,

            "small_body_base": 0.15,
            "small_body_scale": 0.1,
            "small_body_min": 0.15,
            "small_body_max": 0.25,

            "shadow_ratio_base": 1.2,
            "shadow_ratio_scale": 0.8,
            "shadow_ratio_min": 1.2,
            "shadow_ratio_max": 2.0,

            "near_edge": 0.25,
            "highvol_spike": 1.5,
            "lowvol_dip": 0.7,

            "hammer_base": 0.15,
            "hammer_scale": 0.1,
            "hammer_min": 0.15,
            "hammer_max": 0.25,

            "marubozu_base": 0.03,
            "marubozu_scale": 0.02,
            "marubozu_min": 0.03,
            "marubozu_max": 0.05,

            "rng_base": 1e-5,
            "rng_scale": 1e-4,
            "rng_min": 1e-5,
            "rng_max": 1e-4,
        },
        "candle_patterns_NOTUSED": {
            "pattern_window": 5,
            "doji_thresh": 0.1,
            "hammer_thresh": 0.25,
            "marubozu_thresh": 0.05,
            "long_body": 0.6,
            "small_body": 0.25,
            "shadow_ratio": 2.0,
            "near_edge": 0.25,
            "rng_thresh": 1e-4,
        },
        # ------------------------
        # Candle & Signal parameters
        # ------------------------
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
                          "momentum": 0.05},
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
                "threshold": 1.0,
                "factor": 0.2,
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
            "valuation": 0.3,
            "profitability": 0.3,
            "DebtLiquidity": 0.2,
            "Growth": 0.2,
            "Sentiment": 0.1
        },
    },

    # -------------------------
    # User-specific overrides
    # -------------------------
    "user1": {
         "candle_params": {
            "doji_base": 0.02,
            "doji_max": 0.08,
            "long_body_base": 0.35
        },
        "phases": {
            "phase1": {"topN": 30}  # override phase1 topN
        }
    },

    "user2": {
        "profiles": {
            "Swing": {  # override Swing profile
                "ma": 6, "ret": 6, "vol": 6, "roc_thresh": 0.03, "slope_horizon": 6
            }
        },
        "candle_params": {
            "doji_thresh": 0.08  # override candle threshold
        }
    },
       "signals": {
            "timeframes": {
                "Short": {"momentum": 0.07},  # only override momentum
            },
            "penny_stock_adjustment": {
                "factor": 0.3  # override only the factor
            }
        },
}


