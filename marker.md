test git
✅ Explanation of Markers (Why they exist):
Doji – Identifies indecision candles with very small bodies relative to range. Useful to spot potential reversals.
Hammer / HangingMan – Long lower shadow, small body; bullish after downtrend (Hammer) or bearish after uptrend (HangingMan).
InvertedHammer / ShootingStar – Long upper shadow, small body; bullish after downtrend or bearish after uptrend.
Bullish/Bearish Marubozu – Candles with large bodies and minimal shadows, indicating strong continuation.
SuspiciousCandle – Detects tiny range or tiny body, potentially unreliable or “noise” candles.
Engulfing / Harami / HaramiCross / Piercing / DarkCloud / Morning / Evening Star / Three White / Three Black – Multi-bar reversal patterns signaling trend changes.
TweezerTop / TweezerBottom – Exact high/low matching prior candle; potential reversal signals.
InsideBar / OutsideBar – Measures price compression or expansion for continuation/reversal.
NearHigh / NearLow – Rolling window high/low detection, often used to validate momentum continuation or reversal points.
PatternCount / PatternType – Summary metrics: how many patterns fire and which key type is detected, useful for combined signal scoring.

| **Pattern**              | **Meaning / Shape**                                                 | **Why It’s Valuable**                               | **Typical Context / Use**                                                             |
| ------------------------ | ------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **Doji**                 | Very small body relative to range                                   | Flags indecision; neither bulls nor bears dominate  | Precedes potential reversal or consolidation; watch following candle for confirmation |
| **Hammer**               | Long lower shadow, small body, after downtrend                      | Indicates potential bullish reversal                | Use to identify support; stronger when combined with low price volume                 |
| **Hanging Man**          | Long lower shadow, small body, after uptrend                        | Warns of potential bearish reversal                 | Acts as early warning; often confirmed by next candle direction                       |
| **Inverted Hammer**      | Long upper shadow, small body, after downtrend                      | Indicates potential bullish reversal                | Signals failed selling pressure; follow-up candle confirms trend                      |
| **Shooting Star**        | Long upper shadow, small body, after uptrend                        | Indicates potential bearish reversal                | Shows failed buying pressure; strong reversal signal when confirmed                   |
| **Bullish Marubozu**     | Large body, minimal shadows, bullish                                | Confirms strong buying pressure                     | Trend continuation signal; high confidence bullish candle                             |
| **Bearish Marubozu**     | Large body, minimal shadows, bearish                                | Confirms strong selling pressure                    | Trend continuation signal; high confidence bearish candle                             |
| **SuspiciousCandle**     | Tiny body or tiny range                                             | Flags unreliable / “noise” candles                  | Helps filter low-confidence patterns that may distort scoring                         |
| **Bullish Engulfing**    | Current candle fully engulfs prior bearish candle                   | Powerful bullish reversal indicator                 | Use after downtrend; confirms buyer dominance                                         |
| **Bearish Engulfing**    | Current candle fully engulfs prior bullish candle                   | Powerful bearish reversal indicator                 | Use after uptrend; confirms seller dominance                                          |
| **Bullish Harami**       | Small bullish candle within prior bearish body                      | Subtle bullish reversal signal                      | Early trend shift detection; needs confirmation from next candle                      |
| **Bearish Harami**       | Small bearish candle within prior bullish body                      | Subtle bearish reversal signal                      | Early trend shift detection; watch next candle                                        |
| **Harami Cross**         | Doji within prior candle’s body                                     | Strong indecision reversal pattern                  | Confirms potential reversal; often used with trend filters                            |
| **Piercing Line**        | Bullish second candle closes above midpoint of prior bearish candle | Early bullish reversal                              | Used for entry signals with defined risk levels                                       |
| **Dark Cloud Cover**     | Bearish second candle closes below midpoint of prior bullish candle | Early bearish reversal                              | Useful for early exits or short entries                                               |
| **Morning Star**         | Three-bar bullish reversal (small body between two larger ones)     | Confirms trend reversal with multi-bar confirmation | Reliable reversal pattern after downtrend                                             |
| **Evening Star**         | Three-bar bearish reversal (small body between two larger ones)     | Confirms trend reversal with multi-bar confirmation | Reliable reversal pattern after uptrend                                               |
| **Three White Soldiers** | Three consecutive bullish candles with rising closes                | Strong trend continuation / bullish confirmation    | Confirms strong upward momentum                                                       |
| **Three Black Crows**    | Three consecutive bearish candles with falling closes               | Strong trend continuation / bearish confirmation    | Confirms strong downward momentum                                                     |
| **Tweezer Top**          | Exact high matches previous candle’s high                           | Potential bearish reversal                          | Marks local resistance / reversal point                                               |
| **Tweezer Bottom**       | Exact low matches previous candle’s low                             | Potential bullish reversal                          | Marks local support / reversal point                                                  |
| **Inside Bar**           | Current high < prior high, current low > prior low                  | Indicates price compression / potential breakout    | Signals continuation or setup for breakout trade                                      |
| **Outside Bar**          | Current high > prior high, current low < prior low                  | Indicates expansion / potential reversal            | Shows strong directional move; can indicate continuation or exhaustion                |
| **Near High**            | Current high near rolling high                                      | Confirms bullish momentum                           | Helps validate continuation; combined with trend signals for stronger entries         |
| **Near Low**             | Current low near rolling low                                        | Confirms bearish momentum                           | Helps validate continuation; combined with trend signals for stronger exits           |
| **PatternCount**         | Total number of patterns firing for candle                          | Quantifies overall signal intensity                 | Higher counts → higher confidence in combined signal scoring                          |
| **PatternType**          | Dominant / key pattern detected                                     | Quickly identifies most relevant signal             | Guides weighted scoring in hybrid signal calculation                                  |


-------
**pd.DataFrame**  
Original DataFrame with the following additional columns:

- **MA** : Rolling moving average  
- **MA_slope** : % change of MA (normalized slope)  
- **UpTrend_MA, DownTrend_MA** : Trend direction based on MA slope  
- **RecentReturn** : % change over return window  
- **UpTrend_Return, DownTrend_Return** : Trend based on % return  
- **Volatility** : Rolling std deviation of % returns  
- **LowVolatility, HighVolatility** : Volatility relative to median  
- **ROC** : Rate of change over MA window  
- **MomentumUp, MomentumDown** : Trend direction based on ROC thresholds  
- **ConfirmedUpTrend, ConfirmedDownTrend** : Combined trend confirmation  

---

| Feature                                   | When It’s Useful                                                     | Why It Matters                                                                                                                     |
| ----------------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **MA (Moving Average)**                   | Always, baseline trend detection                                     | Smooths out price noise; shows average price over a rolling window.                                                                |
| **MA_slope (normalized % slope)**         | When you want to compare momentum across penny stocks vs. large caps | Converts slope into a % change → scale-invariant. Prevents false signals from absolute price levels.                               |
| **UpTrend_MA / DownTrend_MA**             | Short- to long-term trend following                                  | Simple binary indicator: is the average slope pointing up or down? Good for building rules.                                        |
| **RecentReturn (% change over window)**   | Detecting short-term momentum                                        | Captures raw % gain/loss over the return window. Helps spot sudden moves.                                                          |
| **UpTrend_Return / DownTrend_Return**     | For classifying recent moves into binary up/down trends              | Makes recent return usable in filters, strategies, or ML features.                                                                 |
| **Volatility (rolling std of % returns)** | Risk assessment, breakout strategies, stop-loss sizing               | Captures recent price variability. Useful for distinguishing calm vs. choppy periods.                                              |
| **LowVolatility / HighVolatility**        | Position sizing or strategy switching                                | Flags if volatility is above/below median → helps adjust trading style (mean-reversion vs breakout).                               |
| **ROC (Rate of Change)**                  | Medium-term momentum                                                 | Measures speed of price change, similar to momentum indicators in TA.                                                              |
| **MomentumUp / MomentumDown**             | Breakout or reversal confirmation                                    | Binary flag: is ROC strong enough beyond a threshold (e.g., 2%)? Filters out weak/noisy moves.                                     |
| **ConfirmedUpTrend / ConfirmedDownTrend** | Signal validation for entries/exits                                  | Strongest signal → requires alignment of MA slope, returns, and momentum. Helps avoid false positives from single-indicator moves. |
"

'''
Raw Price Data (Open, High, Low, Close)
           |
           v
---------------------------
1️⃣ Compute Momentum Metrics
   - Return = pct_change(Close)
   - AvgReturn = rolling mean
   - Volatility = rolling std
   - MomentumZ = (Return - AvgReturn) / Volatility
           |
           v
2️⃣ Determine MomentumAction
   - Buy if MomentumZ > BuyThresh
   - Sell if MomentumZ < SellThresh
   - Hold otherwise
           |
           v
---------------------------
3️⃣ Pattern Scoring (Bullish/Bearish Patterns)
   - Rolling sum of confirmed patterns over window
   - Score = Bull - Bear
   - Normalize → PatternScoreNorm
           |
           v
4️⃣ Determine PatternAction
   - Buy if PatternScoreNorm > threshold
   - Sell if PatternScoreNorm < -threshold
   - Hold otherwise
           |
           v
---------------------------
5️⃣ Candlestick Pattern Signals
   - Check candle_columns["Buy"] / ["Sell"]
   - Classify → CandleAction (Buy/Sell/Hold)
           |
           v
---------------------------
6️⃣ CandidateAction
   - Majority vote of MomentumAction, PatternAction, CandleAction
           |
           v
7️⃣ Filter Consecutive Signals
   - If same as previous Buy/Sell → convert to Hold
           |
           v
8️⃣ Determine TomorrowAction
   - TomorrowAction = Action.shift(-1)
   - Trace source: filtered vs unfiltered CandidateAction
           |
           v
---------------------------
9️⃣ Compute Signal Strength / Confidence
   - Count-based: # of valid patterns firing
   - Magnitude-based: PatternScore + MomentumZ
   - Weighted combination → SignalStrengthHybrid
   - ActionConfidence aligned with Action direction
           |
           v
---------------------------
Final Output Columns:
   - Action: today’s filtered action (Buy/Sell/Hold)
   - TomorrowAction: predicted next day action
   - CandidateAction: majority vote before filtering
   - ActionConfidence: hybrid strength score

'''

Price Data (Close)
       │
       ▼
 ┌─────────────┐
 │ Momentum    │───> MomentumAction (Buy/Sell/Hold)
 └─────────────┘
       │
       ▼
 ┌─────────────┐
 │ Patterns    │───> PatternAction (Buy/Sell/Hold)
 └─────────────┘
       │
       ▼
 ┌─────────────┐
 │ Candlesticks│───> CandleAction (Buy/Sell/Hold)
 └─────────────┘
       │
       ▼
 ┌───────────────────┐
 │ Majority Voting   │───> CandidateAction
 └───────────────────┘
       │
       ▼
 ┌───────────────────┐
 │ Consecutive Filter│───> Action
 └───────────────────┘
       │
       ▼
 ┌───────────────────┐
 │ Tomorrow Shift    │───> TomorrowAction
 └───────────────────┘
       │
       ▼
 ┌───────────────────┐
 │ Hybrid Strength   │───> ActionConfidence
 └───────────────────┘


| Category             | Feature Example                                          | Interpretation                                                              | Weight |
| -------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------- | ------ |
| **Valuation**        | `PeRatio`, `PegRatio`, `PbRatio`                         | Lower = better                                                              | 0.2    |
| **Profitability**    | `ReturnOnEquity`, `GrossMarginTTM`, `NetProfitMarginTTM` | Higher = better                                                             | 0.3    |
| **Debt & Liquidity** | `TotalDebtToEquity`, `CurrentRatio`, `InterestCoverage`  | Lower debt, higher liquidity = better                                       | 0.2    |
| **Growth**           | `EpsChangeYear`, `RevChangeYear`                         | Higher = better                                                             | 0.2    |
| **Sentiment / Risk** | `Beta`, `ShortIntToFloat`                                | Depends (low beta = stable, high short interest can be bullish if breakout) | 0.1    |
