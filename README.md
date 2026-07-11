
## Objective

Explore how trader profitability, win rate, position sizing, and trading activity change across different market sentiment regimes (Extreme Fear, Fear, Neutral, Greed, Extreme Greed), and uncover patterns that could inform smarter trading strategies.

## Datasets Used

1. **Bitcoin Fear & Greed Index** — daily sentiment classification (Feb 2018 – May 2025)
2. **Hyperliquid Historical Trader Data** — 211,224 individual trades across 32 accounts and 246 assets (May 2023 – May 2025), including execution price, size, side, closed PnL, fees, and timestamps

## Files in this Repository

| File | Description |

| `Trader_Sentiment_Analysis_Report.docx` | Full analysis report with methodology, charts, key findings, and strategic recommendations |
| `analysis_code.py` | Python script used to merge datasets, compute metrics, and generate all charts |

## Key Findings

- Traders were most profitable during **"Fear"** days (33% of total profit from 29% of trades)
- **"Extreme Fear"** was the weakest regime — lowest win rate (76%) and lowest profit per trade
- Win rate followed a **U-shaped pattern** — highest during Extreme Greed (89%) and Fear (87%)
- Trading activity is **negatively correlated with sentiment value** (≈ −0.25) — traders were more active during fearful conditions, indicating contrarian behavior
- Average position size was largest during Fear ($7,816) and smallest during Extreme Greed ($3,112)

## Tools Used

- Python (pandas, matplotlib)
- Data merged on calendar date between the two datasets

