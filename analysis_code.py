import pandas as pd
import numpy as np

pd.set_option('display.width', 150)

# Load data
trades = pd.read_csv('/mnt/user-data/uploads/historical_data__1_.csv')
fg = pd.read_csv('/mnt/user-data/uploads/fear_greed_index__1_.csv')

# Parse dates
trades['date'] = pd.to_datetime(trades['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce').dt.date
fg['date'] = pd.to_datetime(fg['date']).dt.date

# Merge
merged = trades.merge(fg[['date','classification','value']], on='date', how='left')
print("Rows before merge:", len(trades), "after merge:", len(merged))
print("Unmatched dates (no sentiment):", merged['classification'].isna().sum())

merged.to_csv('/home/claude/merged.csv', index=False)
print(merged['classification'].value_counts(dropna=False))
import pandas as pd
import numpy as np

merged = pd.read_csv('/home/claude/merged.csv')
merged = merged.dropna(subset=['classification'])

order = ['Extreme Fear','Fear','Neutral','Greed','Extreme Greed']

# Only rows with a nonzero closed PnL represent "closing" trades where profit/loss is realized
closes = merged[merged['Closed PnL'] != 0].copy()
print("Total trades:", len(merged), " | Trades with realized PnL:", len(closes))

# 1. Overall PnL by sentiment
summary = merged.groupby('classification').agg(
    total_trades=('Closed PnL','count'),
    total_volume_usd=('Size USD','sum'),
    avg_trade_size_usd=('Size USD','mean'),
    total_closed_pnl=('Closed PnL','sum'),
    avg_closed_pnl=('Closed PnL','mean'),
).reindex(order)
print("\n=== Overall summary by sentiment ===")
print(summary)

# 2. Win rate among trades that closed a position (non-zero PnL)
closes['win'] = closes['Closed PnL'] > 0
winrate = closes.groupby('classification')['win'].mean().reindex(order)
print("\n=== Win rate (of closed-PnL trades) by sentiment ===")
print(winrate)

# 3. Average PnL per closing trade (only nonzero)
avgclosepnl = closes.groupby('classification')['Closed PnL'].agg(['mean','median','sum','count']).reindex(order)
print("\n=== Closed PnL stats (nonzero PnL trades only) ===")
print(avgclosepnl)

# 4. Buy vs Sell side distribution by sentiment
side_dist = pd.crosstab(merged['classification'], merged['Side'], normalize='index').reindex(order)
print("\n=== Buy/Sell side distribution by sentiment ===")
print(side_dist)

# 5. Average trade size (leverage proxy) by sentiment
print("\n=== Avg position size (USD) by sentiment ===")
print(merged.groupby('classification')['Size USD'].mean().reindex(order))

summary.to_csv('/home/claude/summary_overall.csv')
winrate.to_csv('/home/claude/summary_winrate.csv')
avgclosepnl.to_csv('/home/claude/summary_avgclosepnl.csv')
import pandas as pd
import numpy as np

merged = pd.read_csv('/home/claude/merged.csv')
merged = merged.dropna(subset=['classification'])
merged['date'] = pd.to_datetime(merged['date'])

order = ['Extreme Fear','Fear','Neutral','Greed','Extreme Greed']

# Daily aggregation
daily = merged.groupby('date').agg(
    daily_pnl=('Closed PnL','sum'),
    daily_volume=('Size USD','sum'),
    n_trades=('Closed PnL','count'),
    sentiment_value=('value','first'),
    classification=('classification','first')
).reset_index()

# Correlation between sentiment numeric value and daily PnL / volume
corr_pnl = daily['sentiment_value'].corr(daily['daily_pnl'])
corr_vol = daily['sentiment_value'].corr(daily['daily_volume'])
corr_trades = daily['sentiment_value'].corr(daily['n_trades'])
print("Correlation sentiment_value vs daily_pnl:", corr_pnl)
print("Correlation sentiment_value vs daily_volume:", corr_vol)
print("Correlation sentiment_value vs n_trades:", corr_trades)

daily.to_csv('/home/claude/daily.csv', index=False)

# Per-account behavior: does each account perform differently in fear vs greed?
acct = merged.groupby(['Account','classification'])['Closed PnL'].sum().unstack().reindex(columns=order)
acct['total'] = acct.sum(axis=1)
acct_sorted = acct.sort_values('total', ascending=False)
print("\n=== Top 10 accounts total PnL, broken by sentiment ===")
print(acct_sorted.head(10))

# Top coins analysis
top_coins = ['BTC','ETH','SOL','HYPE']
for c in top_coins:
    sub = merged[merged['Coin']==c]
    g = sub.groupby('classification')['Closed PnL'].sum().reindex(order)
    print(f"\n=== {c}: total closed PnL by sentiment ===")
    print(g)

# Leverage proxy: Size USD / |Start Position| change isn't leverage; skip since no margin column.
# Fee analysis
fee_summary = merged.groupby('classification')['Fee'].agg(['sum','mean']).reindex(order)
print("\n=== Fees by sentiment ===")
print(fee_summary)
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

merged = pd.read_csv('/home/claude/merged.csv')
merged = merged.dropna(subset=['classification'])
daily = pd.read_csv('/home/claude/daily.csv')
daily['date'] = pd.to_datetime(daily['date'])

order = ['Extreme Fear','Fear','Neutral','Greed','Extreme Greed']
colors = ['#8B0000','#E74C3C','#95A5A6','#27AE60','#145A32']

plt.rcParams['font.family'] = 'DejaVu Sans'

# Chart 1: Total Closed PnL by sentiment
fig, ax = plt.subplots(figsize=(8,5))
tot_pnl = merged.groupby('classification')['Closed PnL'].sum().reindex(order)
bars = ax.bar(order, tot_pnl.values/1e6, color=colors)
ax.set_ylabel('Total Closed PnL ($ Millions)')
ax.set_title('Total Trader Profit/Loss by Market Sentiment', fontsize=13, fontweight='bold')
ax.axhline(0, color='black', linewidth=0.8)
for b,v in zip(bars, tot_pnl.values/1e6):
    ax.text(b.get_x()+b.get_width()/2, v + (0.05 if v>=0 else -0.15), f'${v:.2f}M', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('charts/1_total_pnl_by_sentiment.png', dpi=150)
plt.close()

# Chart 2: Win rate by sentiment
closes = merged[merged['Closed PnL']!=0].copy()
closes['win'] = closes['Closed PnL']>0
winrate = closes.groupby('classification')['win'].mean().reindex(order)*100
fig, ax = plt.subplots(figsize=(8,5))
bars = ax.bar(order, winrate.values, color=colors)
ax.set_ylabel('Win Rate (%)')
ax.set_title('Win Rate of Closed Trades by Market Sentiment', fontsize=13, fontweight='bold')
ax.set_ylim(0,100)
for b,v in zip(bars, winrate.values):
    ax.text(b.get_x()+b.get_width()/2, v+1, f'{v:.1f}%', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('charts/2_winrate_by_sentiment.png', dpi=150)
plt.close()

# Chart 3: Trade count & volume by sentiment (dual axis)
agg = merged.groupby('classification').agg(trades=('Closed PnL','count'), volume=('Size USD','sum')).reindex(order)
fig, ax1 = plt.subplots(figsize=(8,5))
bars = ax1.bar(order, agg['trades'], color='#3498DB', alpha=0.8, label='Trade Count')
ax1.set_ylabel('Number of Trades', color='#2C3E50')
ax2 = ax1.twinx()
ax2.plot(order, agg['volume']/1e6, color='#E67E22', marker='o', linewidth=2.5, label='Volume ($M)')
ax2.set_ylabel('Total Trading Volume ($ Millions)', color='#E67E22')
ax1.set_title('Trading Activity by Market Sentiment', fontsize=13, fontweight='bold')
fig.legend(loc='upper right', bbox_to_anchor=(0.9,0.88))
plt.tight_layout()
plt.savefig('charts/3_activity_by_sentiment.png', dpi=150)
plt.close()

# Chart 4: Average trade size by sentiment
avgsize = merged.groupby('classification')['Size USD'].mean().reindex(order)
fig, ax = plt.subplots(figsize=(8,5))
bars = ax.bar(order, avgsize.values, color=colors)
ax.set_ylabel('Avg Position Size (USD)')
ax.set_title('Average Position Size by Market Sentiment', fontsize=13, fontweight='bold')
for b,v in zip(bars, avgsize.values):
    ax.text(b.get_x()+b.get_width()/2, v+50, f'${v:,.0f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('charts/4_avgsize_by_sentiment.png', dpi=150)
plt.close()

# Chart 5: Per-coin PnL by sentiment for top coins
coins = ['BTC','ETH','SOL','HYPE']
data = {}
for c in coins:
    sub = merged[merged['Coin']==c]
    data[c] = sub.groupby('classification')['Closed PnL'].sum().reindex(order)/1000
df_coins = pd.DataFrame(data)
fig, ax = plt.subplots(figsize=(10,5.5))
x = range(len(order))
width = 0.2
for i,c in enumerate(coins):
    ax.bar([p + width*i for p in x], df_coins[c].values, width=width, label=c)
ax.set_xticks([p + width*1.5 for p in x])
ax.set_xticklabels(order)
ax.set_ylabel('Total Closed PnL ($ Thousands)')
ax.set_title('Closed PnL by Sentiment — Top Traded Coins', fontsize=13, fontweight='bold')
ax.axhline(0, color='black', linewidth=0.8)
ax.legend()
plt.tight_layout()
plt.savefig('charts/5_coin_pnl_by_sentiment.png', dpi=150)
plt.close()

# Chart 6: Daily PnL vs sentiment value over time (dual axis time series)
daily_sorted = daily.sort_values('date')
daily_sorted['pnl_7d'] = daily_sorted['daily_pnl'].rolling(7, min_periods=1).mean()
daily_sorted['sent_7d'] = daily_sorted['sentiment_value'].rolling(7, min_periods=1).mean()
fig, ax1 = plt.subplots(figsize=(11,5))
ax1.plot(daily_sorted['date'], daily_sorted['pnl_7d'], color='#2980B9', linewidth=1.5, label='7-day avg Daily PnL')
ax1.set_ylabel('7-day Avg Daily Closed PnL ($)', color='#2980B9')
ax1.axhline(0, color='gray', linewidth=0.6)
ax2 = ax1.twinx()
ax2.plot(daily_sorted['date'], daily_sorted['sent_7d'], color='#C0392B', linewidth=1.5, alpha=0.7, label='7-day avg Sentiment Value')
ax2.set_ylabel('7-day Avg Fear/Greed Index Value (0-100)', color='#C0392B')
ax1.set_title('Trader PnL vs Market Sentiment Over Time', fontsize=13, fontweight='bold')
fig.legend(loc='upper left', bbox_to_anchor=(0.08,0.95))
plt.tight_layout()
plt.savefig('charts/6_timeseries_pnl_sentiment.png', dpi=150)
plt.close()

print("Charts saved.")
import os
print(os.listdir('charts'))
