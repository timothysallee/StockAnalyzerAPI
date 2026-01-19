# StockAnalyzerAPI
# Stock Scanner & Trading Bot

A comprehensive Python-based trading system that implements a 3-stage evaluation process (Direction → Goal → Momentum) for intraday trading.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Trading Bot                             │
│                 (main_trading_bot.py)                       │
└──────────────────┬──────────────────────────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
    ▼              ▼              ▼
┌─────────┐  ┌──────────┐  ┌─────────────┐
│Scanner  │  │Strategy  │  │TD Ameritrade│
│Module   │  │Engine    │  │API Client   │
└─────────┘  └──────────┘  └─────────────┘
```

## Features

### 1. Stock Scanner (`stock_scanner_module.py`)
Filters stocks based on:
- Price range: $2-$30
- Shares outstanding: < 10 million
- Volume: 5x average or higher
- Day performance: +3% from open
- 1-minute momentum: +1% on current candle

### 2. Strategy Engine (`stock_scanner_strategy.py`)

**Stage 1: Direction**
- Evaluates trend from 1-minute candles
- Calculates magnitude (% change) and volatility
- Assigns 0-100% probability based on direction strength

**Stage 2: Goal**
- Identifies support/resistance levels
- Clusters price levels within 1% proximity
- Sets stop-loss and take-profit targets
- Evaluates profit potential vs. 3% goal

**Stage 3: Momentum**
- Monitors volume and price delta
- Ensures alignment with entry direction
- Adjusts probability based on momentum strength

### 3. TD Ameritrade Integration (`td_ameritrade_integration.py`)
- OAuth authentication
- Real-time quotes and price history
- Order placement (market, limit, bracket orders)
- Fundamental data retrieval

### 4. Main Trading Bot (`main_trading_bot.py`)
- Orchestrates scanning and trading
- Generates alerts for high-probability setups
- Monitors active positions
- Optional automatic trade execution

## Installation

### Prerequisites
- Python 3.8+
- TD Ameritrade account with API access
- $25,000+ account balance (for PDT-free day trading)

### Setup

1. **Clone your repository**
```bash
cd StockAnalyzerAPI
```

2. **Install dependencies**
```bash
pip install pandas numpy requests
```

3. **Get TD Ameritrade API credentials**
   - Visit [TD Ameritrade Developer Portal](https://developer.tdameritrade.com)
   - Create an app to get your API key (Consumer Key)
   - Set redirect URI to `https://localhost`

4. **Configure the bot**
```bash
python main_trading_bot.py --mode single
```

This will create a `config.json` file. Update it with:
```json
{
  "api_key": "YOUR_API_KEY",
  "account_id": "YOUR_ACCOUNT_ID",
  "watchlist": ["TICKER1", "TICKER2", ...],
  "auto_trade": false,
  ...
}
```

5. **Authenticate**

On first run, the bot will provide an authorization URL:
```python
from td_ameritrade_integration import TDAmeritrade

td = TDAmeritrade(api_key="YOUR_API_KEY")
print(td.get_authorization_url())
# Visit this URL in browser
```

After authorizing, you'll be redirected to `https://localhost?code=...`
Copy the code parameter and exchange it for a refresh token:

```python
td.get_access_token(authorization_code="YOUR_CODE")
print(td.refresh_token)  # Save this!
```

Add the refresh token to `config.json`:
```json
{
  "refresh_token": "YOUR_REFRESH_TOKEN"
}
```

## Usage

### Single Scan Mode (Testing)
```bash
python main_trading_bot.py --mode single
```

Runs one scan cycle and exits. Great for testing.

### Continuous Mode (Live Trading)
```bash
python main_trading_bot.py --mode continuous
```

Runs continuously during market hours:
- Scans watchlist every 60 seconds (configurable)
- Generates alerts for qualifying stocks
- Auto-trades if enabled
- Monitors active positions

### Manual Integration

```python
from main_trading_bot import TradingBot

# Initialize
bot = TradingBot(config_path="config.json")

# Run single scan
results, alerts = bot.run_single_scan()

# View results
print(results)

# Save history
bot.save_results("my_results.csv")
```

## Configuration Options

Edit `config.json`:

```json
{
  "api_key": "YOUR_API_KEY",
  "refresh_token": "YOUR_REFRESH_TOKEN",
  "account_id": "YOUR_ACCOUNT_ID",
  
  "watchlist": ["AAPL", "TSLA", "NVDA"],
  
  "min_price": 2.0,
  "max_price": 30.0,
  "max_shares": 10000000,
  "min_volume_multiplier": 5.0,
  "min_day_change_pct": 3.0,
  "min_1min_change_pct": 1.0,
  
  "target_profit_pct": 3.0,
  "max_position_size": 1000,
  "max_positions": 5,
  
  "auto_trade": false,
  "scan_interval": 60,
  
  "trading_hours": {
    "start": "09:30",
    "end": "16:00"
  }
}
```

## Safety Features

- **Manual mode by default**: `auto_trade: false` prevents automatic trading
- **Position limits**: Configurable max positions and position sizes
- **Market hours only**: Only trades during configured hours
- **Error handling**: Graceful degradation on API errors
- **Logging**: All scans and alerts are logged

## Example Output

```
============================================================
Scanning at 2026-01-19 10:30:45
============================================================
Fetching data for 50 symbols...

Found 3 candidates:
Ticker  Price  Day %  Direction  Dir Prob  Goal Prob  Mom Prob  Overall  Enter
STOCK1  15.50   3.2%   positive      85%       75%       100%     86.5%   True
STOCK2  25.00   4.1%   positive      92%       80%       100%     90.4%   True
STOCK3  12.75   3.5%   positive      78%       65%        50%     64.1%  False

🔔 ALERT: STOCK1
   Price: $15.50
   Overall Score: 86.5%
   Stop Loss: $15.10
   Take Profit: $16.05
   Profit Potential: 3.5%

🔔 ALERT: STOCK2
   Price: $25.00
   Overall Score: 90.4%
   Stop Loss: $24.25
   Take Profit: $26.00
   Profit Potential: 4.0%
```

## Advanced Customization

### Custom Scanner Criteria
```python
from stock_scanner_module import ScannerCriteria

criteria = ScannerCriteria(
    min_price=5.0,
    max_price=20.0,
    min_volume_multiplier=10.0
)
```

### Adjust Strategy Thresholds
```python
from stock_scanner_strategy import TradingStrategy

strategy = TradingStrategy(
    target_profit_pct=5.0,  # Require 5% profit potential
    min_direction_prob=60,
    min_goal_prob=60,
    min_momentum_prob=60
)
```

### Custom Support/Resistance Detection
```python
from stock_scanner_strategy import SupportResistanceDetector

detector = SupportResistanceDetector(
    cluster_threshold=0.005  # 0.5% clustering instead of 1%
)
```

## File Structure

```
StockAnalyzerAPI/
├── config.json                      # Configuration file
├── main_trading_bot.py             # Main orchestrator
├── stock_scanner_module.py         # Scanner with criteria filtering
├── stock_scanner_strategy.py       # 3-stage strategy engine
├── td_ameritrade_integration.py    # API client
├── scan_results.csv                # Results history
└── README.md                       # This file
```

## Backtesting

While this system is designed for live trading, you can test it on historical data:

```python
from stock_scanner_strategy import TradingStrategy
import pandas as pd

# Load historical data
df = pd.read_csv("historical_data.csv", index_col='datetime', parse_dates=True)

# Test strategy on each candle
strategy = TradingStrategy()
for i in range(len(df)):
    current = df.iloc[i]
    history = df.iloc[:i+1]
    
    result = strategy.evaluate_stock(
        ticker="TEST",
        current_candle=current,
        intraday_df=history,
        prev_day_high=26.5,
        prev_day_low=24.0,
        avg_volume=250000
    )
    
    print(f"{current.name}: Score={result.overall_score:.1f}%, Enter={result.should_enter}")
```

## Troubleshooting

### Authentication Issues
- Ensure API key is correct
- Refresh token expires after 90 days - get new one
- Check redirect URI matches exactly

### No Stocks Passing Scanner
- Check if market is open
- Verify watchlist has active stocks
- Reduce `min_day_change_pct` or `min_volume_multiplier`

### API Rate Limits
- TD Ameritrade has rate limits (120 requests/minute)
- Reduce scan frequency or watchlist size
- Add delays between API calls

## Disclaimer

This software is for educational purposes. Trading stocks involves substantial risk of loss. Past performance does not guarantee future results. Always test thoroughly with paper trading before risking real capital.

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## License

MIT License - See LICENSE file for details

## Support

For issues or questions:
- GitHub Issues: [Your repo issues page]
- TD Ameritrade API Docs: https://developer.tdameritrade.com/apis
