"""
Main Trading Bot
Orchestrates scanner, strategy, and trade execution
"""

import time
import json
from datetime import datetime, time as dt_time
from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path

# Import your modules (adjust paths as needed for your repo)
# from td_ameritrade_integration import TDAmeritrade, DataFetcher
# from stock_scanner_module import StockScanner, ScannerCriteria, IntegratedTradingSystem
# from stock_scanner_strategy import TradingStrategy


class TradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize trading bot
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self.load_config(config_path)
        
        # Initialize TD Ameritrade client
        self.td = TDAmeritrade(
            api_key=self.config['api_key'],
            refresh_token=self.config.get('refresh_token')
        )
        
        # Authenticate
        if not self.td.access_token:
            self.td.get_access_token()
        
        # Initialize data fetcher
        self.data_fetcher = DataFetcher(self.td)
        
        # Initialize integrated system
        scanner_criteria = ScannerCriteria(
            min_price=self.config.get('min_price', 2.0),
            max_price=self.config.get('max_price', 30.0),
            max_shares=self.config.get('max_shares', 10_000_000),
            min_volume_multiplier=self.config.get('min_volume_multiplier', 5.0),
            min_day_change_pct=self.config.get('min_day_change_pct', 3.0),
            min_1min_change_pct=self.config.get('min_1min_change_pct', 1.0)
        )
        
        self.system = IntegratedTradingSystem(
            scanner_criteria=scanner_criteria,
            target_profit_pct=self.config.get('target_profit_pct', 3.0)
        )
        
        # Tracking
        self.active_positions = {}
        self.alerts = []
        self.scan_history = []
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        config_file = Path(config_path)
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Create default config
            default_config = {
                "api_key": "YOUR_API_KEY",
                "refresh_token": None,
                "account_id": "YOUR_ACCOUNT_ID",
                "watchlist": ["AAPL", "TSLA", "NVDA"],  # Your watchlist
                "min_price": 2.0,
                "max_price": 30.0,
                "max_shares": 10_000_000,
                "min_volume_multiplier": 5.0,
                "min_day_change_pct": 3.0,
                "min_1min_change_pct": 1.0,
                "target_profit_pct": 3.0,
                "max_position_size": 1000,  # Max shares per position
                "max_positions": 5,  # Max concurrent positions
                "auto_trade": False,  # Set to True for automatic trading
                "scan_interval": 60,  # Seconds between scans
                "trading_hours": {
                    "start": "09:30",
                    "end": "16:00"
                }
            }
            
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            print(f"Created default config at {config_path}")
            print("Please update with your API credentials")
            
            return default_config
    
    def is_market_hours(self) -> bool:
        """Check if current time is within trading hours"""
        now = datetime.now().time()
        start = dt_time.fromisoformat(self.config['trading_hours']['start'])
        end = dt_time.fromisoformat(self.config['trading_hours']['end'])
        return start <= now <= end
    
    def scan_watchlist(self) -> pd.DataFrame:
        """
        Scan watchlist and return evaluation results
        
        Returns:
            DataFrame with scan results
        """
        print(f"\n{'='*60}")
        print(f"Scanning at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # Fetch data for watchlist
        symbols = self.config['watchlist']
        print(f"Fetching data for {len(symbols)} symbols...")
        
        scanner_data = self.data_fetcher.prepare_scanner_data(symbols)
        historical_data, prev_day_data = self.data_fetcher.prepare_strategy_data(symbols)
        
        # Run integrated scan + strategy
        results = self.system.process_watchlist(
            stocks_data=scanner_data,
            historical_data=historical_data,
            prev_day_data=prev_day_data
        )
        
        if not results.empty:
            print(f"\nFound {len(results)} candidates:")
            print(results.to_string(index=False))
            
            # Save to history
            self.scan_history.append({
                'timestamp': datetime.now(),
                'results': results
            })
        else:
            print("No stocks passed screening")
        
        return results
    
    def generate_alerts(self, results: pd.DataFrame) -> List[Dict]:
        """
        Generate alerts for high-probability setups
        
        Args:
            results: Scan results DataFrame
            
        Returns:
            List of alert dictionaries
        """
        if results.empty:
            return []
        
        # Filter for entries
        entry_signals = results[results['Enter'] == True].copy()
        
        alerts = []
        for _, row in entry_signals.iterrows():
            alert = {
                'timestamp': datetime.now(),
                'ticker': row['Ticker'],
                'price': row['Price'],
                'direction_prob': row['Dir Prob'],
                'goal_prob': row['Goal Prob'],
                'momentum_prob': row['Mom Prob'],
                'overall_score': row['Overall'],
                'stop_loss': row['Stop Loss'],
                'take_profit': row['Take Profit'],
                'profit_potential': row['Profit %'],
                'action': 'BUY'
            }
            alerts.append(alert)
            
            # Print alert
            print(f"\n🔔 ALERT: {alert['ticker']}")
            print(f"   Price: ${alert['price']:.2f}")
            print(f"   Overall Score: {alert['overall_score']:.1f}%")
            print(f"   Stop Loss: ${alert['stop_loss']:.2f}")
            print(f"   Take Profit: ${alert['take_profit']:.2f}")
            print(f"   Profit Potential: {alert['profit_potential']:.2f}%")
        
        self.alerts.extend(alerts)
        return alerts
    
    def execute_trade(self, alert: Dict) -> bool:
        """
        Execute trade based on alert
        
        Args:
            alert: Alert dictionary
            
        Returns:
            True if trade executed successfully
        """
        if not self.config.get('auto_trade', False):
            print(f"⚠️  Auto-trade disabled. Manual action required for {alert['ticker']}")
            return False
        
        # Check position limits
        if len(self.active_positions) >= self.config['max_positions']:
            print(f"⚠️  Max positions reached ({self.config['max_positions']})")
            return False
        
        # Calculate position size
        quantity = min(
            self.config['max_position_size'],
            int(self.config.get('position_value', 10000) / alert['price'])
        )
        
        # Create bracket order
        order = self.td.create_bracket_order(
            symbol=alert['ticker'],
            quantity=quantity,
            entry_price=alert['price'],
            stop_loss=alert['stop_loss'],
            take_profit=alert['take_profit']
        )
        
        # Place order
        success = self.td.place_order(
            account_id=self.config['account_id'],
            order=order
        )
        
        if success:
            print(f"✅ Trade executed: {quantity} shares of {alert['ticker']}")
            
            # Track position
            self.active_positions[alert['ticker']] = {
                'entry_time': datetime.now(),
                'entry_price': alert['price'],
                'quantity': quantity,
                'stop_loss': alert['stop_loss'],
                'take_profit': alert['take_profit']
            }
            
            return True
        else:
            print(f"❌ Trade failed for {alert['ticker']}")
            return False
    
    def monitor_positions(self):
        """Monitor active positions and update based on momentum"""
        if not self.active_positions:
            return
        
        print(f"\nMonitoring {len(self.active_positions)} active positions...")
        
        for ticker, position in list(self.active_positions.items()):
            try:
                # Get current data
                quote = self.td.get_quote(ticker)
                current_price = quote['lastPrice']
                
                # Get recent candles for momentum check
                candles = self.td.get_price_history(
                    ticker,
                    period_type='day',
                    period=1,
                    frequency_type='minute',
                    frequency=1
                )
                
                if candles.empty:
                    continue
                
                current_candle = candles.iloc[-1]
                
                # Evaluate momentum
                direction = Direction.POSITIVE  # From entry
                volume_ratio = quote['totalVolume'] / quote.get('avgVolume', 1)
                delta = current_candle['close'] - current_candle['open']
                
                _, _, momentum_prob = self.system.strategy.momentum_eval.evaluate(
                    current_candle['volume'],
                    quote.get('avgVolume', 0),
                    delta,
                    direction
                )
                
                print(f"  {ticker}: ${current_price:.2f} | Momentum: {momentum_prob:.0f}%")
                
                # Decision based on momentum
                if momentum_prob < 25:
                    print(f"    ⚠️  Low momentum - consider exit")
                    # Could implement automatic exit here
                
            except Exception as e:
                print(f"  Error monitoring {ticker}: {e}")
    
    def run_continuous(self):
        """Run bot continuously during market hours"""
        print("Starting trading bot...")
        print(f"Scan interval: {self.config['scan_interval']} seconds")
        print(f"Auto-trade: {self.config.get('auto_trade', False)}")
        
        while True:
            try:
                # Check if market hours
                if not self.is_market_hours():
                    print("Outside trading hours. Waiting...")
                    time.sleep(300)  # Check every 5 minutes
                    continue
                
                # Scan watchlist
                results = self.scan_watchlist()
                
                # Generate alerts
                alerts = self.generate_alerts(results)
                
                # Execute trades if auto-trade enabled
                for alert in alerts:
                    self.execute_trade(alert)
                
                # Monitor existing positions
                self.monitor_positions()
                
                # Wait for next scan
                time.sleep(self.config['scan_interval'])
                
            except KeyboardInterrupt:
                print("\nShutting down...")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(60)  # Wait a minute on error
    
    def run_single_scan(self):
        """Run a single scan (for testing)"""
        results = self.scan_watchlist()
        alerts = self.generate_alerts(results)
        return results, alerts
    
    def save_results(self, filepath: str = "scan_results.csv"):
        """Save scan history to CSV"""
        if not self.scan_history:
            print("No scan history to save")
            return
        
        all_results = pd.concat([
            scan['results'].assign(scan_time=scan['timestamp'])
            for scan in self.scan_history
        ])
        
        all_results.to_csv(filepath, index=False)
        print(f"Saved results to {filepath}")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Trading Bot')
    parser.add_argument('--config', default='config.json', help='Config file path')
    parser.add_argument('--mode', choices=['single', 'continuous'], default='single',
                       help='Run mode: single scan or continuous')
    args = parser.parse_args()
    
    # Initialize bot
    bot = TradingBot(config_path=args.config)
    
    if args.mode == 'single':
        # Run single scan
        results, alerts = bot.run_single_scan()
        print(f"\nGenerated {len(alerts)} alerts")
        
        # Save results
        bot.save_results()
        
    else:
        # Run continuously
        bot.run_continuous()
