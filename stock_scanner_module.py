"""
Stock Scanner Module
Filters stocks based on specified criteria before strategy evaluation
"""

import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, time


@dataclass
class ScannerCriteria:
    """Criteria for stock screening"""
    min_price: float = 2.0
    max_price: float = 30.0
    max_shares: int = 10_000_000  # 10 million shares
    min_volume_multiplier: float = 5.0  # 5x average volume
    min_day_change_pct: float = 3.0  # Up 3% from open
    min_1min_change_pct: float = 1.0  # Up 1% on current candle


@dataclass
class StockCandidate:
    """Represents a stock that passed initial screening"""
    ticker: str
    current_price: float
    day_open: float
    day_change_pct: float
    current_volume: int
    avg_volume: int
    volume_ratio: float
    shares_outstanding: int
    one_min_change_pct: float
    passed_all_criteria: bool
    criteria_met: Dict[str, bool]


class StockScanner:
    """Scans and filters stocks based on multiple criteria"""
    
    def __init__(self, criteria: Optional[ScannerCriteria] = None):
        """
        Initialize scanner with criteria
        
        Args:
            criteria: ScannerCriteria object (uses defaults if None)
        """
        self.criteria = criteria or ScannerCriteria()
    
    def check_price_range(self, price: float) -> bool:
        """Check if price is within acceptable range"""
        return self.criteria.min_price <= price <= self.criteria.max_price
    
    def check_shares_outstanding(self, shares: int) -> bool:
        """Check if shares outstanding is below maximum"""
        return shares < self.criteria.max_shares
    
    def check_volume_multiplier(self, current_volume: int, avg_volume: int) -> bool:
        """Check if current volume is sufficient multiple of average"""
        if avg_volume == 0:
            return False
        return (current_volume / avg_volume) >= self.criteria.min_volume_multiplier
    
    def check_day_change(self, current_price: float, day_open: float) -> bool:
        """Check if stock is up enough from day's open"""
        if day_open == 0:
            return False
        pct_change = ((current_price - day_open) / day_open) * 100
        return pct_change >= self.criteria.min_day_change_pct
    
    def check_1min_change(self, candle_close: float, candle_open: float) -> bool:
        """Check if 1-minute candle shows sufficient increase"""
        if candle_open == 0:
            return False
        pct_change = ((candle_close - candle_open) / candle_open) * 100
        return pct_change >= self.criteria.min_1min_change_pct
    
    def scan_stock(self, ticker: str, current_price: float, day_open: float,
                   current_volume: int, avg_volume: int, shares_outstanding: int,
                   candle_open: float, candle_close: float) -> StockCandidate:
        """
        Evaluate single stock against all criteria
        
        Args:
            ticker: Stock symbol
            current_price: Current stock price
            day_open: Today's opening price
            current_volume: Current day's volume
            avg_volume: Average daily volume
            shares_outstanding: Total shares outstanding
            candle_open: Current 1-min candle open
            candle_close: Current 1-min candle close
            
        Returns:
            StockCandidate with evaluation results
        """
        # Calculate metrics
        day_change_pct = ((current_price - day_open) / day_open * 100) if day_open > 0 else 0
        volume_ratio = (current_volume / avg_volume) if avg_volume > 0 else 0
        one_min_change_pct = ((candle_close - candle_open) / candle_open * 100) if candle_open > 0 else 0
        
        # Check each criterion
        criteria_met = {
            'price_range': self.check_price_range(current_price),
            'shares_limit': self.check_shares_outstanding(shares_outstanding),
            'volume_multiplier': self.check_volume_multiplier(current_volume, avg_volume),
            'day_change': self.check_day_change(current_price, day_open),
            '1min_change': self.check_1min_change(candle_close, candle_open)
        }
        
        passed_all = all(criteria_met.values())
        
        return StockCandidate(
            ticker=ticker,
            current_price=current_price,
            day_open=day_open,
            day_change_pct=day_change_pct,
            current_volume=current_volume,
            avg_volume=avg_volume,
            volume_ratio=volume_ratio,
            shares_outstanding=shares_outstanding,
            one_min_change_pct=one_min_change_pct,
            passed_all_criteria=passed_all,
            criteria_met=criteria_met
        )
    
    def scan_multiple(self, stocks_data: List[Dict]) -> List[StockCandidate]:
        """
        Scan multiple stocks and return candidates
        
        Args:
            stocks_data: List of dictionaries with stock data
                Each dict should contain: ticker, current_price, day_open,
                current_volume, avg_volume, shares_outstanding,
                candle_open, candle_close
        
        Returns:
            List of StockCandidate objects
        """
        candidates = []
        
        for stock in stocks_data:
            candidate = self.scan_stock(
                ticker=stock['ticker'],
                current_price=stock['current_price'],
                day_open=stock['day_open'],
                current_volume=stock['current_volume'],
                avg_volume=stock['avg_volume'],
                shares_outstanding=stock['shares_outstanding'],
                candle_open=stock['candle_open'],
                candle_close=stock['candle_close']
            )
            candidates.append(candidate)
        
        return candidates
    
    def get_passing_stocks(self, stocks_data: List[Dict]) -> List[StockCandidate]:
        """
        Get only stocks that pass all criteria
        
        Args:
            stocks_data: List of stock data dictionaries
            
        Returns:
            List of StockCandidate objects that passed all criteria
        """
        all_candidates = self.scan_multiple(stocks_data)
        return [c for c in all_candidates if c.passed_all_criteria]
    
    def generate_report(self, candidates: List[StockCandidate]) -> pd.DataFrame:
        """
        Generate a report DataFrame from candidates
        
        Args:
            candidates: List of StockCandidate objects
            
        Returns:
            DataFrame with candidate information
        """
        data = []
        for c in candidates:
            data.append({
                'Ticker': c.ticker,
                'Price': c.current_price,
                'Day Change %': c.day_change_pct,
                '1min Change %': c.one_min_change_pct,
                'Volume Ratio': c.volume_ratio,
                'Shares (M)': c.shares_outstanding / 1_000_000,
                'Passed': c.passed_all_criteria,
                'Price OK': c.criteria_met['price_range'],
                'Shares OK': c.criteria_met['shares_limit'],
                'Volume OK': c.criteria_met['volume_multiplier'],
                'Day% OK': c.criteria_met['day_change'],
                '1min% OK': c.criteria_met['1min_change']
            })
        
        return pd.DataFrame(data)


# Integration class to combine scanner + strategy
class IntegratedTradingSystem:
    """Combines scanner and strategy evaluation"""
    
    def __init__(self, scanner_criteria: Optional[ScannerCriteria] = None,
                 target_profit_pct: float = 3.0):
        """
        Initialize integrated system
        
        Args:
            scanner_criteria: Criteria for initial filtering
            target_profit_pct: Target profit percentage for strategy
        """
        self.scanner = StockScanner(scanner_criteria)
        # Import here to avoid circular dependency
        from stock_scanner_strategy import TradingStrategy
        self.strategy = TradingStrategy(target_profit_pct=target_profit_pct)
    
    def process_watchlist(self, stocks_data: List[Dict], 
                         historical_data: Dict[str, pd.DataFrame],
                         prev_day_data: Dict[str, Dict]) -> pd.DataFrame:
        """
        Process entire watchlist through scanner and strategy
        
        Args:
            stocks_data: List of current stock data
            historical_data: Dict mapping ticker to intraday DataFrame
            prev_day_data: Dict mapping ticker to {'high': float, 'low': float}
            
        Returns:
            DataFrame with full evaluation results
        """
        # Step 1: Initial screening
        passing_candidates = self.scanner.get_passing_stocks(stocks_data)
        
        if not passing_candidates:
            print("No stocks passed initial screening")
            return pd.DataFrame()
        
        # Step 2: Strategy evaluation for passing stocks
        results = []
        for candidate in passing_candidates:
            ticker = candidate.ticker
            
            # Get historical data for this stock
            if ticker not in historical_data or ticker not in prev_day_data:
                continue
            
            intraday_df = historical_data[ticker]
            current_candle = intraday_df.iloc[-1]
            
            # Run strategy evaluation
            evaluation = self.strategy.evaluate_stock(
                ticker=ticker,
                current_candle=current_candle,
                intraday_df=intraday_df,
                prev_day_high=prev_day_data[ticker]['high'],
                prev_day_low=prev_day_data[ticker]['low'],
                avg_volume=candidate.avg_volume
            )
            
            results.append({
                'Ticker': ticker,
                'Price': evaluation.current_price,
                'Day %': candidate.day_change_pct,
                'Direction': evaluation.direction.value,
                'Dir Prob': evaluation.direction_probability,
                'Goal Prob': evaluation.goal_probability,
                'Mom Prob': evaluation.momentum_probability,
                'Overall': evaluation.overall_score,
                'Stop Loss': evaluation.stop_loss,
                'Take Profit': evaluation.take_profit,
                'Profit %': evaluation.goal_profit_pct,
                'Enter': evaluation.should_enter,
                'Stay': evaluation.should_stay
            })
        
        df = pd.DataFrame(results)
        
        # Sort by overall score
        if not df.empty:
            df = df.sort_values('Overall', ascending=False)
        
        return df


# Example usage
if __name__ == "__main__":
    # Example: Test scanner with sample data
    sample_stocks = [
        {
            'ticker': 'STOCK1',
            'current_price': 15.50,
            'day_open': 15.00,
            'current_volume': 5_000_000,
            'avg_volume': 800_000,
            'shares_outstanding': 8_000_000,
            'candle_open': 15.45,
            'candle_close': 15.65
        },
        {
            'ticker': 'STOCK2',
            'current_price': 25.00,
            'day_open': 24.00,
            'current_volume': 2_000_000,
            'avg_volume': 1_000_000,
            'shares_outstanding': 5_000_000,
            'candle_open': 24.95,
            'candle_close': 25.30
        },
        {
            'ticker': 'STOCK3',  # This one won't pass (price too high)
            'current_price': 35.00,
            'day_open': 33.00,
            'current_volume': 10_000_000,
            'avg_volume': 1_500_000,
            'shares_outstanding': 3_000_000,
            'candle_open': 34.80,
            'candle_close': 35.50
        }
    ]
    
    # Initialize scanner
    scanner = StockScanner()
    
    # Scan stocks
    candidates = scanner.scan_multiple(sample_stocks)
    
    # Generate report
    report = scanner.generate_report(candidates)
    print("\n=== Scanner Report ===")
    print(report.to_string(index=False))
    
    # Show only passing stocks
    passing = scanner.get_passing_stocks(sample_stocks)
    print(f"\n{len(passing)} stocks passed all criteria:")
    for stock in passing:
        print(f"  {stock.ticker}: ${stock.current_price:.2f} "
              f"(+{stock.day_change_pct:.1f}% day, +{stock.one_min_change_pct:.1f}% 1min)")
