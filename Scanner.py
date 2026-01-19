"""
Stock Scanner and Trading Strategy System
Implements 3-stage evaluation: Direction → Goal → Momentum
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Direction(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class SupportResistanceLevel:
    """Represents a support or resistance level"""
    price: float
    level_type: str  # 'support' or 'resistance'
    strength: int  # number of touches/clusters
    timestamps: List[datetime]


@dataclass
class TargetPrices:
    """Container for identified target prices"""
    previous_day_high: float
    previous_day_low: float
    intermediate_resistance: List[float]
    intermediate_support: List[float]
    all_resistance: List[float]  # Combined resistance levels
    all_support: List[float]  # Combined support levels


@dataclass
class EvaluationResult:
    """Results from the 3-stage evaluation"""
    ticker: str
    timestamp: datetime
    current_price: float
    
    # Stage 1: Direction
    direction: Direction
    direction_magnitude: float  # % change from candle open
    volatility: float  # % range of candle
    direction_probability: float  # 0-100
    
    # Stage 2: Goal
    stop_loss: float
    take_profit: float
    goal_profit_pct: float
    goal_probability: float  # 0-100
    
    # Stage 3: Momentum
    volume_ratio: float  # current vs average
    delta_aligned: bool
    momentum_probability: float  # 0-100
    
    # Overall
    should_enter: bool
    should_stay: bool
    overall_score: float


class SupportResistanceDetector:
    """Detects support and resistance levels with clustering"""
    
    def __init__(self, cluster_threshold: float = 0.01):
        """
        Args:
            cluster_threshold: Price proximity threshold (1% default)
        """
        self.cluster_threshold = cluster_threshold
    
    def find_local_extrema(self, df: pd.DataFrame, window: int = 5) -> Tuple[List, List]:
        """
        Find local minima (support) and maxima (resistance)
        
        Args:
            df: DataFrame with OHLCV data
            window: Window size for local extrema detection
            
        Returns:
            Tuple of (support_levels, resistance_levels)
        """
        highs = df['high'].values
        lows = df['low'].values
        timestamps = df.index
        
        # Find local maxima (resistance)
        resistance = []
        for i in range(window, len(highs) - window):
            if highs[i] == max(highs[i-window:i+window+1]):
                resistance.append({
                    'price': highs[i],
                    'timestamp': timestamps[i]
                })
        
        # Find local minima (support)
        support = []
        for i in range(window, len(lows) - window):
            if lows[i] == min(lows[i-window:i+window+1]):
                support.append({
                    'price': lows[i],
                    'timestamp': timestamps[i]
                })
        
        return support, resistance
    
    def cluster_levels(self, levels: List[Dict]) -> List[SupportResistanceLevel]:
        """
        Cluster price levels within threshold
        
        Args:
            levels: List of price level dictionaries
            
        Returns:
            List of clustered SupportResistanceLevel objects
        """
        if not levels:
            return []
        
        # Sort by price
        sorted_levels = sorted(levels, key=lambda x: x['price'])
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            # Check if within threshold of current cluster average
            cluster_avg = np.mean([l['price'] for l in current_cluster])
            if abs(level['price'] - cluster_avg) / cluster_avg <= self.cluster_threshold:
                current_cluster.append(level)
            else:
                # Save current cluster and start new one
                if len(current_cluster) >= 2:  # Only keep clusters with 2+ touches
                    clusters.append(current_cluster)
                current_cluster = [level]
        
        # Don't forget the last cluster
        if len(current_cluster) >= 2:
            clusters.append(current_cluster)
        
        # Convert to SupportResistanceLevel objects
        sr_levels = []
        for cluster in clusters:
            avg_price = np.mean([l['price'] for l in cluster])
            timestamps = [l['timestamp'] for l in cluster]
            sr_levels.append(SupportResistanceLevel(
                price=avg_price,
                level_type='unknown',  # Set by caller
                strength=len(cluster),
                timestamps=timestamps
            ))
        
        return sr_levels
    
    def identify_targets(self, df: pd.DataFrame, prev_day_high: float, 
                        prev_day_low: float) -> TargetPrices:
        """
        Identify all target prices (PDH, PDL, intermediate S/R)
        
        Args:
            df: Intraday OHLCV data
            prev_day_high: Previous day's high
            prev_day_low: Previous day's low
            
        Returns:
            TargetPrices object with all levels
        """
        support_levels, resistance_levels = self.find_local_extrema(df)
        
        # Cluster intermediate levels
        clustered_support = self.cluster_levels(support_levels)
        for level in clustered_support:
            level.level_type = 'support'
        
        clustered_resistance = self.cluster_levels(resistance_levels)
        for level in clustered_resistance:
            level.level_type = 'resistance'
        
        # Extract prices
        intermediate_support = [l.price for l in clustered_support]
        intermediate_resistance = [l.price for l in clustered_resistance]
        
        # Combine all levels
        all_support = sorted(intermediate_support + [prev_day_low])
        all_resistance = sorted(intermediate_resistance + [prev_day_high])
        
        return TargetPrices(
            previous_day_high=prev_day_high,
            previous_day_low=prev_day_low,
            intermediate_resistance=intermediate_resistance,
            intermediate_support=intermediate_support,
            all_resistance=all_resistance,
            all_support=all_support
        )


class DirectionEvaluator:
    """Stage 1: Evaluate direction and magnitude"""
    
    def evaluate(self, candle: pd.Series) -> Tuple[Direction, float, float, float]:
        """
        Evaluate direction from 1-minute candle
        
        Args:
            candle: Series with open, high, low, close
            
        Returns:
            Tuple of (direction, magnitude_pct, volatility_pct, probability)
        """
        open_price = candle['open']
        close_price = candle['close']
        high_price = candle['high']
        low_price = candle['low']
        
        # Calculate magnitude (% change from open)
        magnitude_pct = ((close_price - open_price) / open_price) * 100
        
        # Calculate volatility (% range)
        volatility_pct = ((high_price - low_price) / open_price) * 100
        
        # Determine direction
        if magnitude_pct >= 1.0:
            direction = Direction.POSITIVE
            probability = min(100, 50 + (magnitude_pct * 5))  # Scale probability
        elif magnitude_pct <= -1.0:
            direction = Direction.NEGATIVE
            probability = min(100, 50 + (abs(magnitude_pct) * 5))
        else:
            direction = Direction.NEUTRAL
            probability = abs(magnitude_pct) * 50  # Low probability for neutral
        
        return direction, magnitude_pct, volatility_pct, probability


class GoalEvaluator:
    """Stage 2: Evaluate entry goals and profit potential"""
    
    def __init__(self, target_profit_pct: float = 3.0):
        self.target_profit_pct = target_profit_pct
    
    def find_nearest_levels(self, current_price: float, targets: TargetPrices) -> Tuple[float, float]:
        """
        Find nearest support (for stop) and resistance (for target)
        
        Args:
            current_price: Current stock price
            targets: TargetPrices object
            
        Returns:
            Tuple of (stop_loss, take_profit)
        """
        # Find nearest support below current price
        support_below = [s for s in targets.all_support if s < current_price]
        stop_loss = max(support_below) + 0.01 if support_below else current_price * 0.97
        
        # Find nearest resistance above current price
        resistance_above = [r for r in targets.all_resistance if r > current_price]
        take_profit = min(resistance_above) - 0.01 if resistance_above else current_price * 1.05
        
        return stop_loss, take_profit
    
    def evaluate(self, current_price: float, targets: TargetPrices) -> Tuple[float, float, float, float]:
        """
        Evaluate goal probability
        
        Args:
            current_price: Current stock price
            targets: TargetPrices object
            
        Returns:
            Tuple of (stop_loss, take_profit, profit_pct, probability)
        """
        stop_loss, take_profit = self.find_nearest_levels(current_price, targets)
        
        # Calculate potential profit percentage
        profit_pct = ((take_profit - current_price) / current_price) * 100
        
        # Calculate probability based on profit potential
        if profit_pct >= self.target_profit_pct:
            # Can meet goal
            probability = min(100, 50 + (profit_pct * 10))
        elif profit_pct > 0:
            # Below goal but possible
            probability = (profit_pct / self.target_profit_pct) * 50
        else:
            # No profit potential
            probability = 0
        
        # Bonus: if above all intermediate support and PDH
        if (current_price > targets.previous_day_high and 
            all(current_price > s for s in targets.intermediate_support)):
            probability = 100
        
        return stop_loss, take_profit, profit_pct, probability


class MomentumEvaluator:
    """Stage 3: Evaluate momentum for staying in trade"""
    
    def evaluate(self, current_volume: float, avg_volume: float, 
                 delta: float, direction: Direction) -> Tuple[float, bool, float]:
        """
        Evaluate momentum probability
        
        Args:
            current_volume: Current period volume
            avg_volume: Average volume
            delta: Price change (positive/negative)
            direction: Original entry direction
            
        Returns:
            Tuple of (volume_ratio, delta_aligned, probability)
        """
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        
        # Check if delta aligns with direction
        delta_aligned = (
            (direction == Direction.POSITIVE and delta > 0) or
            (direction == Direction.NEGATIVE and delta < 0)
        )
        
        # Calculate probability
        if current_volume == 0:
            probability = 25
        elif volume_ratio >= 1.0 and delta_aligned:
            probability = 100
        elif volume_ratio >= 1.0 and not delta_aligned:
            probability = 0  # Volume up but wrong direction
        elif volume_ratio < 1.0 and volume_ratio >= 0.5:
            probability = 50  # Decreasing but not critical
        else:
            probability = 25  # Low volume
        
        return volume_ratio, delta_aligned, probability


class TradingStrategy:
    """Main strategy orchestrator"""
    
    def __init__(self, target_profit_pct: float = 3.0, 
                 min_direction_prob: float = 50,
                 min_goal_prob: float = 50,
                 min_momentum_prob: float = 50):
        self.sr_detector = SupportResistanceDetector()
        self.direction_eval = DirectionEvaluator()
        self.goal_eval = GoalEvaluator(target_profit_pct)
        self.momentum_eval = MomentumEvaluator()
        
        self.min_direction_prob = min_direction_prob
        self.min_goal_prob = min_goal_prob
        self.min_momentum_prob = min_momentum_prob
    
    def evaluate_stock(self, ticker: str, current_candle: pd.Series,
                       intraday_df: pd.DataFrame, prev_day_high: float,
                       prev_day_low: float, avg_volume: float) -> EvaluationResult:
        """
        Complete 3-stage evaluation
        
        Args:
            ticker: Stock symbol
            current_candle: Current 1-minute candle
            intraday_df: Intraday price history
            prev_day_high: Previous day high
            prev_day_low: Previous day low
            avg_volume: Average volume for momentum calc
            
        Returns:
            EvaluationResult with all probabilities
        """
        current_price = current_candle['close']
        
        # Stage 1: Direction
        direction, magnitude, volatility, dir_prob = self.direction_eval.evaluate(current_candle)
        
        # Stage 2: Goal
        targets = self.sr_detector.identify_targets(intraday_df, prev_day_high, prev_day_low)
        stop_loss, take_profit, profit_pct, goal_prob = self.goal_eval.evaluate(current_price, targets)
        
        # Stage 3: Momentum
        current_volume = current_candle.get('volume', avg_volume)
        delta = current_candle['close'] - current_candle['open']
        vol_ratio, delta_aligned, momentum_prob = self.momentum_eval.evaluate(
            current_volume, avg_volume, delta, direction
        )
        
        # Decision logic
        should_enter = (
            direction == Direction.POSITIVE and
            dir_prob >= self.min_direction_prob and
            goal_prob >= self.min_goal_prob and
            momentum_prob >= self.min_momentum_prob
        )
        
        should_stay = momentum_prob >= self.min_momentum_prob
        
        # Overall score (weighted average)
        overall_score = (dir_prob * 0.3 + goal_prob * 0.4 + momentum_prob * 0.3)
        
        return EvaluationResult(
            ticker=ticker,
            timestamp=datetime.now(),
            current_price=current_price,
            direction=direction,
            direction_magnitude=magnitude,
            volatility=volatility,
            direction_probability=dir_prob,
            stop_loss=stop_loss,
            take_profit=take_profit,
            goal_profit_pct=profit_pct,
            goal_probability=goal_prob,
            volume_ratio=vol_ratio,
            delta_aligned=delta_aligned,
            momentum_probability=momentum_prob,
            should_enter=should_enter,
            should_stay=should_stay,
            overall_score=overall_score
        )


# Example usage
if __name__ == "__main__":
    # Example: Create sample data
    dates = pd.date_range(start='2024-01-19 09:30', periods=100, freq='1min')
    sample_data = pd.DataFrame({
        'open': np.random.uniform(25, 26, 100),
        'high': np.random.uniform(25.5, 26.5, 100),
        'low': np.random.uniform(24.5, 25.5, 100),
        'close': np.random.uniform(25, 26, 100),
        'volume': np.random.uniform(100000, 500000, 100)
    }, index=dates)
    
    # Initialize strategy
    strategy = TradingStrategy(target_profit_pct=3.0)
    
    # Evaluate current candle
    current_candle = sample_data.iloc[-1]
    result = strategy.evaluate_stock(
        ticker='EXAMPLE',
        current_candle=current_candle,
        intraday_df=sample_data,
        prev_day_high=26.50,
        prev_day_low=24.00,
        avg_volume=250000
    )
    
    print(f"\n=== Evaluation for {result.ticker} ===")
    print(f"Current Price: ${result.current_price:.2f}")
    print(f"\nStage 1 - Direction: {result.direction.value}")
    print(f"  Magnitude: {result.direction_magnitude:.2f}%")
    print(f"  Probability: {result.direction_probability:.1f}%")
    print(f"\nStage 2 - Goal:")
    print(f"  Stop Loss: ${result.stop_loss:.2f}")
    print(f"  Take Profit: ${result.take_profit:.2f}")
    print(f"  Profit Potential: {result.goal_profit_pct:.2f}%")
    print(f"  Probability: {result.goal_probability:.1f}%")
    print(f"\nStage 3 - Momentum:")
    print(f"  Volume Ratio: {result.volume_ratio:.2f}x")
    print(f"  Delta Aligned: {result.delta_aligned}")
    print(f"  Probability: {result.momentum_probability:.1f}%")
    print(f"\n{'='*40}")
    print(f"Overall Score: {result.overall_score:.1f}%")
    print(f"Should Enter: {result.should_enter}")
    print(f"Should Stay: {result.should_stay}")
