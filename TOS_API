"""
TD Ameritrade API Integration
Handles authentication, data retrieval, and order placement
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
from urllib.parse import urlencode
import time


class TDAmeritrade:
    """TD Ameritrade API client"""
    
    BASE_URL = "https://api.tdameritrade.com/v1"
    
    def __init__(self, api_key: str, redirect_uri: str = "https://localhost",
                 refresh_token: Optional[str] = None):
        """
        Initialize TD Ameritrade client
        
        Args:
            api_key: Your TD Ameritrade API key (Consumer Key)
            redirect_uri: OAuth redirect URI
            refresh_token: Refresh token for authentication
        """
        self.api_key = api_key
        self.redirect_uri = redirect_uri
        self.refresh_token = refresh_token
        self.access_token = None
        self.token_expires_at = None
    
    def get_authorization_url(self) -> str:
        """
        Generate OAuth authorization URL
        
        Returns:
            Authorization URL for user to visit
        """
        params = {
            'response_type': 'code',
            'redirect_uri': self.redirect_uri,
            'client_id': f"{self.api_key}@AMER.OAUTHAP"
        }
        return f"https://auth.tdameritrade.com/auth?{urlencode(params)}"
    
    def get_access_token(self, authorization_code: Optional[str] = None) -> bool:
        """
        Get access token using authorization code or refresh token
        
        Args:
            authorization_code: Code from OAuth callback (for initial auth)
            
        Returns:
            True if successful
        """
        url = f"{self.BASE_URL}/oauth2/token"
        
        if authorization_code:
            # Initial authentication
            data = {
                'grant_type': 'authorization_code',
                'access_type': 'offline',
                'code': authorization_code,
                'client_id': self.api_key,
                'redirect_uri': self.redirect_uri
            }
        elif self.refresh_token:
            # Refresh existing token
            data = {
                'grant_type': 'refresh_token',
                'refresh_token': self.refresh_token,
                'client_id': self.api_key
            }
        else:
            raise ValueError("Need authorization_code or refresh_token")
        
        response = requests.post(url, data=data)
        
        if response.status_code == 200:
            token_data = response.json()
            self.access_token = token_data['access_token']
            
            if 'refresh_token' in token_data:
                self.refresh_token = token_data['refresh_token']
            
            # Token expires in seconds
            expires_in = token_data.get('expires_in', 1800)
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
            
            return True
        else:
            print(f"Authentication failed: {response.text}")
            return False
    
    def ensure_authenticated(self):
        """Check and refresh token if needed"""
        if not self.access_token or datetime.now() >= self.token_expires_at:
            if not self.get_access_token():
                raise Exception("Failed to authenticate")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication"""
        self.ensure_authenticated()
        return {
            'Authorization': f'Bearer {self.access_token}'
        }
    
    def get_quote(self, symbol: str) -> Dict:
        """
        Get real-time quote for a symbol
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Quote data dictionary
        """
        url = f"{self.BASE_URL}/marketdata/{symbol}/quotes"
        response = requests.get(url, headers=self._get_headers())
        
        if response.status_code == 200:
            return response.json()[symbol]
        else:
            raise Exception(f"Failed to get quote: {response.text}")
    
    def get_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get quotes for multiple symbols
        
        Args:
            symbols: List of stock tickers
            
        Returns:
            Dictionary mapping symbol to quote data
        """
        url = f"{self.BASE_URL}/marketdata/quotes"
        params = {'symbol': ','.join(symbols)}
        response = requests.get(url, headers=self._get_headers(), params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get quotes: {response.text}")
    
    def get_price_history(self, symbol: str, period_type: str = 'day',
                         period: int = 1, frequency_type: str = 'minute',
                         frequency: int = 1) -> pd.DataFrame:
        """
        Get historical price data
        
        Args:
            symbol: Stock ticker
            period_type: 'day', 'month', 'year', 'ytd'
            period: Number of periods
            frequency_type: 'minute', 'daily', 'weekly', 'monthly'
            frequency: Frequency interval
            
        Returns:
            DataFrame with OHLCV data
        """
        url = f"{self.BASE_URL}/marketdata/{symbol}/pricehistory"
        params = {
            'periodType': period_type,
            'period': period,
            'frequencyType': frequency_type,
            'frequency': frequency,
            'needExtendedHoursData': False
        }
        
        response = requests.get(url, headers=self._get_headers(), params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'candles' not in data or not data['candles']:
                return pd.DataFrame()
            
            df = pd.DataFrame(data['candles'])
            df['datetime'] = pd.to_datetime(df['datetime'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            return df[['open', 'high', 'low', 'close', 'volume']]
        else:
            raise Exception(f"Failed to get price history: {response.text}")
    
    def get_fundamentals(self, symbol: str) -> Dict:
        """
        Get fundamental data for a symbol
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Fundamental data dictionary
        """
        url = f"{self.BASE_URL}/instruments"
        params = {
            'symbol': symbol,
            'projection': 'fundamental'
        }
        
        response = requests.get(url, headers=self._get_headers(), params=params)
        
        if response.status_code == 200:
            data = response.json()
            return data.get(symbol, {}).get('fundamental', {})
        else:
            raise Exception(f"Failed to get fundamentals: {response.text}")
    
    def place_order(self, account_id: str, order: Dict) -> bool:
        """
        Place an order
        
        Args:
            account_id: Your TD Ameritrade account ID
            order: Order specification dictionary
            
        Returns:
            True if successful
        """
        url = f"{self.BASE_URL}/accounts/{account_id}/orders"
        headers = self._get_headers()
        headers['Content-Type'] = 'application/json'
        
        response = requests.post(url, headers=headers, json=order)
        
        if response.status_code == 201:
            return True
        else:
            print(f"Order failed: {response.text}")
            return False
    
    def create_market_order(self, symbol: str, quantity: int, 
                           instruction: str = 'BUY') -> Dict:
        """
        Create a market order specification
        
        Args:
            symbol: Stock ticker
            quantity: Number of shares
            instruction: 'BUY' or 'SELL'
            
        Returns:
            Order specification dictionary
        """
        return {
            "orderType": "MARKET",
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": instruction,
                    "quantity": quantity,
                    "instrument": {
                        "symbol": symbol,
                        "assetType": "EQUITY"
                    }
                }
            ]
        }
    
    def create_bracket_order(self, symbol: str, quantity: int,
                            entry_price: float, stop_loss: float, 
                            take_profit: float) -> Dict:
        """
        Create a bracket order (entry + stop loss + take profit)
        
        Args:
            symbol: Stock ticker
            quantity: Number of shares
            entry_price: Entry limit price
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Order specification dictionary
        """
        return {
            "orderType": "LIMIT",
            "session": "NORMAL",
            "price": entry_price,
            "duration": "DAY",
            "orderStrategyType": "TRIGGER",
            "orderLegCollection": [
                {
                    "instruction": "BUY",
                    "quantity": quantity,
                    "instrument": {
                        "symbol": symbol,
                        "assetType": "EQUITY"
                    }
                }
            ],
            "childOrderStrategies": [
                {
                    "orderType": "STOP",
                    "session": "NORMAL",
                    "duration": "GTC",
                    "stopPrice": stop_loss,
                    "orderStrategyType": "SINGLE",
                    "orderLegCollection": [
                        {
                            "instruction": "SELL",
                            "quantity": quantity,
                            "instrument": {
                                "symbol": symbol,
                                "assetType": "EQUITY"
                            }
                        }
                    ]
                },
                {
                    "orderType": "LIMIT",
                    "session": "NORMAL",
                    "duration": "GTC",
                    "price": take_profit,
                    "orderStrategyType": "SINGLE",
                    "orderLegCollection": [
                        {
                            "instruction": "SELL",
                            "quantity": quantity,
                            "instrument": {
                                "symbol": symbol,
                                "assetType": "EQUITY"
                            }
                        }
                    ]
                }
            ]
        }


class DataFetcher:
    """Fetches and prepares data for scanner and strategy"""
    
    def __init__(self, td_client: TDAmeritrade):
        """
        Initialize data fetcher
        
        Args:
            td_client: Authenticated TDAmeritrade client
        """
        self.td = td_client
    
    def prepare_scanner_data(self, symbols: List[str]) -> List[Dict]:
        """
        Prepare data for scanner from TD Ameritrade
        
        Args:
            symbols: List of stock symbols to scan
            
        Returns:
            List of dictionaries with scanner data
        """
        quotes = self.td.get_quotes(symbols)
        scanner_data = []
        
        for symbol, quote in quotes.items():
            # Get fundamental data for shares outstanding
            try:
                fundamentals = self.td.get_fundamentals(symbol)
                shares = fundamentals.get('sharesOutstanding', 0)
            except:
                shares = 0
            
            # Get 1-minute data for current candle
            try:
                candles = self.td.get_price_history(
                    symbol, 
                    period_type='day',
                    period=1,
                    frequency_type='minute',
                    frequency=1
                )
                
                if not candles.empty:
                    current_candle = candles.iloc[-1]
                    candle_open = current_candle['open']
                    candle_close = current_candle['close']
                else:
                    candle_open = quote['openPrice']
                    candle_close = quote['lastPrice']
            except:
                candle_open = quote['openPrice']
                candle_close = quote['lastPrice']
            
            scanner_data.append({
                'ticker': symbol,
                'current_price': quote['lastPrice'],
                'day_open': quote['openPrice'],
                'current_volume': quote['totalVolume'],
                'avg_volume': quote.get('avgVolume', 0),
                'shares_outstanding': shares,
                'candle_open': candle_open,
                'candle_close': candle_close
            })
        
        return scanner_data
    
    def prepare_strategy_data(self, symbols: List[str]) -> tuple:
        """
        Prepare historical data and previous day data for strategy
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Tuple of (historical_data dict, prev_day_data dict)
        """
        historical_data = {}
        prev_day_data = {}
        
        for symbol in symbols:
            # Get today's 1-minute data
            intraday = self.td.get_price_history(
                symbol,
                period_type='day',
                period=1,
                frequency_type='minute',
                frequency=1
            )
            historical_data[symbol] = intraday
            
            # Get previous day data
            daily = self.td.get_price_history(
                symbol,
                period_type='day',
                period=5,
                frequency_type='daily',
                frequency=1
            )
            
            if len(daily) >= 2:
                prev_day = daily.iloc[-2]
                prev_day_data[symbol] = {
                    'high': prev_day['high'],
                    'low': prev_day['low']
                }
            else:
                # Fallback to current day
                prev_day_data[symbol] = {
                    'high': intraday['high'].max(),
                    'low': intraday['low'].min()
                }
        
        return historical_data, prev_day_data


# Example usage
if __name__ == "__main__":
    # Configuration
    API_KEY = "YOUR_API_KEY_HERE"
    
    # Initialize client
    td = TDAmeritrade(api_key=API_KEY)
    
    # For first-time setup, get authorization URL
    print("Authorization URL:")
    print(td.get_authorization_url())
    print("\nVisit this URL, authorize, and copy the code from redirect URL")
    
    # After getting authorization code:
    # auth_code = "YOUR_AUTH_CODE"
    # td.get_access_token(authorization_code=auth_code)
    # print(f"Refresh token: {td.refresh_token}")
    # # Save this refresh token for future use
    
    # For subsequent use with saved refresh token:
    # td.refresh_token = "YOUR_SAVED_REFRESH_TOKEN"
    # td.get_access_token()
    
    # Example: Get quote
    # quote = td.get_quote("AAPL")
    # print(f"AAPL: ${quote['lastPrice']}")
    
    # Example: Get price history
    # df = td.get_price_history("AAPL", period_type='day', period=1,
    #                          frequency_type='minute', frequency=1)
    # print(df.tail())
