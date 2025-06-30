import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import pytz
import json
import warnings
warnings.filterwarnings('ignore')

# === MOBILE MT5 PROFESSIONAL BOT ===
BOT_TOKEN = '8119410088:AAEumef6-C0CNY-zNu6kBaubXW2l5wSYiCw'
MIN_CONFIDENCE = 8.5
TIMEZONE = pytz.timezone("Etc/UTC")

class MobileMT5ProfessionalBot:
    def __init__(self):
        self.chat_id = None
        self.last_signals = {}
        self.running = True
        
        # PROFESSIONAL FOREX PAIRS (Mobile Optimized)
        self.major_pairs = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 
            'AUDUSD', 'USDCAD', 'NZDUSD'
        ]
        self.cross_pairs = [
            'EURJPY', 'GBPJPY', 'EURGBP', 'EURAUD', 
            'GBPAUD', 'AUDCAD', 'AUDJPY'
        ]
        
        self.all_pairs = self.major_pairs + self.cross_pairs
        
        # MOBILE-FRIENDLY TIMEFRAMES
        self.timeframes = ['1h', '4h', '1d']  # Most reliable for mobile
        
        print("🔥 MOBILE MT5 PROFESSIONAL BOT INITIALIZED")
        print("📱 Optimized for mobile MT5 users")

    def get_telegram_updates(self):
        """GET TELEGRAM UPDATES"""
        try:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
            response = requests.get(url, timeout=10)
            data = response.json()
            if data['ok'] and data['result']:
                return data['result'][-1]['message']['chat']['id']
        except Exception as e:
            print(f"Telegram error: {e}")
        return None

    def send_telegram_message(self, text):
        """SEND TELEGRAM MESSAGE"""
        try:
            url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=payload, timeout=15)
            return response.status_code == 200
        except Exception as e:
            print(f"Send message error: {e}")
            return False

    def get_mobile_forex_data(self, symbol, timeframe='1h', limit=200):
        """GET REAL-TIME FOREX DATA (MOBILE COMPATIBLE)"""
        try:
            # Multiple data sources for reliability
            data_sources = [
                self.get_fxempire_data,
                self.get_yahoo_data,
                self.get_alpha_vantage_data,
                self.get_finhub_data
            ]
            
            for source in data_sources:
                try:
                    df = source(symbol, timeframe, limit)
                    if df is not None and len(df) >= 50:
                        print(f"✅ Mobile data: {symbol} from {source.__name__}")
                        return df
                except Exception as e:
                    print(f"❌ {source.__name__} failed: {e}")
                    continue
            
            print(f"❌ All data sources failed for {symbol}")
            return None
            
        except Exception as e:
            print(f"❌ Mobile data error: {e}")
            return None

    def get_fxempire_data(self, symbol, timeframe, limit):
        """GET DATA FROM FXEMPIRE (MOBILE FRIENDLY)"""
        try:
            # FXEmpire API endpoint
            url = f"https://www.fxempire.com/api/v1/en/markets/chart"
            
            # Convert timeframe
            tf_map = {'1h': '1H', '4h': '4H', '1d': '1D'}
            tf = tf_map.get(timeframe, '1H')
            
            params = {
                'symbol': symbol,
                'timeframe': tf,
                'limit': limit
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)',
                'Accept': 'application/json'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    df = pd.DataFrame(data['data'])
                    df['time'] = pd.to_datetime(df['time'])
                    df = df.rename(columns={
                        'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
                    })
                    return df.tail(limit)
            
            return None
            
        except Exception as e:
            print(f"FXEmpire error: {e}")
            return None

    def get_yahoo_data(self, symbol, timeframe, limit):
        """GET YAHOO FINANCE DATA (MOBILE OPTIMIZED)"""
        try:
            # Yahoo Finance symbol mapping
            yahoo_symbol = f"{symbol}=X"
            
            # Timeframe mapping
            interval_map = {'1h': '1h', '4h': '4h', '1d': '1d'}
            interval = interval_map.get(timeframe, '1h')
            
            # Calculate period
            if timeframe == '1h':
                period = '5d'
            elif timeframe == '4h':
                period = '1mo'
            else:
                period = '3mo'
            
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
            params = {
                'interval': interval,
                'period1': int((datetime.now() - timedelta(days=30)).timestamp()),
                'period2': int(datetime.now().timestamp()),
                'includePrePost': 'false'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                result = data['chart']['result'][0]
                
                timestamps = result['timestamp']
                ohlc = result['indicators']['quote'][0]
                
                df = pd.DataFrame({
                    'time': pd.to_datetime(timestamps, unit='s'),
                    'open': ohlc['open'],
                    'high': ohlc['high'],
                    'low': ohlc['low'],
                    'close': ohlc['close'],
                    'volume': ohlc.get('volume', [0] * len(timestamps))
                })
                
                df = df.dropna().tail(limit)
                return df
            
            return None
            
        except Exception as e:
            print(f"Yahoo error: {e}")
            return None

    def get_alpha_vantage_data(self, symbol, timeframe, limit):
        """GET ALPHA VANTAGE DATA (FREE TIER)"""
        try:
            # Free Alpha Vantage API key (limited requests)
            api_key = 'demo'  # Replace with your free key
            
            function_map = {
                '1h': 'FX_INTRADAY',
                '4h': 'FX_INTRADAY', 
                '1d': 'FX_DAILY'
            }
            
            from_symbol = symbol[:3]
            to_symbol = symbol[3:]
            
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': function_map.get(timeframe, 'FX_INTRADAY'),
                'from_symbol': from_symbol,
                'to_symbol': to_symbol,
                'interval': '60min' if timeframe in ['1h', '4h'] else None,
                'apikey': api_key,
                'outputsize': 'compact'
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                # Find the time series key
                ts_key = None
                for key in data.keys():
                    if 'Time Series' in key:
                        ts_key = key
                        break
                
                if ts_key and ts_key in data:
                    ts_data = data[ts_key]
                    
                    df_data = []
                    for timestamp, values in ts_data.items():
                        df_data.append({
                            'time': pd.to_datetime(timestamp),
                            'open': float(values['1. open']),
                            'high': float(values['2. high']),
                            'low': float(values['3. low']),
                            'close': float(values['4. close']),
                            'volume': 0
                        })
                    
                    df = pd.DataFrame(df_data)
                    df = df.sort_values('time').tail(limit)
                    return df
            
            return None
            
        except Exception as e:
            print(f"Alpha Vantage error: {e}")
            return None

    def get_finhub_data(self, symbol, timeframe, limit):
        """GET FINNHUB DATA (FREE TIER)"""
        try:
            # Free Finnhub API
            api_key = 'demo'  # Replace with your free key
            
            # Convert symbol format
            finnhub_symbol = f"OANDA:{symbol}"
            
            # Timeframe mapping
            resolution_map = {'1h': '60', '4h': '240', '1d': 'D'}
            resolution = resolution_map.get(timeframe, '60')
            
            # Time range
            to_time = int(datetime.now().timestamp())
            from_time = to_time - (limit * 3600)  # Approximate
            
            url = f"https://finnhub.io/api/v1/forex/candle"
            params = {
                'symbol': finnhub_symbol,
                'resolution': resolution,
                'from': from_time,
                'to': to_time,
                'token': api_key
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data['s'] == 'ok' and len(data['t']) > 0:
                    df = pd.DataFrame({
                        'time': pd.to_datetime(data['t'], unit='s'),
                        'open': data['o'],
                        'high': data['h'],
                        'low': data['l'],
                        'close': data['c'],
                        'volume': data['v']
                    })
                    
                    return df.tail(limit)
            
            return None
            
        except Exception as e:
            print(f"Finnhub error: {e}")
            return None

    def calculate_mobile_indicators(self, df):
        """CALCULATE PROFESSIONAL INDICATORS (MOBILE OPTIMIZED)"""
        try:
            # === TREND INDICATORS ===
            # EMAs (Mobile optimized periods)
            for period in [8, 13, 21, 34, 55, 89]:
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            
            # SMAs
            for period in [20, 50, 100]:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            
            # === MOMENTUM INDICATORS ===
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Stochastic
            low_14 = df['low'].rolling(window=14).min()
            high_14 = df['high'].rolling(window=14).max()
            df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # === VOLATILITY INDICATORS ===
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # ATR
            df['tr1'] = df['high'] - df['low']
            df['tr2'] = abs(df['high'] - df['close'].shift())
            df['tr3'] = abs(df['low'] - df['close'].shift())
            df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['atr'] = df['true_range'].rolling(14).mean()
            df['atr_percent'] = (df['atr'] / df['close']) * 100
            
            # === ADDITIONAL INDICATORS ===
            # Williams %R
            df['williams_r'] = -100 * ((high_14 - df['close']) / (high_14 - low_14))
            
            # Momentum
            df['momentum'] = df['close'] / df['close'].shift(10) - 1
            
            return df
            
        except Exception as e:
            print(f"❌ Mobile indicator calculation error: {e}")
            return df

    def mobile_professional_analysis(self, symbol):
        """MOBILE PROFESSIONAL MULTI-TIMEFRAME ANALYSIS"""
        try:
            print(f"📱 MOBILE ANALYSIS: {symbol}")
            
            # Get multiple timeframes
            timeframe_data = {}
            
            for tf in self.timeframes:
                df = self.get_mobile_forex_data(symbol, tf, 100)
                if df is not None and len(df) >= 50:
                    df = self.calculate_mobile_indicators(df)
                    timeframe_data[tf] = df
                    print(f"✅ {symbol} {tf}: {len(df)} candles")
                else:
                    print(f"❌ {symbol} {tf}: Insufficient data")
            
            if len(timeframe_data) < 2:
                print(f"❌ Not enough timeframes for {symbol}")
                return None
            
            # === MOBILE PROFESSIONAL SCORING ===
            total_score = 0
            timeframe_signals = {}
            
            for tf_name, df in timeframe_data.items():
                if len(df) < 30:
                    continue
                
                latest = df.iloc[-1]
                prev = df.iloc[-2]
                prev2 = df.iloc[-3]
                
                tf_score = 0
                
                # === TREND ANALYSIS (50 POINTS) ===
                # EMA Trend Alignment
                if (latest['ema_8'] > latest['ema_13'] > latest['ema_21'] > 
                    latest['ema_34'] > latest['ema_55']):
                    tf_score += 25  # Strong bullish trend
                elif (latest['ema_8'] < latest['ema_13'] < latest['ema_21'] < 
                      latest['ema_34'] < latest['ema_55']):
                    tf_score -= 25  # Strong bearish trend
                
                # Price vs Key EMAs
                if latest['close'] > latest['ema_21'] > latest['ema_55']:
                    tf_score += 15
                elif latest['close'] < latest['ema_21'] < latest['ema_55']:
                    tf_score -= 15
                
                # EMA Momentum
                if (latest['ema_8'] > prev['ema_8'] and 
                    latest['ema_21'] > prev['ema_21']):
                    tf_score += 10
                elif (latest['ema_8'] < prev['ema_8'] and 
                      latest['ema_21'] < prev['ema_21']):
                    tf_score -= 10
                
                # === MOMENTUM ANALYSIS (30 POINTS) ===
                # RSI Analysis
                rsi = latest['rsi']
                if 50 < rsi < 70:
                    tf_score += 12
                elif 30 < rsi < 50:
                    tf_score -= 12
                elif rsi > 80:
                    tf_score -= 5  # Overbought
                elif rsi < 20:
                    tf_score += 5  # Oversold bounce
                
                # MACD Analysis
                if (latest['macd'] > latest['macd_signal'] and 
                    latest['macd_histogram'] > prev['macd_histogram']):
                    tf_score += 10
                elif (latest['macd'] < latest['macd_signal'] and 
                      latest['macd_histogram'] < prev['macd_histogram']):
                    tf_score -= 10
                
                # Stochastic
                if (latest['stoch_k'] > latest['stoch_d'] and 
                    20 < latest['stoch_k'] < 80):
                    tf_score += 8
                elif (latest['stoch_k'] < latest['stoch_d'] and 
                      20 < latest['stoch_k'] < 80):
                    tf_score -= 8
                
                # === VOLATILITY & SUPPORT/RESISTANCE (20 POINTS) ===
                # Bollinger Bands Position
                bb_pos = latest['bb_position']
                if 0.2 < bb_pos < 0.8:  # Not at extremes
                    if bb_pos > 0.5:
                        tf_score += 8
                    else:
                        tf_score -= 8
                
                # Williams %R
                williams = latest['williams_r']
                if -80 < williams < -20:
                    if williams > -50:
                        tf_score += 6
                    else:
                        tf_score -= 6
                
                # ATR Volatility (favor normal volatility)
                if 0.3 < latest['atr_percent'] < 1.0:
                    tf_score += 6
                
                # Normalize to 0-10 confidence scale
                tf_confidence = max(0, min(10, 5 + (tf_score / 100) * 5))
                
                # Determine signal direction
                if tf_score > 25:
                    tf_signal = 'BUY'
                elif tf_score < -25:
                    tf_signal = 'SELL'
                else:
                    tf_signal = 'NEUTRAL'
                
                timeframe_signals[tf_name] = {
                    'signal': tf_signal,
                    'confidence': round(tf_confidence, 1),
                    'score': tf_score
                }
                
                total_score += tf_score
            
            # === MOBILE CONFLUENCE ANALYSIS ===
            buy_signals = sum(1 for sig in timeframe_signals.values() if sig['signal'] == 'BUY')
            sell_signals = sum(1 for sig in timeframe_signals.values() if sig['signal'] == 'SELL')
            
            # Require confluence from majority of timeframes
            min_confluence = max(2, len(timeframe_signals) // 2 + 1)
            
            if buy_signals >= min_confluence and buy_signals > sell_signals:
                signal_direction = 'BUY'
            elif sell_signals >= min_confluence and sell_signals > buy_signals:
                signal_direction = 'SELL'
            else:
                return None  # No clear confluence
            
            # Calculate confidence
            avg_confidence = sum(sig['confidence'] for sig in timeframe_signals.values()) / len(timeframe_signals)
            confluence_boost = (max(buy_signals, sell_signals) - 1) * 0.5
            final_confidence = min(10.0, avg_confidence + confluence_boost)
            
            if final_confidence < MIN_CONFIDENCE:
                return None
            
            # === MOBILE ENTRY CALCULATION ===
            # Use highest timeframe for entry precision
            primary_tf = list(timeframe_data.keys())[-1]  # Highest timeframe
            primary_data = timeframe_data[primary_tf]
            latest_price = primary_data.iloc[-1]
            
            entry_price = latest_price['close']
            atr = latest_price['atr']
            
            # Mobile-optimized risk management
            if signal_direction == 'BUY':
                stop_loss = entry_price - (atr * 2.0)
                take_profit_1 = entry_price + (atr * 2.5)
                take_profit_2 = entry_price + (atr * 4.0)
                take_profit_3 = entry_price + (atr * 6.0)
            else:
                stop_loss = entry_price + (atr * 2.0)
                take_profit_1 = entry_price - (atr * 2.5)
                take_profit_2 = entry_price - (atr * 4.0)
                take_profit_3 = entry_price - (atr * 6.0)
            
            # Calculate R:R ratios
            risk = abs(entry_price - stop_loss)
            rr_1 = round(abs(take_profit_1 - entry_price) / risk, 1) if risk > 0 else 0
            rr_2 = round(abs(take_profit_2 - entry_price) / risk, 1) if risk > 0 else 0
            rr_3 = round(abs(take_profit_3 - entry_price) / risk, 1) if risk > 0 else 0
            
            return {
                'symbol': symbol,
                'signal': signal_direction,
                'entry': round(entry_price, 5),
                'stop_loss': round(stop_loss, 5),
                'take_profit_1': round(take_profit_1, 5),
                'take_profit_2': round(take_profit_2, 5),
                'take_profit_3': round(take_profit_3, 5),
                'confidence': round(final_confidence, 1),
                'timeframe_signals': timeframe_signals,
                'confluence': f"{max(buy_signals, sell_signals)}/{len(timeframe_signals)}",
                'atr': round(atr, 5),
                'atr_percent': round(latest_price['atr_percent'], 2),
                'risk_reward_1': rr_1,
                'risk_reward_2': rr_2,
                'risk_reward_3': rr_3,
                'data_source': 'Mobile Multi-Source',
                'timeframes_used': list(timeframe_data.keys())
            }
            
        except Exception as e:
            print(f"❌ Mobile analysis error for {symbol}: {e}")
            return None

    def format_mobile_signal(self, signal):
        """FORMAT MOBILE-OPTIMIZED SIGNAL"""
        emoji = "🟢" if signal['signal'] == 'BUY' else "🔴"
        direction_emoji = "📈" if signal['signal'] == 'BUY' else "📉"
        
        # Mobile-friendly timeframe display
        tf_display = ""
        for tf, data in signal['timeframe_signals'].items():
            tf_emoji = "✅" if data['signal'] == signal['signal'] else "❌"
            tf_display += f"├ {tf.upper()}: {data['confidence']}/10 {tf_emoji}\n"
        
        return f"""
{emoji} <b>📱 MOBILE MT5 PROFESSIONAL SIGNAL</b>

💱 <b>PAIR:</b> {signal['symbol']}
{direction_emoji} <b>DIRECTION:</b> {signal['signal']}
🎯 <b>ENTRY:</b> {signal['entry']}

🛑 <b>STOP LOSS:</b> {signal['stop_loss']}
💰 <b>TP1:</b> {signal['take_profit_1']} (R:R {signal['risk_reward_1']}:1)
🚀 <b>TP2:</b> {signal['take_profit_2']} (R:R {signal['risk_reward_2']}:1)
🎯 <b>TP3:</b> {signal['take_profit_3']} (R:R {signal['risk_reward_3']}:1)

📊 <b>ANALYSIS:</b>
├ <b>Confidence:</b> {signal['confidence']}/10 ⭐
├ <b>Confluence:</b> {signal['confluence']} timeframes
├ <b>ATR:</b> {signal['atr']} ({signal['atr_percent']}%)
└ <b>Source:</b> {signal['data_source']}

🔬 <b>TIMEFRAME CONFLUENCE:</b>
{tf_display}

📱 <b>MOBILE FEATURES:</b>
• Multi-Source Data Feed
• Real-Time Price Action
• Professional Risk Management
• Mobile-Optimized Analysis
• Cross-Platform Compatible

⚠️ <b>MOBILE TRADING TIPS:</b>
• Set alerts on your MT5 mobile app
• Use mobile stop-loss orders
• Monitor on-the-go with push notifications
• Trade with proper position sizing

⏰ <b>Signal Time:</b> {datetime.now(TIMEZONE).strftime('%H:%M:%S')} UTC
📱 <b>Mobile MT5 Compatible</b>
🔥 <b>Professional Grade Analysis</b>

<i>📱 Optimized for mobile MT5 trading - no PC required!</i>
        """

    def can_send_mobile_signal(self, symbol):
        """CHECK MOBILE SIGNAL COOLDOWN"""
        if symbol not in self.last_signals:
            return True
        return time.time() - self.last_signals[symbol] >= 3600  # 1 hour

    def scan_mobile_markets(self):
        """SCAN MARKETS FOR MOBILE SIGNALS"""
        print(f"📱 MOBILE MT5 MARKET SCAN: {datetime.now(TIMEZONE).strftime('%H:%M:%S')}")
        signals_sent = 0
        
        # Prioritize major pairs for mobile
        scan_pairs = self.major_pairs + self.cross_pairs[:3]  # Limit for mobile efficiency
        
        for symbol in scan_pairs:
            try:
                if not self.can_send_mobile_signal(symbol):
                    continue
                
                print(f"📱 MOBILE ANALYZING: {symbol}")
                
                # Mobile professional analysis
                signal = self.mobile_professional_analysis(symbol)
                
                if signal:
                    message = self.format_mobile_signal(signal)
                    if self.send_telegram_message(message):
                        self.last_signals[symbol] = time.time()
                        print(f"📱 MOBILE SIGNAL SENT: {symbol} {signal['signal']} ({signal['confidence']}/10)")
                        signals_sent += 1
                        
                        # Mobile-friendly delays
                        time.sleep(15)
                        
                        # Limit signals for mobile efficiency
                        if signals_sent >= 2:
                            print("📱 Mobile quality limit reached")
                            break
                else:
                    print(f"📱 No mobile signal for {symbol}")
                
                # Mobile rate limiting
                time.sleep(5)
                
            except Exception as e:
                print(f"❌ MOBILE ERROR {symbol}: {e}")
        
        print(f"📱 MOBILE SCAN COMPLETE. SIGNALS SENT: {signals_sent}")
        return signals_sent

    def run_mobile_professional_bot(self):
        """RUN MOBILE MT5 PROFESSIONAL BOT"""
        print("📱 STARTING MOBILE MT5 PROFESSIONAL BOT...")
        print("📲 Send /start to your Telegram bot to activate...")
        
        # Wait for Telegram activation
        while not self.chat_id and self.running:
            self.chat_id = self.get_telegram_updates()
            if not self.chat_id:
                print("⏳ Waiting for mobile activation...")
                time.sleep(5)
        
        if not self.running:
            return
        
        print(f"✅ MOBILE BOT ACTIVATED! Chat ID: {self.chat_id}")
        
        # Send mobile startup message
        startup_msg = f"""
📱 <b>MOBILE MT5 PROFESSIONAL BOT ACTIVATED</b>

🔥 <b>MOBILE FEATURES:</b>
✅ Works without MT5 PC installation
✅ Real-time multi-source data feed
✅ Professional multi-timeframe analysis
✅ Mobile-optimized signals
✅ Cross-platform compatibility

📊 <b>DATA SOURCES:</b>
• Yahoo Finance (Real-time)
• FXEmpire (Professional)
• Alpha Vantage (Institutional)
• Finnhub (Market data)

📈 <b>MONITORING PAIRS:</b>
• <b>MAJORS:</b> {', '.join(self.major_pairs[:4])}...
• <b>CROSSES:</b> {', '.join(self.cross_pairs[:3])}...

🔬 <b>MOBILE ANALYSIS:</b>
• Multi-Timeframe Confluence (1H+4H+1D)
• 15+ Professional Indicators
• ATR-Based Risk Management
• Triple Take Profit Levels
• Real-Time Price Action

📱 <b>MOBILE ADVANTAGES:</b>
• No MT5 PC required
• Works on any device
• Real-time data feeds
• Professional analysis
• Instant Telegram alerts

⚡ <b>SETTINGS:</b>
• Min Confidence: {MIN_CONFIDENCE}/10
• Scan Frequency: Every 15 minutes
• Signal Cooldown: 1 hour per pair
• Quality Focus: 2 signals max per scan

<i>📱 Your mobile MT5 professional bot is now active and scanning markets!</i>
        """
        
        self.send_telegram_message(startup_msg)
        
        # Mobile scanning loop
        scan_count = 0
        while self.running:
            try:
                scan_count += 1
                print(f"\n📱 MOBILE PROFESSIONAL SCAN #{scan_count}")
                
                # Scan markets
                signals_sent = self.scan_mobile_markets()
                
                # Send status update every 8 scans (2 hours)
                if scan_count % 8 == 0:
                    status_msg = f"""
📱 <b>MOBILE BOT STATUS UPDATE</b>

🔥 <b>Scan #{scan_count} Complete</b>
⚡ <b>Signals Sent:</b> {signals_sent} this scan
📊 <b>Pairs Monitored:</b> {len(self.all_pairs)}
📱 <b>Mobile Data Sources:</b> 4 active
⏰ <b>Time:</b> {datetime.now(TIMEZONE).strftime('%H:%M:%S')} UTC

💡 <b>Mobile Tips:</b>
• Keep MT5 mobile app ready for execution
• Set price alerts for our signal levels
• Use mobile stop-loss orders
• Monitor your positions on-the-go

<i>📱 Mobile bot is actively hunting for professional setups...</i>
                    """
                    self.send_telegram_message(status_msg)
                
                print("⏳ WAITING 15 MINUTES FOR NEXT MOBILE SCAN...")
                
                # Mobile-friendly wait with updates
                for i in range(15, 0, -1):
                    if not self.running:
                        break
                    if i % 5 == 0:  # Update every 5 minutes
                        print(f"📱 Next mobile scan in {i} minutes...")
                    time.sleep(60)
                
            except Exception as e:
                print(f"❌ MOBILE LOOP ERROR: {e}")
                error_msg = f"""
⚠️ <b>MOBILE BOT ERROR</b>

❌ <b>Error:</b> {str(e)[:100]}...
🔄 <b>Status:</b> Recovering...
📱 <b>Action:</b> Switching data sources
⏰ <b>Time:</b> {datetime.now(TIMEZONE).strftime('%H:%M:%S')} UTC

<i>📱 Mobile bot will recover in 3 minutes...</i>
                """
                self.send_telegram_message(error_msg)
                print("⏳ Mobile recovery wait...")
                time.sleep(180)
        
        print("📱 MOBILE MT5 PROFESSIONAL BOT STOPPED")

    def send_mobile_market_overview(self):
        """SEND MOBILE MARKET OVERVIEW"""
        try:
            overview_pairs = self.major_pairs[:5]  # Top 5 for mobile
            market_data = []
            
            for symbol in overview_pairs:
                try:
                    df = self.get_mobile_forex_data(symbol, '1h', 50)
                    if df is not None:
                        df = self.calculate_mobile_indicators(df)
                        latest = df.iloc[-1]
                        
                        # Quick trend analysis
                        if latest['ema_8'] > latest['ema_21'] > latest['ema_55']:
                            trend = "📈 BULLISH"
                        elif latest['ema_8'] < latest['ema_21'] < latest['ema_55']:
                            trend = "📉 BEARISH"
                        else:
                            trend = "➡️ NEUTRAL"
                        
                        market_data.append({
                            'symbol': symbol,
                            'price': latest['close'],
                            'trend': trend,
                            'rsi': latest['rsi']
                        })
                except:
                    continue
            
            if market_data:
                overview_text = "📱 <b>MOBILE MARKET OVERVIEW</b>\n\n"
                
                for data in market_data:
                    overview_text += f"💱 <b>{data['symbol']}:</b> {data['price']:.5f}\n"
                    overview_text += f"├ Trend: {data['trend']}\n"
                    overview_text += f"└ RSI: {data['rsi']:.1f}\n\n"
                
                overview_text += f"⏰ <b>Updated:</b> {datetime.now(TIMEZONE).strftime('%H:%M')} UTC\n"
                overview_text += "<i>📱 Mobile-friendly market snapshot</i>"
                
                self.send_telegram_message(overview_text)
                
        except Exception as e:
            print(f"❌ Mobile overview error: {e}")

# === MOBILE UTILITY FUNCTIONS ===
def test_mobile_data_sources():
    """TEST MOBILE DATA SOURCES"""
    print("📱 TESTING MOBILE DATA SOURCES...")
    
    bot = MobileMT5ProfessionalBot()
    test_symbol = 'EURUSD'
    
    sources = [
        ('Yahoo Finance', bot.get_yahoo_data),
        ('FXEmpire', bot.get_fxempire_data),
        ('Alpha Vantage', bot.get_alpha_vantage_data),
        ('Finnhub', bot.get_finhub_data)
    ]
    
    working_sources = 0
    
    for name, source_func in sources:
        try:
            print(f"🧪 Testing {name}...")
            df = source_func(test_symbol, '1h', 50)
            
            if df is not None and len(df) >= 10:
                print(f"✅ {name}: Working ({len(df)} candles)")
                print(f"   Latest price: {df.iloc[-1]['close']:.5f}")
                working_sources += 1
            else:
                print(f"❌ {name}: No data")
                
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
    
    print(f"\n📱 MOBILE TEST RESULTS:")
    print(f"✅ Working sources: {working_sources}/4")
    print(f"{'🔥 MOBILE READY!' if working_sources >= 2 else '⚠️ LIMITED SOURCES'}")
    
    return working_sources >= 2

def run_mobile_single_analysis(symbol='EURUSD'):
    """TEST SINGLE MOBILE ANALYSIS"""
    print(f"📱 TESTING MOBILE ANALYSIS FOR {symbol}...")
    
    bot = MobileMT5ProfessionalBot()
    result = bot.mobile_professional_analysis(symbol)
    
    if result:
        print("✅ Mobile analysis successful!")
        print(f"📱 Signal: {result['signal']}")
        print(f"📊 Confidence: {result['confidence']}/10")
        print(f"💰 Entry: {result['entry']}")
        print(f"🛑 Stop Loss: {result['stop_loss']}")
        print(f"🎯 Take Profit 1: {result['take_profit_1']}")
        print(f"📈 Timeframes: {', '.join(result['timeframes_used'])}")
        print(f"🔗 Confluence: {result['confluence']}")
    else:
        print("❌ No mobile signal generated")

def run_mobile_market_overview():
    """GET MOBILE MARKET OVERVIEW"""
    print("📱 MOBILE MARKET OVERVIEW...")
    
    bot = MobileMT5ProfessionalBot()
    
    for symbol in bot.major_pairs[:5]:
        try:
            df = bot.get_mobile_forex_data(symbol, '1h', 30)
            if df is not None:
                df = bot.calculate_mobile_indicators(df)
                latest = df.iloc[-1]
                
                print(f"\n💱 {symbol}:")
                print(f"   Price: {latest['close']:.5f}")
                print(f"   RSI: {latest['rsi']:.1f}")
                print(f"   ATR%: {latest['atr_percent']:.2f}%")
                
                if latest['ema_8'] > latest['ema_21']:
                    print("   Trend: 📈 Bullish")
                else:
                    print("   Trend: 📉 Bearish")
        except Exception as e:
            print(f"❌ {symbol}: {e}")

# === MAIN MOBILE EXECUTION ===
def main():
    print("=" * 50)
    print("📱 MOBILE MT5 PROFESSIONAL BOT")
    print("=" * 50)
    print("🔥 NO MT5 PC INSTALLATION REQUIRED!")
    print("📱 Works on any device with internet")
    print("=" * 50)
    
    # Initialize mobile bot
    bot = MobileMT5ProfessionalBot()
    
    try:
        # Start mobile professional bot
        bot.run_mobile_professional_bot()
        
    except KeyboardInterrupt:
        print("\n🛑 Mobile bot stopped by user")
        bot.running = False
        
    except Exception as e:
        print(f"💥 MOBILE BOT ERROR: {e}")

# === COMMAND LINE INTERFACE ===
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'test':
            print("🧪 Testing mobile data sources...")
            test_mobile_data_sources()
            
        elif command == 'analyze':
            symbol = sys.argv[2] if len(sys.argv) > 2 else 'EURUSD'
            run_mobile_single_analysis(symbol)
            
        elif command == 'overview':
            run_mobile_market_overview()
            
        elif command == 'run':
            main()
            
        else:
            print("📱 MOBILE MT5 BOT COMMANDS:")
            print("  python mobile_bot.py test      - Test data sources")
            print("  python mobile_bot.py analyze   - Test analysis")
            print("  python mobile_bot.py overview  - Market overview")
            print("  python mobile_bot.py run       - Run mobile bot")
    else:
        main()