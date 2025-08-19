# Silent1_BingX_v1.py - Trading Bot for BingX with Advanced Features
# Author: AI Assistant
# Version: 1.1.0 (Adapted for BingX with Paper Trading Mode)

import logging
import os
import asyncio
import datetime
import json
import threading
import time
import joblib
import random
import numpy as np
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
from flask import Flask, jsonify
from logging.handlers import RotatingFileHandler

# --- ML & Analysis Imports ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance

# --- Import the new BingX SDK ---
from bingx_sdk import BingXClient

# --- Constants & Configuration ---
# NEW: Paper Trading Configuration
PAPER_TRADING_ENABLED = True  # Set to True for paper trading, False for live trading
PAPER_TRADING_STARTING_BALANCE = 10000.0 # Starting virtual balance in USDT

MODEL_FILE_PATH = 'bingx_trading_model.pkl'
# Use BingX symbol format (e.g., BTC-USDT)
ALL_SYMBOLS = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'XRP-USDT', 'ADA-USDT', 'DOGE-USDT', 'AVAX-USDT']
data_lock = asyncio.Lock()
MAX_DAILY_LOSS = -50.0  # Max loss in USDT (negative value)

# --- Global Status & State Variables ---
system_status = {
    "api_authenticated": False,
    "websocket_connected": False,
    "trading_active": True,
    "server_running": True,
    "last_error": None,
    "connection_errors": [],
    "market_activity": {},
    "model_status": "Not trained yet."
}

# NEW: Paper Wallet for simulated trading
paper_wallet = {
    'USDT': PAPER_TRADING_STARTING_BALANCE
} if PAPER_TRADING_ENABLED else {}

active_positions = {}
trade_history = []
performance_stats = {"win_rate": 0, "max_drawdown": 0, "total_pnl": 0}
daily_pnl = 0.0

# === Load Environment Variables ===
load_dotenv()
# IMPORTANT: Use your BingX API keys in the .env file
API_KEY = os.getenv('BINGX_API_KEY')
API_SECRET = os.getenv('BINGX_API_SECRET')

if not API_KEY or not API_SECRET:
    raise ValueError("CRITICAL: BINGX_API_KEY or BINGX_API_SECRET not found in .env file.")

# === BingX SDK Integration ===
bingx_client = BingXClient(API_KEY, API_SECRET)

# === Enhanced Logging Setup ===
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log_handler = RotatingFileHandler('bingx_bot_diagnostics.log', maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
log_handler.setFormatter(log_formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logging.basicConfig(level=logging.INFO, handlers=[log_handler, console_handler])

# === Connection & Risk Management Functions (Updated for BingX) ===
async def perform_connection_tests():
    """Performs connection and authentication tests with BingX API."""
    system_status["connection_errors"] = []
    logging.info("Testing BingX API Connection...")
    ping_res = await bingx_client.rest.get_klines(symbol='BTC-USDT', limit=1)
    if ping_res.get('code') != 0:
        system_status["connection_errors"].append(f"API Connection Failed: {ping_res.get('msg')}")

    if not system_status["connection_errors"]:
        # In live mode, we test authentication. In paper mode, we skip it.
        if not PAPER_TRADING_ENABLED:
            logging.info("Testing BingX API Authentication...")
            auth_res = await bingx_client.rest.get_balance()
            if auth_res.get('code') != 0:
                system_status["connection_errors"].append(f"API Authentication Failed: {auth_res.get('msg')}")
            else:
                system_status["api_authenticated"] = True
        else:
            logging.info("Paper trading mode: Skipping API authentication test.")
            system_status["api_authenticated"] = True # Assume success for paper mode
    
    return {"success": not system_status["connection_errors"], "errors": system_status["connection_errors"]}

async def emergency_close_all_positions():
    """Closes all active positions in case of a critical error."""
    logging.warning("!!! EMERGENCY: Closing all active positions on BingX. !!!")
    system_status["trading_active"] = False
    for symbol, position in list(active_positions.items()):
        logging.info(f"Closing emergency position for {symbol}")
        # We need a recent price to simulate the close in paper mode
        # This part is tricky as we might not have it during a crash. We'll use entry for simplicity.
        price_for_emergency_close = position.get('entry_price', 0) 
        await execute_real_trade(symbol, "SELL", quantity=position['quantity'], is_emergency=True, current_price=price_for_emergency_close)
    logging.info("All positions have been instructed to close.")

def check_daily_loss_limit():
    """Checks if the daily P&L has exceeded the max loss limit."""
    global daily_pnl
    if daily_pnl <= MAX_DAILY_LOSS:
        if system_status["trading_active"]:
            logging.critical(f"Daily loss limit of ${MAX_DAILY_LOSS} reached. Current PNL: ${daily_pnl:.2f}. Halting all new trades for today.")
            system_status["trading_active"] = False
        return False
    return True

# === Performance Metrics ===
def calculate_performance_metrics(trades):
    """Calculates win rate, max drawdown, and total profit from a list of trades."""
    if not trades:
        return {"win_rate": 0, "max_drawdown": 0, "total_pnl": 0}

    profits = [t['pnl'] for t in trades]
    total_pnl = sum(profits)
    wins = sum(1 for p in profits if p > 0)
    win_rate = wins / len(profits) if profits else 0

    cumulative_pnl = np.cumsum(profits)
    # In paper trading, add starting balance to correctly calculate drawdown from peak equity
    initial_balance = PAPER_TRADING_STARTING_BALANCE if PAPER_TRADING_ENABLED else 0
    equity_curve = initial_balance + cumulative_pnl
    
    high_watermarks = np.maximum.accumulate(equity_curve)
    drawdowns = high_watermarks - equity_curve
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0

    return {
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "total_pnl": total_pnl
    }

# === Order Functions (Updated for Paper Trading) ===
async def fetch_balance():
    """
    Fetches account balance. In paper trading mode, it returns the virtual balance.
    """
    if PAPER_TRADING_ENABLED:
        logging.info(f"Paper trading mode: Fetching virtual balance. Current USDT: ${paper_wallet.get('USDT', 0):.2f}")
        # Mimic the structure of the real API response
        return {'USDT': {'free': paper_wallet.get('USDT', 0)}}

    # Original code for live trading
    balance_data = await bingx_client.rest.get_balance()
    if balance_data.get('code') == 0 and 'data' in balance_data:
        usdt_balance = next((float(asset.get('free', 0)) for asset in balance_data['data']['balances'] if asset['asset'] == 'USDT'), 0.0)
        return {'USDT': {'free': usdt_balance}}
    raise ConnectionError(f"Failed to fetch balance: {balance_data.get('msg')}")

async def execute_real_trade(symbol: str, side: str, quantity: float = None, quote_order_qty: float = None, is_emergency=False, current_price: float = None):
    """
    Places an order. In paper trading mode, it simulates the trade and updates the virtual wallet.
    """
    if side.upper() == 'BUY' and not is_emergency:
        if not system_status["trading_active"]:
            logging.warning(f"Trade blocked for {symbol}: Trading is inactive.")
            return None

    if PAPER_TRADING_ENABLED:
        # --- Paper Trading Simulation ---
        logging.info(f"[PAPER TRADE] Executing {side} for {symbol}")
        if not current_price:
            logging.error("[PAPER TRADE] Critical error: current_price must be provided for paper trading.")
            return None

        if side.upper() == 'BUY':
            if paper_wallet.get('USDT', 0) < quote_order_qty:
                logging.error(f"[PAPER TRADE] Insufficient virtual funds to buy {symbol}. Needed: {quote_order_qty}, Have: {paper_wallet.get('USDT', 0)}")
                return None
            
            simulated_quantity = quote_order_qty / current_price
            paper_wallet['USDT'] -= quote_order_qty
            
            logging.info(f"[PAPER TRADE] BOUGHT {simulated_quantity:.6f} {symbol} for ${quote_order_qty:.2f}. New Wallet Balance: ${paper_wallet['USDT']:.2f}")
            # Mimic the real API response for a successful buy order
            return {
                'orderId': f"paper_{int(time.time() * 1000)}",
                'symbol': symbol,
                'cummulativeQuoteQty': str(quote_order_qty),
                'executedQty': str(simulated_quantity)
            }
        
        elif side.upper() == 'SELL':
            if not quantity:
                logging.error("[PAPER TRADE] Critical error: quantity must be provided for a SELL order.")
                return None
            simulated_quote_qty = quantity * current_price
            paper_wallet['USDT'] += simulated_quote_qty
            
            logging.info(f"[PAPER TRADE] SOLD {quantity:.6f} {symbol} for ${simulated_quote_qty:.2f}. New Wallet Balance: ${paper_wallet['USDT']:.2f}")
            # Mimic the real API response for a successful sell order
            return {
                'orderId': f"paper_{int(time.time() * 1000)}",
                'symbol': symbol,
                'cummulativeQuoteQty': str(simulated_quote_qty),
                'executedQty': str(quantity)
            }
        return None

    # --- Original Live Trading Logic ---
    order_result = await bingx_client.rest.place_order(
        symbol=symbol,
        side=side,
        order_type="MARKET",
        quantity=quantity,
        quote_order_qty=quote_order_qty
    )
    
    if order_result.get('code') == 0:
        logging.info(f"Successfully placed {side} order for {symbol}: {order_result['data']}")
        return order_result['data']
    
    logging.error(f"Failed to place {side} order for {symbol}: {order_result}")
    system_status["last_error"] = f"Order failed: {order_result.get('msg')}"
    return None

# === TA & ML Functions (Unchanged Logic) ===
def detect_market_regime(df):
    if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
        df['regime'] = np.where(df['SMA_20'] > df['SMA_50'], 'Bull', 'Bear')
    else:
        df['regime'] = 'Neutral'
    return df

def calculate_indicators(df):
    if df.empty or len(df) < 50: return df
    try:
        df['SMA_20'] = ta.sma(df['close'], length=20)
        df['SMA_50'] = ta.sma(df['close'], length=50)
        df['RSI'] = ta.rsi(df['close'], length=14)
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            df['MACD'] = macd[f'MACD_{12}_{26}_{9}']
            df['macd_hist'] = macd[f'MACDh_{12}_{26}_{9}']
        bbands = ta.bbands(df['close'], length=20, std=2)
        if bbands is not None and not bbands.empty:
            df['upper_band'] = bbands[f'BBU_20_2.0']
            df['lower_band'] = bbands[f'BBL_20_2.0']
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
        if stoch is not None and not stoch.empty:
            df['stoch_k'] = stoch[f'STOCHk_14_3_3']
        
        df['price_momentum'] = df['close'].pct_change(periods=5)
        df['rsi_slope'] = df['RSI'].diff(periods=3)
        df['volatility'] = (df['high'] - df['low']) / df['close']
        
        df = detect_market_regime(df)
        
        return df.fillna(method='ffill').dropna()
    except Exception as e:
        logging.warning(f"Could not calculate all indicators: {e}")
        return df

def analyze_feature_importance(model, X, y, features):
    try:
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
        importance_df = pd.DataFrame({
            'feature': features,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std,
        }).sort_values('importance_mean', ascending=False)
        logging.info("--- Feature Importance Analysis ---\n" + importance_df.to_string())
    except Exception as e:
        logging.error(f"Could not analyze feature importance: {e}")

def generate_ml_signals(df, force_retrain=False):
    features = [
        'SMA_20', 'RSI', 'MACD', 'stoch_k', 'ATR', 'upper_band', 'lower_band',
        'price_momentum', 'rsi_slope', 'macd_hist', 'volatility'
    ]
    
    if not all(f in df.columns for f in features) or df[features].isnull().values.any():
        return df

    df_clean = df.dropna(subset=features).copy()
    if df_clean.empty: return df

    try:
        if os.path.exists(MODEL_FILE_PATH) and not force_retrain:
            model = joblib.load(MODEL_FILE_PATH)
        else:
            logging.info("Model not found or retraining forced. Starting training process...")
            df_clean['target'] = (df_clean['close'].shift(-5) > df_clean['close']).astype(int)
            df_clean.dropna(inplace=True)
            
            if len(df_clean) < 200:
                logging.warning(f"Not enough data to train model ({len(df_clean)} rows).")
                return df

            X, y = df_clean[features], df_clean['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

            param_grid = {
                'n_estimators': [50, 100], 'max_depth': [8, 10, None],
                'min_samples_split': [2, 5], 'min_samples_leaf': [1, 3]
            }
            grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            joblib.dump(model, MODEL_FILE_PATH)
            accuracy = accuracy_score(y_test, model.predict(X_test))
            system_status["model_status"] = f"Trained. Accuracy: {accuracy:.2%}"
            logging.info(f"New model trained and saved. Accuracy: {accuracy:.2%}, Best Params: {grid_search.best_params_}")
            
            analyze_feature_importance(model, X_test, y_test, features)

        predictions = model.predict(df_clean[features])
        df.loc[df_clean.index, 'ml_signal'] = predictions
        return df

    except Exception as e:
        logging.error(f"Error in ML signal generation: {e}", exc_info=True)
        return df

async def periodic_model_retrainer(dfs):
    while system_status["server_running"]:
        await asyncio.sleep(360 * 60) # Retrain every 6 hours
        try:
            logging.info("--- Kicking off scheduled model retraining ---")
            async with data_lock:
                # Use a copy to avoid locking data for too long
                df_train = dfs.get('BTC-USDT', pd.DataFrame()).copy()
            if not df_train.empty:
                df_processed = calculate_indicators(df_train)
                generate_ml_signals(df_processed, force_retrain=True)
            else:
                logging.warning("Skipping scheduled retrain, not enough data.")
        except Exception as e:
            logging.error(f"Error during periodic model retraining: {e}", exc_info=True)

def is_market_weak(df, sma_period=50, volatility_threshold=0.002):
    if len(df) < sma_period: return True, 'Untradable'
    if 'SMA_50' not in df.columns or 'volatility' not in df.columns: return True, 'No Indicators'
    
    current_price = df['close'].iloc[-1]
    sma_value = df['SMA_50'].iloc[-1]
    trend = 'Bull' if current_price > sma_value else 'Bear'
    
    if trend == 'Bear': return True, 'Bear Trend'
    if df['volatility'].iloc[-1] < volatility_threshold: return True, 'Low Volatility'

    return False, 'Active'

# === Real-Time WebSocket Handler (Updated for BingX) ===
async def websocket_data_handler(symbols_to_watch, dfs):
    """Handles incoming WebSocket data from BingX."""
    async def on_message_callback(message):
        try:
            data = json.loads(message)
            symbol_key = data.get('s')
            if not symbol_key or symbol_key not in symbols_to_watch: return

            last_price = float(data.get('p', 0))
            volume = float(data.get('q', 0))
            if last_price == 0: return

            timestamp = pd.to_datetime(data.get('E'), unit='ms', utc=True)
            
            async with data_lock:
                df_current = dfs.get(symbol_key)
                if df_current is None or df_current.empty: return

                last_candle_ts = df_current.index[-1]
                
                if timestamp < last_candle_ts + pd.Timedelta(minutes=1):
                    # Update the last candle
                    df_current.loc[last_candle_ts, 'close'] = last_price
                    df_current.loc[last_candle_ts, 'high'] = max(df_current.loc[last_candle_ts, 'high'], last_price)
                    df_current.loc[last_candle_ts, 'low'] = min(df_current.loc[last_candle_ts, 'low'], last_price)
                    df_current.loc[last_candle_ts, 'volume'] += volume
                else:
                    # Create a new candle
                    new_candle_ts = timestamp.floor('T')
                    new_candle = pd.DataFrame([{
                        'open': last_price, 'high': last_price,
                        'low': last_price, 'close': last_price, 'volume': volume
                    }], index=[new_candle_ts])
                    new_candle.index.name = 'timestamp'
                    df_current = pd.concat([df_current, new_candle])
                
                df_updated = df_current.tail(1000) # Keep memory usage in check
                df_with_indicators = calculate_indicators(df_updated.copy())
                final_df = generate_ml_signals(df_with_indicators)
                dfs[symbol_key] = final_df

        except Exception as e:
            logging.error(f"Error processing BingX WebSocket message: {e} | Message: {message[:200]}", exc_info=True)

    if await bingx_client.websocket.connect(on_message_callback=on_message_callback):
        system_status["websocket_connected"] = True
        await bingx_client.websocket.subscribe_to_tickers(symbols_to_watch)
        while system_status["server_running"]:
            if not bingx_client.websocket.connected:
                system_status["websocket_connected"] = False
                logging.warning("Detected WebSocket disconnect. Will attempt to reconnect...")
                # Optional: Add reconnect logic here if needed
            await asyncio.sleep(10)
    else:
        system_status["websocket_connected"] = False
        logging.critical("Could not establish WebSocket connection with BingX.")

# === Main Trading Engine (Updated for BingX) ===
async def run_trading_engine():
    global daily_pnl
    dfs = {s: pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume']).set_index(pd.to_datetime([]).rename('timestamp')) for s in ALL_SYMBOLS}
    try:
        if PAPER_TRADING_ENABLED:
            logging.info("--- Starting Trading Engine in PAPER TRADING MODE ---")
        else:
            logging.info("--- Starting Trading Engine v1.1 for BingX in LIVE MODE ---")
        
        await perform_connection_tests()
        if system_status["connection_errors"]:
            raise ConnectionError(f"Initial diagnostics failed: {system_status['connection_errors']}")

        logging.info("Fetching initial historical data...")
        for symbol in ALL_SYMBOLS:
            klines_res = await bingx_client.rest.get_klines(symbol=symbol, interval='1m', limit=500)
            if klines_res.get('code') == 0 and klines_res.get('data'):
                kline_data = klines_res['data']
                df = pd.DataFrame(kline_data, columns=['open', 'close', 'high', 'low', 'volume', 'timestamp'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                df.set_index('timestamp', inplace=True)
                df = df.astype(float)
                df_indicators = calculate_indicators(df)
                dfs[symbol] = generate_ml_signals(df_indicators)
                logging.info(f"Loaded {len(df)} historical klines for {symbol}.")
            else:
                logging.warning(f"Could not fetch historical data for {symbol}.")
            await asyncio.sleep(0.5)

        symbols_for_today = random.sample(ALL_SYMBOLS, random.randint(3, 5))
        logging.info(f"Selected symbols for today: {symbols_for_today}")
        
        balance_info = await fetch_balance()
        total_balance = balance_info.get('USDT', {}).get('free', 0)
        if total_balance < 20:
            raise ValueError(f"Insufficient balance ({total_balance} USDT) to start.")
        
        usdt_per_trade = max(10, (total_balance / len(symbols_for_today)) * 0.15)
        logging.info(f"Allocating ~${usdt_per_trade:.2f} per trade.")

        asyncio.create_task(websocket_data_handler(symbols_for_today, dfs))
        asyncio.create_task(periodic_model_retrainer(dfs))
        
        await asyncio.sleep(15)
        logging.info("--- Trading Engine is now LIVE using BingX WebSocket ---")

        last_day_checked = datetime.date.today()

        while system_status["server_running"]:
            if datetime.date.today() != last_day_checked:
                logging.info(f"New day. Resetting daily PNL from {daily_pnl:.2f} to 0.")
                daily_pnl = 0.0
                last_day_checked = datetime.date.today()
                system_status["trading_active"] = True

            async with data_lock:
                for symbol in symbols_for_today:
                    if not dfs.get(symbol, pd.DataFrame()).empty:
                        await trade_and_manage_position(dfs[symbol].copy(), symbol, usdt_per_trade)
            
            await asyncio.sleep(random.uniform(10, 20))

    except Exception as e:
        logging.critical(f"Trading engine CRASHED: {e}", exc_info=True)
        await emergency_close_all_positions()
    finally:
        system_status["trading_active"] = False
        logging.info("--- Initiating safe shutdown ---")
        if not PAPER_TRADING_ENABLED:
            await emergency_close_all_positions()
        await bingx_client.websocket.close()
        logging.info("--- Trading Engine Shutdown Complete ---")

async def trade_and_manage_position(df, symbol, usdt_per_trade):
    global active_positions, daily_pnl, performance_stats, trade_history
    if df.empty or 'ml_signal' not in df.columns or 'regime' not in df.columns: return

    last_row = df.iloc[-1]
    current_price = last_row['close']
    market_regime = last_row['regime']

    is_weak, reason = is_market_weak(df.tail(60))
    system_status["market_activity"][symbol] = f"{market_regime} - {reason}"
    if is_weak: return

    if symbol in active_positions:
        pos = active_positions[symbol]
        pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
        
        should_close = False
        if last_row['ml_signal'] == 0:
            logging.info(f"SELL signal for {symbol}. Closing position.")
            should_close = True
        elif pnl_pct <= -0.02:
            logging.info(f"STOP LOSS for {symbol}. Closing position.")
            should_close = True
        elif current_price < pos.get('trailing_stop_price', 0):
             logging.info(f"TRAILING STOP for {symbol} triggered. Closing position.")
             should_close = True

        if should_close:
            sell_result = await execute_real_trade(symbol, "SELL", quantity=pos['quantity'], current_price=current_price)
            if sell_result:
                pnl_amount = (current_price - pos['entry_price']) * pos['quantity']
                daily_pnl += pnl_amount
                trade_history.append({'symbol': symbol, 'pnl': pnl_amount, 'pnl_pct': pnl_pct, 'timestamp': datetime.datetime.utcnow()})
                performance_stats = calculate_performance_metrics(trade_history)
                logging.info(f"Closed {symbol}. PNL: ${pnl_amount:.2f}. Daily PNL: ${daily_pnl:.2f}")
                del active_positions[symbol]
        else:
            new_stop_price = current_price * 0.98
            if new_stop_price > pos.get('trailing_stop_price', 0):
                 pos['trailing_stop_price'] = new_stop_price

    elif last_row['ml_signal'] == 1:
        if not check_daily_loss_limit(): return
        if market_regime != 'Bull':
            logging.info(f"Buy signal for {symbol} ignored due to non-Bullish regime ({market_regime}).")
            return
        
        logging.info(f"BUY signal for {symbol} at {current_price:.4f}. Executing trade.")
        order_result = await execute_real_trade(symbol, "BUY", quote_order_qty=usdt_per_trade, current_price=current_price)
        if order_result and 'cummulativeQuoteQty' in order_result:
            filled_amount = float(order_result['cummulativeQuoteQty'])
            filled_quantity = float(order_result['executedQty'])
            entry_price = filled_amount / filled_quantity if filled_quantity > 0 else 0
            
            if entry_price > 0:
                active_positions[symbol] = {
                    'quantity': filled_quantity, 'entry_price': entry_price,
                    'trailing_stop_price': entry_price * 0.98,
                    'timestamp': datetime.datetime.utcnow().isoformat()
                }
                logging.info(f"New position opened for {symbol}: {filled_quantity} @ ${entry_price:.4f}")

# === Web Interface & Startup ===
app = Flask(__name__)
@app.route('/')
def home():
    mode = "Paper Trading" if PAPER_TRADING_ENABLED else "Live Trading"
    return f"<h1>Trading Bot Status (BingX)</h1><h2>Mode: {mode}</h2><p><a href='/status'>View JSON Status</a></p>"

@app.route('/status')
def status():
    system_status["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    
    # Calculate current value of open positions for equity calculation
    open_positions_value = 0
    if PAPER_TRADING_ENABLED:
        current_equity = paper_wallet.get('USDT', 0)
        for symbol, pos in active_positions.items():
            # This part needs a live price feed for accuracy, which we don't have here.
            # We'll use the entry price as an approximation of the asset's value.
            open_positions_value += pos['quantity'] * pos['entry_price']
        current_equity += open_positions_value
    else:
        current_equity = "N/A (Live Mode)"


    status_data = {
        "bot_mode": "Paper Trading" if PAPER_TRADING_ENABLED else "Live Trading",
        "system_status": system_status,
        "performance": {**performance_stats, "daily_pnl_usd": daily_pnl},
        "wallet_and_equity": {
            "paper_wallet_balance_usd": f"{paper_wallet.get('USDT', 0):.2f}" if PAPER_TRADING_ENABLED else "N/A",
            "paper_current_equity_usd": f"{current_equity:.2f}" if PAPER_TRADING_ENABLED else "N/A",
        },
        "risk_management": {"max_daily_loss_usd": MAX_DAILY_LOSS, "trading_enabled": system_status["trading_active"]},
        "active_positions": active_positions,
        "trade_count": len(trade_history),
    }
    return jsonify(status_data)

def run_server():
    port = int(os.getenv("PORT", 5000))
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=port, threaded=True)

async def main():
    threading.Thread(target=run_server, daemon=True).start()
    try:
        await run_trading_engine()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logging.info("Shutdown signal received.")
    finally:
        system_status["server_running"] = False
        await bingx_client.websocket.close()
        await asyncio.sleep(2)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Program exiting.")