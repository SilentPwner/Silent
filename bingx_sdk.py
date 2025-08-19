# bingx_sdk.py - A Python SDK for the BingX REST and WebSocket API
# Author: AI Assistant
# Version: 1.0.0

import asyncio
import json
import hmac
import hashlib
import time
import logging
import aiohttp
import websockets
import random

class BingXClient:
    """Main client to interact with BingX API."""
    BASE_URL = "https://open-api.bingx.com"
    
    def __init__(self, api_key: str, api_secret: str):
        if not api_key or not api_secret:
            raise ValueError("API key and secret cannot be empty.")
        self.api_key = api_key
        self.api_secret = api_secret
        self.rest = RestClient(self)
        self.websocket = WebSocketClient(self)

    def _generate_signature(self, params_str: str) -> str:
        """Generates the HMAC-SHA256 signature for a request."""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            params_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

class RestClient:
    """Handles REST API requests for BingX."""
    def __init__(self, client: BingXClient):
        self._client = client

    async def _request(self, method: str, path: str, params: dict = None, signed: bool = False):
        """Generic async request method."""
        url = f"{self._client.BASE_URL}{path}"
        if params is None:
            params = {}

        if signed:
            params['timestamp'] = int(time.time() * 1000)
            query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
            signature = self._client._generate_signature(query_string)
            query_string += f"&signature={signature}"
            url = f"{self._client.BASE_URL}{path}?{query_string}"
            
        headers = {
            'X-BX-APIKEY': self._client.api_key,
            'Content-Type': 'application/json'
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, headers=headers) as response:
                    response.raise_for_status()
                    data = await response.json()
                    if data.get('code') != 0:
                        logging.error(f"BingX API Error: {data.get('msg')} (Code: {data.get('code')})")
                    return data
        except aiohttp.ClientError as e:
            logging.error(f"HTTP Request failed: {e}")
            return {"code": -1, "msg": str(e), "data": {}}
        except Exception as e:
            logging.error(f"An unexpected error occurred in RestClient: {e}")
            return {"code": -1, "msg": str(e), "data": {}}

    async def get_balance(self):
        """Fetches account balance."""
        return await self._request('GET', '/openApi/spot/v1/account/balance', signed=True)

    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float = None, quote_order_qty: float = None):
        """Places a spot order."""
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
        }
        if quantity:
            params['quantity'] = f"{quantity:.8f}".rstrip('0').rstrip('.')
        if quote_order_qty:
            params['quoteOrderQty'] = f"{quote_order_qty:.8f}".rstrip('0').rstrip('.')
            
        if 'quantity' not in params and 'quoteOrderQty' not in params:
            raise ValueError("Either quantity or quote_order_qty must be specified.")
            
        return await self._request('POST', '/openApi/spot/v1/trade/order', params=params, signed=True)

    async def get_klines(self, symbol: str, interval: str = '1m', limit: int = 100):
        """Fetches historical k-line (candlestick) data."""
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        return await self._request('GET', '/openApi/spot/v1/market/klines', params=params)

class WebSocketClient:
    """Handles WebSocket connections and subscriptions for BingX."""
    # Note: BingX uses a custom socket.io-like protocol over WebSocket
    WS_URL = "wss://open-api.bingx.com/socket.io/?transport=websocket"

    def __init__(self, client: BingXClient):
        self._client = client
        self._ws = None
        self._on_message_callback = None
        self.connected = False

    async def connect(self, on_message_callback):
        """Connects to the WebSocket and handles the socket.io handshake."""
        self._on_message_callback = on_message_callback
        try:
            self._ws = await websockets.connect(self.WS_URL, ping_interval=18, ping_timeout=10)
            
            # socket.io handshake
            initial_message = await self._ws.recv()
            if initial_message.startswith('0'): # Session ID packet
                await self._ws.send('40') # Acknowledge connection
                self.connected = True
                logging.info("WebSocket connected and handshake completed.")
                asyncio.create_task(self._listen())
                asyncio.create_task(self._send_pings())
                return True
            else:
                logging.error(f"Unexpected initial WebSocket message: {initial_message}")
                return False
        except Exception as e:
            logging.error(f"Failed to connect to WebSocket: {e}")
            self.connected = False
            return False

    async def _send_pings(self):
        """Send periodic pings to keep the connection alive (socket.io style)."""
        while self.connected:
            try:
                await self._ws.send('2') # Ping packet
                await asyncio.sleep(25) # BingX requires pings every 30s
            except websockets.exceptions.ConnectionClosed:
                logging.warning("Connection closed while sending ping.")
                self.connected = False
                break
    
    async def _listen(self):
        """Listens for incoming messages and routes them."""
        try:
            while self.connected:
                message = await self._ws.recv()
                if message == '3': # Pong response to our ping
                    continue
                if message.startswith('42'): # Data packet
                    try:
                        data_part = json.loads(message[2:])
                        event = data_part[0]
                        payload = data_part[1]
                        if event == "prices.update" and self._on_message_callback:
                           await self._on_message_callback(json.dumps(payload)) # Pass only the payload
                    except json.JSONDecodeError:
                        logging.warning(f"Could not decode JSON from WebSocket message: {message}")
                    except Exception as e:
                        logging.error(f"Error processing WebSocket data packet: {e} | Message: {message}")
                
        except websockets.exceptions.ConnectionClosed as e:
            logging.warning(f"WebSocket connection closed: {e}")
        except Exception as e:
            logging.error(f"An error occurred in WebSocket listener: {e}")
        finally:
            self.connected = False

    async def subscribe_to_tickers(self, symbols: list):
        """Subscribes to ticker updates for a list of symbols."""
        if not self._ws or not self.connected:
            logging.error("Cannot subscribe, WebSocket is not connected.")
            return

        # BingX requires one subscription message per symbol
        for symbol in symbols:
            sub_request = {
                "id": f"{symbol}-{int(time.time() * 1000)}",
                "dataType": f"{symbol}@trade" # Real-time trade updates
            }
            try:
                await self._ws.send(f'42["subscribe", {json.dumps(sub_request)}]')
                logging.info(f"Sent subscription request for {symbol}")
                await asyncio.sleep(0.1) # Avoid rate limiting
            except Exception as e:
                logging.error(f"Failed to send subscription for {symbol}: {e}")
    
    async def close(self):
        """Closes the WebSocket connection."""
        if self._ws:
            self.connected = False
            await self._ws.close()
            logging.info("WebSocket connection closed.")