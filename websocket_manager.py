#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import json
import logging
import websockets
import gzip
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

class BingXWebSocketManager:
    
    def __init__(self, api_key: str = None, secret_key: str = None, telegram_bot=None):
        self.api_key = api_key
        self.secret_key = secret_key
        self.telegram_bot = telegram_bot
        self.ws_url = "wss://open-api-ws.bingx.com/market"
        self.connections = {}
        self.callbacks = {}
        self.prices = {}
        self.reconnect_delay = 5
        self.max_reconnect_attempts = 10
        self.running = True
        
        from config import WEBSOCKET_ANALYSIS_SETTINGS
        self.settings = WEBSOCKET_ANALYSIS_SETTINGS
        
        self.price_history = {}
        self.signal_counters = {}
        
        logger.info("✅ BingX WebSocket Manager инициализирован")
    
    async def connect_ticker_stream(self, symbols: List[str], callback: Callable):
        # Показываем первые 5 символов для информации
        preview = ', '.join(symbols[:5])
        if len(symbols) > 5:
            preview += f" и еще {len(symbols)-5}"
        
        stream_name = f"ticker_{preview}"
        
        subscribe_msg = {
            "id": random.randint(1, 10000),
            "reqType": "sub",
            "dataType": "ticker"
        }
        
        asyncio.create_task(self._run_websocket(stream_name, subscribe_msg, symbols, callback))
        logger.info(f"✅ WebSocket подключен для {len(symbols)} символов: {preview}")
    
    async def _run_websocket(self, stream_name: str, subscribe_msg: Dict, symbols: List[str], callback: Callable):
        attempts = 0
        
        while self.running and attempts < self.max_reconnect_attempts:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    logger.info(f"🔌 WebSocket {stream_name} подключен")
                    await ws.send(json.dumps(subscribe_msg))
                    
                    try:
                        await asyncio.wait_for(ws.recv(), timeout=10)
                        logger.info(f"📡 WebSocket {stream_name} подписка подтверждена")
                    except asyncio.TimeoutError:
                        pass
                    
                    while self.running:
                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=60)
                            await self._handle_message(message, symbols, callback)
                        except asyncio.TimeoutError:
                            try:
                                await ws.send(json.dumps({"ping": int(datetime.now().timestamp() * 1000)}))
                            except:
                                pass
                            continue
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning(f"⚠️ WebSocket {stream_name} соединение закрыто")
                            break
                    
                    attempts = 0
                    
            except Exception as e:
                attempts += 1
                logger.error(f"❌ WebSocket ошибка: {e}, попытка {attempts}")
                await asyncio.sleep(self.reconnect_delay * attempts)
    
    async def _handle_message(self, message: str, symbols: List[str], callback: Callable):
        try:
            if isinstance(message, bytes):
                try:
                    decompressed = gzip.decompress(message)
                    message = decompressed.decode('utf-8')
                except:
                    message = message.decode('utf-8', errors='ignore')
            
            data = json.loads(message)
            
            if 'data' in data and 'c' in data.get('data', {}):
                await self._process_ticker(data, symbols, callback)
                
        except Exception as e:
            logger.error(f"Ошибка обработки: {e}")
    
    async def _process_ticker(self, data: Dict, symbols: List[str], callback: Callable):
        try:
            symbol_data = data.get('data', {})
            bingx_symbol = symbol_data.get('s', '')
            
            if not bingx_symbol:
                return
            
            symbol = self._convert_symbol(bingx_symbol)
            if symbol not in symbols:
                return
            
            current_price = float(symbol_data.get('c', 0))
            volume = float(symbol_data.get('v', 0))
            
            self.prices[symbol] = {
                'price': current_price,
                'timestamp': datetime.now(),
                'volume': volume
            }
            
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            self.price_history[symbol].append({
                'price': current_price,
                'time': datetime.now()
            })
            
            max_history = self.settings.get('price_history_size', 100)
            if len(self.price_history[symbol]) > max_history:
                self.price_history[symbol] = self.price_history[symbol][-max_history:]
            
            instant_signal = await self._check_instant_movement(symbol)
            if instant_signal:
                logger.info(f"⚡ Движение {symbol}: {instant_signal['change_percent']:.1f}%")
                await callback('instant', symbol, current_price, instant_signal)
                
        except Exception as e:
            logger.error(f"Ошибка обработки тикера: {e}")
    
    def _convert_symbol(self, bingx_symbol: str) -> str:
        if bingx_symbol.endswith('USDT'):
            coin = bingx_symbol[:-4]
            if coin.startswith('1000'):
                coin = coin[4:]
            return f"{coin}/USDT:USDT"
        return bingx_symbol
    
    async def _check_instant_movement(self, symbol: str) -> Optional[Dict]:
        if symbol not in self.price_history or len(self.price_history[symbol]) < 5:
            return None
        
        history = self.price_history[symbol]
        coin = symbol.split('/')[0].upper()
        majors = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP']
        is_shitcoin = coin not in majors
        coin_type = 'shitcoin' if is_shitcoin else 'major'
        
        for window in self.settings.get('time_windows', [3, 5, 10, 30, 60]):
            target_time = datetime.now() - timedelta(seconds=window)
            oldest_price = None
            oldest_time = None
            
            for record in reversed(history):
                if record['time'] <= target_time:
                    oldest_price = record['price']
                    oldest_time = record['time']
                    break
            
            if not oldest_price:
                continue
            
            current_price = history[-1]['price']
            change_percent = (current_price - oldest_price) / oldest_price * 100
            time_diff = (history[-1]['time'] - oldest_time).total_seconds()
            
            threshold_key = f"{window}s"
            thresholds = self.settings.get('thresholds', {})
            coin_thresholds = thresholds.get(coin_type, {})
            threshold = coin_thresholds.get(threshold_key, 2.0)
            
            if self._check_rate_limit(symbol):
                continue
            
            if abs(change_percent) >= threshold:
                return {
                    'change_percent': change_percent,
                    'time_window': round(time_diff, 1),
                    'start_price': oldest_price,
                    'end_price': current_price,
                    'is_shitcoin': is_shitcoin,
                    'threshold_used': threshold,
                    'window_used': window
                }
        
        return None
    
    def _check_rate_limit(self, symbol: str) -> bool:
        current_minute = datetime.now().strftime('%Y%m%d%H%M')
        
        if symbol not in self.signal_counters:
            self.signal_counters[symbol] = {'count': 1, 'minute': current_minute}
            return False
        
        counter = self.signal_counters[symbol]
        
        if counter['minute'] != current_minute:
            counter['count'] = 1
            counter['minute'] = current_minute
            return False
        
        max_per_minute = self.settings.get('max_signals_per_minute', 5)
        if counter['count'] >= max_per_minute:
            return True
        
        counter['count'] += 1
        return False
    
    def stop(self):
        self.running = False
        logger.info("🛑 WebSocket менеджер остановлен")
