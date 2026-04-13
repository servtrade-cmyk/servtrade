#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
import ccxt.async_support as ccxt
from dotenv import load_dotenv
# Telegram
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.error import RetryAfter, TimedOut
import time
import json
import aiohttp
import random
from io import BytesIO
# Для продвинутых структур данных
import heapq
from collections import deque

# Графики
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# Импорт конфигурации
from config import (
    TELEGRAM_TOKEN,
    TELEGRAM_CHAT_ID,
    PUMP_CHAT_ID,
    STATS_CHAT_ID,
    ACCUMULATION_CHAT_ID,
    UPDATE_INTERVAL,
    PUMP_SCAN_INTERVAL,
    MIN_CONFIDENCE,
    TIMEFRAMES,
    REF_LINKS,
    FEATURES,
    INDICATOR_SETTINGS,
    INDICATOR_WEIGHTS,
    PUMP_DUMP_SETTINGS,
    PUMP_SCAN_SETTINGS,
    IMBALANCE_SETTINGS,
    LIQUIDITY_SETTINGS,
    SMC_SETTINGS,
    FRACTAL_SETTINGS,
    PAIRS_TO_SCAN,
    DISPLAY_SETTINGS,
    FIBONACCI_SETTINGS,
    FIB_HISTORY_SETTINGS,
    VOLUME_PROFILE_SETTINGS,
    STATS_SETTINGS,
    ACCUMULATION_SETTINGS,
    ATR_SETTINGS,
    PERFORMANCE_SETTINGS,
    VOLUME_ANALYSIS_SETTINGS,
    DISPERSION_ANALYSIS_SETTINGS,
    LEVEL_ANALYSIS_SETTINGS,      
    LEVEL_STRENGTH_SETTINGS,
    SNIPER_ENTRY_SETTINGS,
    SCAN_MODE,
    SMART_REPEAT_SETTINGS,
    PUMP_DUMP_FILTER,
    WEBSOCKET_ANALYSIS_SETTINGS,
    FAKEOUT_SETTINGS,
    TIMEFRAME_WEIGHTS,
    EMA_TOUCH_SETTINGS,
    BREAKOUT_CONFIRMATION_SETTINGS,
    MINOR_TF_SETTINGS,      
    EXCHANGES,              
    PROXY_SETTINGS,         
    SIGNAL_FORMAT_SETTINGS,   
    TF_ALIGNMENT_SETTINGS,      
    DYNAMIC_TARGET_SETTINGS,
    ACCUMULATION_SIGNAL_SETTINGS,
    ENTRY_ZONES_SETTINGS,
    SIGNAL_TYPE_SETTINGS,
    STOP_HUNT_SETTINGS,
    POST_STOP_HUNT_SETTINGS,
    LIQUIDITY_ZONES_SETTINGS,
    FVG_SETTINGS,
    SIGNAL_TIMEFRAMES,
)

# from config import BREAKOUT_CONFIRMATION_SETTINGS

# Импорт системы статистики
from signal_stats import SignalStatistics

# Импорт настроек объемов и дисперсии
from config import VOLUME_ANALYSIS_SETTINGS, DISPERSION_ANALYSIS_SETTINGS

# Импорт снайперские точки входа
from config import SNIPER_ENTRY_SETTINGS

from config import ACCUMULATION_SIGNAL_SETTINGS

from config import STRATEGY_SETTINGS

from config import PATTERN_SETTINGS

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ============== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==============

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_atr(high, low, close, period=14):
    high_low = high - low
    high_close = abs(high - close.shift())
    low_close = abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def calculate_bollinger_bands(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return sma, upper, lower

def calculate_sma(series, period):
    return series.rolling(window=period).mean()

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap

# ============== КЭШИРОВАНИЕ ДАННЫХ ==============

class CacheManager:
    """Кэширование данных для уменьшения количества запросов к бирже"""
    
    def __init__(self, ttl=60):
        self.cache = {}
        self.ttl = ttl
        logger.info(f"✅ CacheManager инициализирован (TTL: {ttl} сек)")
    
    def get(self, key: str) -> Optional[any]:
        """Получение данных из кэша"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, data: any):
        """Сохранение данных в кэш"""
        self.cache[key] = (data, time.time())
    
    def clear(self):
        """Очистка кэша"""
        self.cache.clear()
        logger.info("🧹 Кэш очищен")

# ============== ИСТОРИЯ ПОДХОДОВ К УРОВНЯМ ФИБОНАЧЧИ ==============

class FibHistoryTracker:
    """
    Хранит историю подходов к уровням Фибоначчи на разных таймфреймах
    """
    
    def __init__(self):
        self.history = {}  # key: f"{symbol}_{tf}_{level}"
        self.ttl = FIB_HISTORY_SETTINGS.get('ttl_seconds', 2592000)
        self.max_approaches = FIB_HISTORY_SETTINGS.get('max_approaches', 10)
        logger.info("✅ FibHistoryTracker инициализирован")
    
    def _make_key(self, symbol: str, tf: str, level: float) -> str:
        """Создание ключа для истории"""
        return f"{symbol}_{tf}_{level:.3f}"
    
    def add_approach(self, symbol: str, tf: str, level: float, price: float):
        """
        Добавление подхода к уровню
        """
        key = self._make_key(symbol, tf, level)
        now = datetime.now().timestamp()
        
        if key not in self.history:
            self.history[key] = {
                'symbol': symbol,
                'tf': tf,
                'level': level,
                'price': price,
                'approaches': 1,
                'first_approach': now,
                'last_approach': now,
                'is_broken': False,
                'broken_at': None
            }
            logger.debug(f"📊 Фибо {tf} {level:.3f} для {symbol}: первый подход")
        else:
            # Проверяем, не устарел ли уровень
            if now - self.history[key]['last_approach'] > self.ttl:
                # Устарел — сбрасываем
                self.history[key]['approaches'] = 1
                self.history[key]['first_approach'] = now
                logger.debug(f"📊 Фибо {tf} {level:.3f} для {symbol}: сброс (устарел)")
            else:
                self.history[key]['approaches'] = min(self.history[key]['approaches'] + 1, self.max_approaches)
                logger.debug(f"📊 Фибо {tf} {level:.3f} для {symbol}: подход #{self.history[key]['approaches']}")
            
            self.history[key]['last_approach'] = now
            self.history[key]['price'] = price
    
    def get_approach_count(self, symbol: str, tf: str, level: float) -> int:
        """
        Получение количества подходов к уровню
        """
        key = self._make_key(symbol, tf, level)
        if key not in self.history:
            return 0
        
        # Проверяем на устаревание
        now = datetime.now().timestamp()
        if now - self.history[key]['last_approach'] > self.ttl:
            del self.history[key]
            return 0
        
        return self.history[key]['approaches']
    
    def mark_broken(self, symbol: str, tf: str, level: float, price: float):
        """
        Отметить уровень как пробитый
        """
        key = self._make_key(symbol, tf, level)
        if key in self.history:
            self.history[key]['is_broken'] = True
            self.history[key]['broken_at'] = datetime.now().timestamp()
            self.history[key]['broken_price'] = price
            logger.debug(f"📊 Фибо {tf} {level:.3f} для {symbol}: ПРОБИТ")
    
    def get_strength_text(self, approaches: int) -> str:
        """
        Получить текстовое описание силы уровня по количеству подходов
        """
        if approaches == 0:
            return ""
        elif approaches == 1:
            return "1-й подход (экстремально сильный)"
        elif approaches == 2:
            return "2-й подход (очень сильный)"
        elif approaches == 3:
            return "3-й подход (сильный)"
        else:
            return f"{approaches}-й подход (ослаблен)"
    
    def cleanup(self):
        """Очистка устаревших записей"""
        now = datetime.now().timestamp()
        to_delete = []
        for key, data in self.history.items():
            if now - data['last_approach'] > self.ttl:
                to_delete.append(key)
        for key in to_delete:
            del self.history[key]
        if to_delete:
            logger.info(f"🧹 FibHistoryTracker: удалено {len(to_delete)} устаревших записей")

# ============== АНАЛИЗАТОР НАКОПЛЕНИЯ ==============

class AccumulationAnalyzer:
    """
    Анализатор фаз накопления/распределения
    Находит моменты, когда крупные игроки собирают позицию перед импульсом
    """
    
    def __init__(self, settings: Dict = None):
        self.settings = settings or ACCUMULATION_SETTINGS
        self.ad_threshold = self.settings.get('ad_threshold', 2.0)
        self.volume_spike_threshold = self.settings.get('volume_spike_threshold', 2.0)
        self.range_width_threshold = self.settings.get('range_width_threshold', 5.0)
        self.min_signals = self.settings.get('min_signals', 2)
        self.lookback = self.settings.get('lookback_period', 50)
    
    def calculate_ad_line(self, df: pd.DataFrame) -> pd.Series:
        """Расчет линии накопления/распределения (A/D)"""
        high_low = df['high'] - df['low']
        high_low = high_low.replace(0, 0.001)
        
        money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / high_low
        money_flow_volume = money_flow_multiplier * df['volume']
        return money_flow_volume.cumsum()
    
    def detect_ad_divergence(self, df: pd.DataFrame) -> Dict:
        """Поиск дивергенции между ценой и A/D линией"""
        df = df.copy()
        df['ad_line'] = self.calculate_ad_line(df)
        
        recent = df.tail(20)
        
        price_lows = recent['close'].rolling(5).min()
        ad_lows = recent['ad_line'].rolling(5).min()
        
        price_trend = price_lows.is_monotonic_decreasing
        ad_trend = ad_lows.is_monotonic_increasing
        
        if price_trend and ad_trend:
            strength = min(80, abs(price_lows.iloc[-1] - price_lows.iloc[0]) / price_lows.iloc[0] * 500)
            return {
                'accumulation': True,
                'strength': strength,
                'description': f"📈 Дивергенция: цена падает, A/D растет (сила {strength:.0f}%)"
            }
        
        price_highs = recent['close'].rolling(5).max()
        ad_highs = recent['ad_line'].rolling(5).max()
        
        price_trend_up = price_highs.is_monotonic_increasing
        ad_trend_down = ad_highs.is_monotonic_decreasing
        
        if price_trend_up and ad_trend_down:
            strength = min(80, abs(price_highs.iloc[-1] - price_highs.iloc[0]) / price_highs.iloc[0] * 500)
            return {
                'accumulation': True,
                'distribution': True,
                'strength': strength,
                'description': f"📉 Распределение: цена растет, A/D падает (сила {strength:.0f}%)"
            }
        
        return {'accumulation': False}
    
    def detect_volume_spikes_in_range(self, df: pd.DataFrame) -> Dict:
        """Поиск всплесков объема внутри консолидации"""
        recent = df.tail(self.lookback)
        
        range_high = recent['high'].max()
        range_low = recent['low'].min()
        current_price = df['close'].iloc[-1]
        
        range_width = (range_high - range_low) / range_low * 100
        
        if current_price <= range_high and current_price >= range_low and range_width < self.range_width_threshold:
            avg_volume = recent['volume'].mean()
            last_volume = recent['volume'].iloc[-5:].mean()
            volume_ratio = last_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio > self.volume_spike_threshold:
                strength = min(90, volume_ratio * 30)
                return {
                    'accumulation': True,
                    'strength': strength,
                    'description': f"📊 Аномальный объем x{volume_ratio:.1f} в консолидации (сила {strength:.0f}%)"
                }
        
        return {'accumulation': False}
    
    def detect_silent_accumulation(self, df: pd.DataFrame) -> Dict:
        """Поиск тихой аккумуляции"""
        recent = df.tail(30)
        
        price_range = (recent['high'].max() - recent['low'].min()) / recent['close'].mean() * 100
        
        if price_range < 3:
            volume_sma_5 = recent['volume'].tail(5).mean()
            volume_sma_20 = recent['volume'].mean()
            volume_ratio = volume_sma_5 / volume_sma_20 if volume_sma_20 > 0 else 1
            
            lows_increasing = recent['low'].tail(10).is_monotonic_increasing
            
            signals = 0
            reasons = []
            
            if volume_ratio > 1.3:
                signals += 1
                reasons.append(f"объем +{(volume_ratio-1)*100:.0f}%")
            
            if lows_increasing:
                signals += 1
                reasons.append("минимумы растут")
            
            if signals >= 1:
                strength = signals * 35
                return {
                    'accumulation': True,
                    'strength': strength,
                    'description': f"📦 Тихая аккумуляция: {', '.join(reasons)} (сила {strength:.0f}%)"
                }
        
        return {'accumulation': False}
    
    def calculate_potential(self, df: pd.DataFrame, dataframes: Dict[str, pd.DataFrame], 
                        fvg_analysis: Dict = None, liquidity_zones: List = None) -> Dict:
        """
        Расчет потенциала роста до ближайшей сильной зоны на старших ТФ
        С учетом FVG и зон ликвидности
        """
        # ✅ ДОБАВИТЬ СЛОВАРЬ ПЕРЕВОДА
        tf_short = {
            'monthly': '1м',
            'weekly': '1н',
            'daily': '1д',
            'four_hourly': '4ч',
            'hourly': '1ч',
            'current': '15м'
        }
        
        current_price = df['close'].iloc[-1]
        potential = {
            'has_potential': False,
            'target_price': None,
            'target_pct': 0,
            'target_level': '',
            'timeframe': '',
            'reasons': [],
            'confluence_strength': 0,
            'level_count': 0
        }
        
        # Анализируем старшие таймфреймы
        target_tfs = ['hourly', 'daily', 'weekly', 'monthly']
        
        for tf_name in target_tfs:
            if tf_name not in dataframes or dataframes[tf_name] is None:
                continue
            
            tf_df = dataframes[tf_name]
            
            # Получаем FVG для этого ТФ из анализа
            tf_fvg = []
            if fvg_analysis and fvg_analysis.get('zones'):
                tf_fvg = [z for z in fvg_analysis['zones'] if z.get('tf') == tf_name]
            
            # Получаем зоны ликвидности для этого ТФ
            tf_liquidity = []
            if liquidity_zones:
                tf_liquidity = [z for z in liquidity_zones if z.get('timeframe') == tf_name]
            
            # Ищем уровни с учетом FVG и ликвидности
            levels = self._find_strong_levels(tf_df, tf_fvg, tf_liquidity)
            
            # Ищем конфлюенцию
            confluence_zones = self.find_confluence(levels, current_price, tolerance=0.5)
            
            # Проверяем зоны конфлюенции
            for zone in confluence_zones:
                zone_price = zone['price']
                zone_type = zone['zone_type']
                
                if zone_type == 'resistance' and zone_price > current_price:
                    distance = ((zone_price - current_price) / current_price) * 100
                    
                    if distance < 50:
                        if not potential['target_price'] or zone_price < potential['target_price']:
                            potential['has_potential'] = True
                            potential['target_price'] = zone_price
                            potential['target_pct'] = round(distance, 2)
                            potential['target_level'] = zone['description']
                            potential['timeframe'] = tf_name
                            potential['confluence_strength'] = zone['strength']
                            potential['level_count'] = zone['count']
                            
                            # Эмодзи в зависимости от силы
                            if zone['count'] >= 4:
                                emoji = "🔥🔥🔥"
                            elif zone['count'] >= 3:
                                emoji = "🔥🔥"
                            elif zone['count'] >= 2:
                                emoji = "🔥"
                            else:
                                emoji = "⭐"
                            
                            potential['reasons'].append(
                                f"{emoji} {zone['description']} на {tf_short.get(tf_name, tf_name)}: +{distance:.2f}% "
                                f"(сила {zone['strength']}%, {zone['count']} уровней)"
                            )
                            break
            
            # Если нет конфлюенции — берем ближайший одиночный уровень
            if not potential['has_potential']:
                for level in levels:
                    level_price = level['price']
                    level_type = level['type']
                    
                    if level_price > current_price:
                        distance = ((level_price - current_price) / current_price) * 100
                        
                        if distance < 50:
                            if not potential['target_price'] or level_price < potential['target_price']:
                                potential['has_potential'] = True
                                potential['target_price'] = level_price
                                potential['target_pct'] = round(distance, 2)
                                potential['target_level'] = level_type
                                potential['timeframe'] = tf_name
                                potential['confluence_strength'] = level['strength']
                                potential['level_count'] = 1
                                potential['reasons'].append(
                                    f"📊 До {level_type} на {tf_short.get(tf_name, tf_name)}: +{distance:.2f}%"
                                )
                                break
        
        return potential
    
    def _find_strong_levels(self, df: pd.DataFrame, fvg_zones: List = None, liquidity_zones: List = None) -> List[Dict]:
        levels = []
        
        # 1. EMA (короткие)
        short_emas = [7, 14, 21, 28, 50]
        for period in short_emas:
            col = f'ema_{period}'
            if col in df.columns and pd.notna(df[col].iloc[-1]):
                levels.append({
                    'price': df[col].iloc[-1],
                    'type': f'EMA {period}',
                    'strength': 60 + (period // 10),
                    'category': 'ema_short'
                })
        
        # 2. SMA (длинные)
        long_smas = [50, 100, 200]
        for period in long_smas:
            col = f'sma_{period}'
            if col in df.columns and pd.notna(df[col].iloc[-1]):
                strength = 70 if period == 50 else 75 if period == 100 else 85
                levels.append({
                    'price': df[col].iloc[-1],
                    'type': f'SMA {period}',
                    'strength': strength,
                    'category': 'sma_long'
                })
        
        # 3. VWAP
        if 'vwap' in df.columns and pd.notna(df['vwap'].iloc[-1]):
            levels.append({
                'price': df['vwap'].iloc[-1],
                'type': 'VWAP',
                'strength': 80,
                'category': 'vwap'
            })
        
        # 4. Локальные экстремумы
        recent = df.tail(50)
        levels.append({
            'price': recent['high'].max(),
            'type': 'Локальный максимум',
            'strength': 60,
            'category': 'swing'
        })
        levels.append({
            'price': recent['low'].min(),
            'type': 'Локальный минимум',
            'strength': 60,
            'category': 'swing'
        })
        
        # 5. Уровни Фибоначчи (0.236, 0.382, 0.5, 0.618, 0.786)
        if 'fib_levels' in df.columns:
            fib_levels = ['0.236', '0.382', '0.5', '0.618', '0.786']
            for fib in fib_levels:
                col = f'fib_{fib}'
                if col in df.columns and pd.notna(df[col].iloc[-1]):
                    strength = 75 if fib == '0.618' else 65
                    levels.append({
                        'price': df[col].iloc[-1],
                        'type': f'Фибо {fib}',
                        'strength': strength,
                        'category': 'fibonacci'
                    })
        
        # 6. FVG зоны
        if fvg_zones:
            for fvg in fvg_zones[:3]:
                if fvg['type'] == 'bullish':
                    level_price = fvg['max']
                    level_type = f"FVG {fvg.get('tf_short', '')}"
                else:
                    level_price = fvg['min']
                    level_type = f"FVG {fvg.get('tf_short', '')}"
                
                levels.append({
                    'price': level_price,
                    'type': level_type,
                    'strength': fvg.get('strength', 75),
                    'category': 'fvg'
                })
        
        # 7. Зоны ликвидности
        if liquidity_zones:
            for zone in liquidity_zones[:3]:
                levels.append({
                    'price': zone['price'],
                    'type': f"Зона ликвидности ({zone['type']})",
                    'strength': zone.get('strength', 70),
                    'category': 'liquidity'
                })
        
        return levels

    def find_confluence(self, levels: List[Dict], current_price: float, tolerance: float = 0.5) -> List[Dict]:
        """
        Поиск сходящихся уровней в одной ценовой зоне
        tolerance: допустимое отклонение в процентах (0.5% = уровни в радиусе 0.5% от цены)
        
        Возвращает список зон конфлюенции, отсортированных по силе
        """
        if not levels:
            return []
        
        # Группируем уровни по цене (с допуском)
        zones = []
        used = set()
        
        for i, level1 in enumerate(levels):
            if i in used:
                continue
            
            # Создаем новую зону
            zone = {
                'price': level1['price'],
                'levels': [level1],
                'types': [level1['type']],
                'categories': [level1.get('category', 'unknown')],
                'strength': level1['strength'],
                'count': 1
            }
            
            # Ищем уровни, близкие к текущему
            for j, level2 in enumerate(levels):
                if j == i or j in used:
                    continue
                
                # Разница в процентах
                diff = abs(level1['price'] - level2['price']) / level1['price'] * 100
                
                if diff <= tolerance:
                    zone['levels'].append(level2)
                    zone['types'].append(level2['type'])
                    zone['categories'].append(level2.get('category', 'unknown'))
                    zone['strength'] += level2['strength']
                    zone['count'] += 1
                    # Усредняем цену зоны
                    zone['price'] = (zone['price'] + level2['price']) / 2
                    used.add(j)
            
            # Если в зоне больше 1 уровня — это конфлюенция
            if zone['count'] > 1:
                # Нормализуем силу (максимум 100)
                zone['strength'] = min(100, zone['strength'])
                
                # Определяем тип зоны (поддержка или сопротивление)
                if zone['price'] < current_price:
                    zone['zone_type'] = 'support'
                    zone['direction'] = 'LONG'
                else:
                    zone['zone_type'] = 'resistance'
                    zone['direction'] = 'SHORT'
                
                # Расстояние до зоны
                zone['distance_pct'] = abs(zone['price'] - current_price) / current_price * 100
                
                # Формируем описание
                unique_types = list(dict.fromkeys(zone['types']))  # убираем дубликаты
                if len(unique_types) == 2:
                    zone['description'] = f"КОНФЛЮЕНЦИЯ: {unique_types[0]} + {unique_types[1]}"
                else:
                    zone['description'] = f"КОНФЛЮЕНЦИЯ: {', '.join(unique_types[:3])}"
                    if len(unique_types) > 3:
                        zone['description'] += f" +{len(unique_types)-3}"
                
                zones.append(zone)
        
        # Сортируем по силе (чем больше уровней, тем выше приоритет)
        zones.sort(key=lambda x: (x['count'], x['strength']), reverse=True)
        
        return zones    

    def detect_compression(self, df: pd.DataFrame) -> Dict:
        """Обнаружение сжатия волатильности перед импульсом"""
        try:
            recent = df.tail(20)
            high_low_range = (recent['high'].max() - recent['low'].min()) / recent['close'].mean() * 100
            
            atr_pct = 0
            if 'atr' in recent.columns:
                atr = recent['atr'].mean()
                atr_pct = atr / recent['close'].mean() * 100
            
            if high_low_range < 5 and atr_pct < 2:
                return {
                    'compression': True,
                    'strength': 80,
                    'description': f"📉 Сжатие волатильности: диапазон {high_low_range:.1f}%, ATR {atr_pct:.1f}%"
                }
        except Exception as e:
            logger.debug(f"Ошибка detect_compression: {e}")
        return {'compression': False}
    
    def detect_volume_growth(self, df: pd.DataFrame) -> Dict:
        """Обнаружение роста объемов (подготовка к импульсу)"""
        try:
            if len(df) < 20:
                return {'volume_growth': False}
            
            avg_volume_old = df['volume'].tail(20).head(10).mean()
            avg_volume_new = df['volume'].tail(10).mean()
            volume_growth = avg_volume_new / avg_volume_old if avg_volume_old > 0 else 1
            
            threshold = self.settings.get('volume_growth_threshold', 1.3)
            if volume_growth >= threshold:
                return {
                    'volume_growth': True,
                    'growth_pct': round((volume_growth - 1) * 100, 1),
                    'description': f"📊 Рост объемов +{(volume_growth-1)*100:.0f}% перед импульсом"
                }
        except Exception as e:
            logger.debug(f"Ошибка detect_volume_growth: {e}")
        return {'volume_growth': False}

    def detect_compression_to_level(self, df: pd.DataFrame, current_price: float) -> Dict:
        """
        Обнаружение поджатия цены к уровню (границе диапазона)
        """
        try:
            recent = df.tail(20)
            range_high = recent['high'].max()
            range_low = recent['low'].min()
            
            # Расстояние до границ
            distance_to_high = (range_high - current_price) / current_price * 100
            distance_to_low = (current_price - range_low) / current_price * 100
            min_distance = min(distance_to_high, distance_to_low)
            
            # Проверяем, близко ли к границе
            threshold = self.settings.get('compression_distance_pct', 1.0)
            if min_distance > threshold:
                return {'compression': False}
            
            # Определяем направление поджатия
            if distance_to_high < distance_to_low:
                direction = 'LONG'
                level_price = range_high
                level_type = 'сопротивлению'
            else:
                direction = 'SHORT'
                level_price = range_low
                level_type = 'поддержке'
            
            # Проверяем рост объема
            avg_volume_old = df['volume'].tail(20).head(10).mean()
            avg_volume_new = df['volume'].tail(10).mean()
            volume_growth = avg_volume_new / avg_volume_old if avg_volume_old > 0 else 1
            volume_ok = volume_growth >= self.settings.get('compression_volume_threshold', 1.3)
            
            # Проверяем сжатие волатильности
            if 'atr' in df.columns:
                atr_pct = df['atr'].tail(10).mean() / current_price * 100
                atr_ok = atr_pct < self.settings.get('compression_atr_threshold', 1.5)
            else:
                atr_ok = True
            
            if volume_ok and atr_ok:
                return {
                    'compression': True,
                    'direction': direction,
                    'level_price': level_price,
                    'level_type': level_type,
                    'distance': min_distance,
                    'volume_growth': (volume_growth - 1) * 100,
                    'strength': 70,
                    'description': f"📊 ПОДЖАТИЕ К {level_type.upper()}: {level_price:.4f} (рост объемов +{(volume_growth-1)*100:.0f}%, ATR {atr_pct:.1f}%)"
                }
            
            return {'compression': False}
            
        except Exception as e:
            logger.debug(f"Ошибка detect_compression_to_level: {e}")
            return {'compression': False}
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Полный анализ накопления
        Накопление = наличие ХОТЯ БЫ ОДНОГО признака
        """
        result = {
            'has_accumulation': False,
            'signals': [],
            'strength': 0,
            'direction': None
        }
        
        # 1. A/D дивергенция
        ad_div = self.detect_ad_divergence(df)
        if ad_div.get('accumulation'):
            result['has_accumulation'] = True
            result['signals'].append(ad_div['description'])
            result['strength'] = max(result['strength'], ad_div.get('strength', 0))
            if ad_div.get('distribution'):
                result['direction'] = 'SHORT'
            else:
                result['direction'] = 'LONG'
            logger.info(f"  ✅ Накопление: A/D дивергенция (сила {result['strength']:.0f}%)")
        
        # 2. Аномальный объем в консолидации
        volume_spike = self.detect_volume_spikes_in_range(df)
        if volume_spike.get('accumulation'):
            result['has_accumulation'] = True
            result['signals'].append(volume_spike['description'])
            result['strength'] = max(result['strength'], volume_spike.get('strength', 0))
            logger.info(f"  ✅ Накопление: аномальный объем (сила {result['strength']:.0f}%)")
        
        # 3. Тихая аккумуляция
        silent = self.detect_silent_accumulation(df)
        if silent.get('accumulation'):
            result['has_accumulation'] = True
            result['signals'].append(silent['description'])
            result['strength'] = max(result['strength'], silent.get('strength', 0))
            logger.info(f"  ✅ Накопление: тихая аккумуляция (сила {result['strength']:.0f}%)")

        # ✅ НОВЫЕ ПРОВЕРКИ
        # Сжатие волатильности
        compression = self.detect_compression(df)
        if compression.get('compression'):
            result['has_accumulation'] = True
            result['signals'].append(compression['description'])
            result['strength'] = max(result['strength'], compression.get('strength', 70))
            logger.info(f"  ✅ Накопление: сжатие волатильности")
        
        # Рост объемов
        volume_growth = self.detect_volume_growth(df)
        if volume_growth.get('volume_growth'):
            result['has_accumulation'] = True
            result['signals'].append(volume_growth['description'])
            result['strength'] = max(result['strength'], min(100, volume_growth.get('growth_pct', 0) * 2))
            logger.info(f"  ✅ Накопление: рост объемов +{volume_growth.get('growth_pct', 0)}%")
        
        # Если есть накопление, но направление не определено — определяем по VWAP
        if result['has_accumulation'] and not result['direction']:
            if 'vwap' in df.columns:
                if df['close'].iloc[-1] > df['vwap'].iloc[-1]:
                    result['direction'] = 'LONG'
                else:
                    result['direction'] = 'SHORT'
        
        # Логируем результат
        if result['has_accumulation']:
            logger.info(f"  📦 НАКОПЛЕНИЕ ОБНАРУЖЕНО: {len(result['signals'])} признаков, сила {result['strength']:.0f}%")
        else:
            logger.info(f"  ⚠️ Накопление НЕ обнаружено (0 признаков)")
        
        return result

# ============== АНАЛИЗАТОР ТРЕНДОВЫХ ЛИНИЙ ==============

class TrendLineAnalyzer:
    """Анализ наклонных уровней поддержки/сопротивления"""
    
    def find_trend_lines(self, df: pd.DataFrame, touch_count: int = 3) -> List[Dict]:
        """
        Поиск наклонных уровней с несколькими касаниями
        """
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        trend_lines = []
        
        # Поиск нисходящей линии сопротивления (соединяем максимумы)
        for i in range(len(highs) - 20, len(highs) - 5):
            for j in range(i + 5, len(highs)):
                # Пробуем провести линию через две точки
                x1, y1 = i, highs[i]
                x2, y2 = j, highs[j]
                
                # Наклон должен быть отрицательным (нисходящий тренд)
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
                if slope >= 0:
                    continue
                
                # Считаем касания
                touches = 0
                touch_points = []
                
                for k in range(j, len(highs)):
                    # Расчет ожидаемого значения на линии
                    expected = y1 + slope * (k - i)
                    # Проверяем, касается ли свеча линии
                    if abs(highs[k] - expected) / expected < 0.003:  # допуск 0.3%
                        touches += 1
                        touch_points.append(k)
                
                if touches >= touch_count:
                    # Проверяем, пробита ли линия сейчас
                    last_price = closes[-1]
                    last_expected = y1 + slope * (len(highs)-1 - i)
                    is_broken = last_price > last_expected * 1.01  # пробой на 1%
                    
                    trend_lines.append({
                        'type': 'resistance',
                        'slope': slope,
                        'touches': touches,
                        'touch_points': touch_points,
                        'current_level': last_expected,
                        'is_broken': is_broken,
                        'strength': min(100, touches * 25)  # сила от количества касаний
                    })
        
        # Поиск восходящей линии поддержки (соединяем минимумы)
        for i in range(len(lows) - 20, len(lows) - 5):
            for j in range(i + 5, len(lows)):
                # Пробуем провести линию через две точки
                x1, y1 = i, lows[i]
                x2, y2 = j, lows[j]
                
                # Наклон должен быть положительным (восходящий тренд)
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
                if slope <= 0:
                    continue
                
                # Считаем касания
                touches = 0
                touch_points = []
                
                for k in range(j, len(lows)):
                    # Расчет ожидаемого значения на линии
                    expected = y1 + slope * (k - i)
                    # Проверяем, касается ли свеча линии
                    if abs(lows[k] - expected) / expected < 0.003:  # допуск 0.3%
                        touches += 1
                        touch_points.append(k)
                
                if touches >= touch_count:
                    # Проверяем, пробита ли линия сейчас
                    last_price = closes[-1]
                    last_expected = y1 + slope * (len(lows)-1 - i)
                    is_broken = last_price < last_expected * 0.99  # пробой на 1% вниз
                    
                    trend_lines.append({
                        'type': 'support',
                        'slope': slope,
                        'touches': touches,
                        'touch_points': touch_points,
                        'current_level': last_expected,
                        'is_broken': is_broken,
                        'strength': min(100, touches * 25)
                    })
        
        return trend_lines[-5:]  # последние 5 линий

        # Пробой наклонного уровня
    def check_approaching_trendline(self, df: pd.DataFrame, current_price: float, touch_count: int = 3, threshold: float = 0.5) -> List[Dict]:
        """
        Проверка приближения цены к трендовой линии (до пробоя)
        threshold: процент от цены, при котором считаем "приближение" (0.5% по умолчанию)
        """
        warnings = []
        
        # Находим все трендовые линии
        trend_lines = self.find_trend_lines(df, touch_count)
        
        for line in trend_lines:
            if line['is_broken']:
                continue  # уже пробита - не интересно
                
            current_level = line['current_level']
            
            # Для линии сопротивления (цена под линией)
            if line['type'] == 'resistance' and current_price < current_level:
                distance_to_line = ((current_level - current_price) / current_price) * 100
                
                # Если цена приблизилась к линии на threshold%
                if distance_to_line <= threshold:
                    warnings.append({
                        'type': 'resistance',
                        'level': current_level,
                        'distance': distance_to_line,
                        'touches': line['touches'],
                        'message': f"⚠️ Цена приближается к наклонному сопротивлению ({distance_to_line:.1f}% до пробоя)"
                    })
            
            # Для линии поддержки (цена над линией)
            elif line['type'] == 'support' and current_price > current_level:
                distance_to_line = ((current_price - current_level) / current_price) * 100
                
                if distance_to_line <= threshold:
                    warnings.append({
                        'type': 'support',
                        'level': current_level,
                        'distance': distance_to_line,
                        'touches': line['touches'],
                        'message': f"⚠️ Цена приближается к наклонной поддержке ({distance_to_line:.1f}% до пробоя)"
                    })
        
        return warnings    

# ============== ОТСЛЕЖИВАНИЕ ПРОБОЕВ ==============

class BreakoutTracker:
    """Отслеживание пробоев с комбинированным подтверждением"""
    
    def __init__(self):
        self.potential_breakouts = {}  # отслеживаем потенциальные пробои
        self.confirmed_breakouts = set()
    
    def check_breakout_confirmation(self, symbol: str, tf: str, df: pd.DataFrame, line: Dict, current_price: float, 
                                   required_candles: int = 3, required_percent: float = 0.5,
                                   volume_confirmation: float = 2.0, confirmation_mode: str = 'any_two') -> Optional[Dict]:
        """
        Комбинированная проверка закрепления пробоя
        - required_candles: сколько свечей нужно
        - required_percent: на сколько процентов нужно закрепиться
        - volume_confirmation: какой объем нужен для подтверждения (x от среднего)
        - confirmation_mode: 'any_two', 'all', 'any_one'
        """
        key = f"{symbol}_{tf}_{id(line)}"
        
        # Определяем направление пробоя
        if line['type'] == 'resistance':
            is_broken = current_price > line['current_level']
            confirmation_price = line['current_level'] * (1 + required_percent/100)
            direction = "вверх"
        else:
            is_broken = current_price < line['current_level']
            confirmation_price = line['current_level'] * (1 - required_percent/100)
            direction = "вниз"
        
        # Если пробоя нет
        if not is_broken:
            if key in self.potential_breakouts:
                del self.potential_breakouts[key]
            return None
        
        # Есть пробой
        if key not in self.potential_breakouts:
            # Первый раз видим пробой
            avg_volume = df['volume'].rolling(20).mean().iloc[-1] if len(df) > 20 else df['volume'].mean()
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            self.potential_breakouts[key] = {
                'line': line,
                'first_cross_time': datetime.now(),
                'first_cross_price': current_price,
                'direction': direction,
                'confirmations': {
                    'candles': 1,
                    'price': current_price >= confirmation_price if direction == 'вверх' else current_price <= confirmation_price,
                    'volume': volume_ratio >= volume_confirmation
                },
                'max_price': current_price,
                'min_price': current_price,
                'tf': tf,
                'volume_ratio': volume_ratio
            }
            return None
        
        # Уже отслеживаем этот пробой
        tracker = self.potential_breakouts[key]
        
        # Обновляем максимум/минимум
        tracker['max_price'] = max(tracker['max_price'], current_price)
        tracker['min_price'] = min(tracker['min_price'], current_price)
        
        # Обновляем подтверждения
        tracker['confirmations']['candles'] += 1
        
        if direction == 'вверх':
            tracker['confirmations']['price'] = current_price >= confirmation_price
        else:
            tracker['confirmations']['price'] = current_price <= confirmation_price
        
        # Считаем количество выполненных условий
        conditions_met = 0
        if tracker['confirmations']['candles'] >= required_candles:
            conditions_met += 1
        if tracker['confirmations']['price']:
            conditions_met += 1
        if tracker['confirmations']['volume']:
            conditions_met += 1
        
        # Проверяем по выбранному режиму
        should_confirm = False
        if confirmation_mode == 'all':
            should_confirm = conditions_met == 3
        elif confirmation_mode == 'any_two':
            should_confirm = conditions_met >= 2
        elif confirmation_mode == 'any_one':
            should_confirm = conditions_met >= 1
        
        if should_confirm:
            # Пробой подтвержден!
            self.confirmed_breakouts.add(key)
            
            # Рассчитываем размер движения
            if direction == 'вверх':
                move_percent = ((tracker['max_price'] - line['current_level']) / line['current_level']) * 100
            else:
                move_percent = ((line['current_level'] - tracker['min_price']) / line['current_level']) * 100
            
            result = {
                'line': line,
                'tf': tf,
                'direction': direction,
                'touches': line['touches'],
                'breakout_price': tracker['first_cross_price'],
                'current_price': current_price,
                'move_percent': move_percent,
                'confirmations': tracker['confirmations'],
                'message': (f"✅ ПРОБОЙ {direction} на {tf} ПОДТВЕРЖДЕН! "
                           f"(свечей: {tracker['confirmations']['candles']}/{required_candles}, "
                           f"закрепление: {'✅' if tracker['confirmations']['price'] else '❌'}, "
                           f"объем: x{tracker['volume_ratio']:.1f})")
            }
            
            del self.potential_breakouts[key]
            return result
        
        return None

# ============== ДЕТЕКТОР ЛОЖНЫХ ПРОБОЕВ  ==============

class FakeoutDetector:
    """Детектор ложных пробоев (fakeouts)"""
    
    def __init__(self):
        self.potential_fakeouts = {}  # отслеживаем подозрительные пробои
        self.confirmed_fakeouts = set()  # подтвержденные ложные пробои
    
    def check_fakeout(self, symbol: str, tf: str, df: pd.DataFrame, line: Dict, current_price: float,
                     breakout_distance: float = None,  # теперь берется из конфига
                     retrace_threshold: float = None,
                     confirmation_candles: int = None) -> Optional[Dict]:
        """
        Проверка на ложный пробой с настройками из конфига
        """
        from config import FAKEOUT_SETTINGS
        
        # Берем настройки из конфига, если не переданы явно
        if breakout_distance is None:
            breakout_distance = FAKEOUT_SETTINGS.get('breakout_distance', 2.0)
        if retrace_threshold is None:
            retrace_threshold = FAKEOUT_SETTINGS.get('retrace_threshold', 70) / 100
        if confirmation_candles is None:
            confirmation_candles = FAKEOUT_SETTINGS.get('confirmation_candles', 2)
        
        key = f"{symbol}_{tf}_{id(line)}"
        
        # Определяем тип линии
        if line['type'] == 'resistance':
            is_breakout = current_price > line['current_level']
            breakout_direction = "вверх"
        else:
            is_breakout = current_price < line['current_level']
            breakout_direction = "вниз"
        
        # Если пробоя нет - ничего не делаем
        if not is_breakout:
            if key in self.potential_fakeouts:
                del self.potential_fakeouts[key]
            return None
        
        # Есть пробой
        if key not in self.potential_fakeouts:
            breakout_price = current_price
            breakout_size = abs((current_price - line['current_level']) / line['current_level'] * 100)
            
            # Проверяем, достаточно ли большой пробой
            if breakout_size >= breakout_distance:
                self.potential_fakeouts[key] = {
                    'line': line,
                    'breakout_price': breakout_price,
                    'breakout_size': breakout_size,
                    'breakout_time': datetime.now(),
                    'max_price': current_price,
                    'min_price': current_price,
                    'direction': breakout_direction,
                    'candles_after': 0,
                    'tf': tf
                }
                logger.info(f"  🔍 Отслеживаю потенциальный пробой {breakout_direction} на {tf} ({breakout_size:.1f}%)")
            return None
        
        # Уже отслеживаем этот пробой
        tracker = self.potential_fakeouts[key]
        tracker['candles_after'] += 1
        
        # Обновляем максимум/минимум
        if current_price > tracker['max_price']:
            tracker['max_price'] = current_price
        if current_price < tracker['min_price']:
            tracker['min_price'] = current_price
        
        # Ждем нужное количество свечей для подтверждения
        if tracker['candles_after'] < confirmation_candles:
            return None
        
        # Анализируем, был ли это ложный пробой
        if line['type'] == 'resistance':
            max_price = tracker['max_price']
            current_retrace = ((max_price - current_price) / (max_price - line['current_level'])) * 100
            
            if current_retrace >= retrace_threshold * 100:
                fakeout = {
                    'type': 'fakeout',
                    'direction': 'вверх',
                    'line': line,
                    'breakout_price': tracker['breakout_price'],
                    'max_price': max_price,
                    'current_price': current_price,
                    'breakout_size': tracker['breakout_size'],
                    'retrace_percent': current_retrace,
                    'touches': line['touches'],
                    'tf': tf,
                    'message': (f"🚨 ЛОЖНЫЙ ПРОБОЙ {breakout_direction} на {tf}! "
                               f"Цена вернулась на {current_retrace:.0f}% от пробоя")
                }
                
                self.confirmed_fakeouts.add(key)
                del self.potential_fakeouts[key]
                return fakeout
        else:
            min_price = tracker['min_price']
            current_retrace = ((current_price - min_price) / (line['current_level'] - min_price)) * 100
            
            if current_retrace >= retrace_threshold * 100:
                fakeout = {
                    'type': 'fakeout',
                    'direction': 'вниз',
                    'line': line,
                    'breakout_price': tracker['breakout_price'],
                    'min_price': min_price,
                    'current_price': current_price,
                    'breakout_size': tracker['breakout_size'],
                    'retrace_percent': current_retrace,
                    'touches': line['touches'],
                    'tf': tf,
                    'message': (f"🚨 ЛОЖНЫЙ ПРОБОЙ {breakout_direction} на {tf}! "
                               f"Цена вернулась на {current_retrace:.0f}% от пробоя")
                }
                
                self.confirmed_fakeouts.add(key)
                del self.potential_fakeouts[key]
                return fakeout
        
        return None

# ============== Детектор выбива стопов Стоп-хaнт (Stop Hunt) ==============
class StopHuntDetector:
    """
    Детектор выбива стопов (Stop Hunt)
    Определяет, когда цена пробила уровень ликвидности и вернулась
    """
    
    def __init__(self, settings: Dict = None):
        self.settings = settings or STOP_HUNT_SETTINGS  # если STOP_HUNT_SETTINGS уже импортирован
        self.tracked_breakouts = {}  # отслеживаем потенциальные стоп-ханты
        
    def find_liquidity_levels(self, df: pd.DataFrame, lookback: int = 100) -> List[Dict]:
        """
        Поиск уровней ликвидности (где толпа держит стопы)
        - Локальные максимумы (выше которых стопы у шортистов)
        - Локальные минимумы (ниже которых стопы у лонгистов)
        """
        levels = []
        window = 5
        
        # Поиск локальных максимумов (сопротивление)
        for i in range(window, len(df) - window):
            if df['high'].iloc[i] == max(df['high'].iloc[i-window:i+window]):
                levels.append({
                    'type': 'resistance',
                    'price': df['high'].iloc[i],
                    'index': i,
                    'strength': self._calculate_level_strength(df, i, 'high')
                })
        
        # Поиск локальных минимумов (поддержка)
        for i in range(window, len(df) - window):
            if df['low'].iloc[i] == min(df['low'].iloc[i-window:i+window]):
                levels.append({
                    'type': 'support',
                    'price': df['low'].iloc[i],
                    'index': i,
                    'strength': self._calculate_level_strength(df, i, 'low')
                })
        
        # Сортируем по силе и берем топ-5
        levels.sort(key=lambda x: x['strength'], reverse=True)
        return levels[:5]
    
    def _calculate_level_strength(self, df: pd.DataFrame, idx: int, price_type: str) -> int:
        """Расчет силы уровня (количество касаний + объем)"""
        price = df[price_type].iloc[idx]
        touches = 0
        volume_sum = 0
        
        for i in range(max(0, idx - 100), min(len(df), idx + 100)):
            if i == idx:
                continue
            if price_type == 'high':
                if abs(df['high'].iloc[i] - price) / price < 0.003:
                    touches += 1
                    volume_sum += df['volume'].iloc[i]
            else:
                if abs(df['low'].iloc[i] - price) / price < 0.003:
                    touches += 1
                    volume_sum += df['volume'].iloc[i]
        
        # Сила = касания (до 50) + объем (до 50)
        strength = min(50, touches * 10) + min(50, (volume_sum / df['volume'].mean()) * 10 if df['volume'].mean() > 0 else 0)
        return min(100, int(strength))
    
    def detect_stop_hunt(self, symbol: str, tf: str, df: pd.DataFrame, 
                         current_price: float) -> Optional[Dict]:
        """
        Обнаружение выбива стопов
        Возвращает информацию о стоп-ханте, если обнаружен
        """
        key = f"{symbol}_{tf}"
        levels = self.find_liquidity_levels(df, self.settings['lookback_bars'])
        
        for level in levels:
            # Проверяем пробой уровня
            if level['type'] == 'resistance':
                is_break = current_price > level['price']
                breakout_direction = 'up'
            else:
                is_break = current_price < level['price']
                breakout_direction = 'down'
            
            if not is_break:
                # Если не пробой — сбрасываем отслеживание
                if key in self.tracked_breakouts:
                    del self.tracked_breakouts[key]
                continue
            
            # Есть пробой
            if key not in self.tracked_breakouts:
                # Первый раз видим пробой
                self.tracked_breakouts[key] = {
                    'level': level,
                    'break_price': current_price,
                    'break_time': datetime.now(),
                    'max_price': current_price,
                    'min_price': current_price,
                    'breakout_pct': abs((current_price - level['price']) / level['price'] * 100)
                }
                continue
            
            # Уже отслеживаем
            tracker = self.tracked_breakouts[key]
            
            # Обновляем экстремумы
            tracker['max_price'] = max(tracker['max_price'], current_price)
            tracker['min_price'] = min(tracker['min_price'], current_price)
            
            # Проверяем, прошел ли лимит времени
            time_elapsed = (datetime.now() - tracker['break_time']).total_seconds()
            if time_elapsed > self.settings['max_retrace_time']:
                # Слишком долго — не стоп-хант
                del self.tracked_breakouts[key]
                continue
            
            # Проверяем, достаточно ли большой был пробой
            if tracker['breakout_pct'] < self.settings['min_breakout_pct']:
                continue
            
            # Проверяем возврат
            if level['type'] == 'resistance':
                # Для сопротивления: цена должна вернуться ниже уровня
                retrace_pct = ((tracker['max_price'] - current_price) / 
                              (tracker['max_price'] - level['price']) * 100) if tracker['max_price'] > level['price'] else 0
                
                if current_price <= level['price'] and retrace_pct >= self.settings['retrace_threshold_pct']:
                    # Стоп-хант обнаружен!
                    result = {
                        'type': 'stop_hunt',
                        'direction': 'LONG' if level['type'] == 'resistance' else 'SHORT',
                        'level': level['price'],
                        'breakout_pct': tracker['breakout_pct'],
                        'retrace_pct': retrace_pct,
                        'strength': level['strength'],
                        'timeframe': tf,
                        'message': f"🎯 ВЫБИВ СТОПОВ на {tf}: пробой {level['price']:.4f} на {tracker['breakout_pct']:.1f}%, возврат на {retrace_pct:.0f}%"
                    }
                    del self.tracked_breakouts[key]
                    return result
                    
            else:  # support
                # Для поддержки: цена должна вернуться выше уровня
                retrace_pct = ((current_price - tracker['min_price']) / 
                              (level['price'] - tracker['min_price']) * 100) if level['price'] > tracker['min_price'] else 0
                
                if current_price >= level['price'] and retrace_pct >= self.settings['retrace_threshold_pct']:
                    result = {
                        'type': 'stop_hunt',
                        'direction': 'SHORT' if level['type'] == 'support' else 'LONG',
                        'level': level['price'],
                        'breakout_pct': tracker['breakout_pct'],
                        'retrace_pct': retrace_pct,
                        'strength': level['strength'],
                        'timeframe': tf,
                        'message': f"🎯 ВЫБИВ СТОПОВ на {tf}: пробой {level['price']:.4f} на {tracker['breakout_pct']:.1f}%, возврат на {retrace_pct:.0f}%"
                    }
                    del self.tracked_breakouts[key]
                    return result
        
        return None

# ============== Детектор зон ликвидности ==============
class LiquidityZoneDetector:
    """
    Детектор зон ликвидности
    Находит уровни, где вероятно скопление стоп-лоссов
    """
    
    def __init__(self, settings: Dict = None):
        from config import LIQUIDITY_ZONES_SETTINGS
        self.settings = settings or LIQUIDITY_ZONES_SETTINGS
    
    def find_liquidity_zones(self, df: pd.DataFrame, tf_name: str) -> List[Dict]:
        """
        Поиск зон ликвидности на одном таймфрейме
        Возвращает список зон с их силой
        """
        zones = []
        lookback = self.settings['lookback_bars']
        df_work = df.tail(lookback).copy()
        
        # Поиск локальных максимумов (зоны сопротивления)
        resistance_zones = self._find_swing_highs(df_work, tf_name)
        zones.extend(resistance_zones)
        
        # Поиск локальных минимумов (зоны поддержки)
        support_zones = self._find_swing_lows(df_work, tf_name)
        zones.extend(support_zones)
        
        # Сортируем по силе
        zones.sort(key=lambda x: x['strength'], reverse=True)
        
        return zones[:self.settings['max_zones']]
    
    def _find_swing_highs(self, df: pd.DataFrame, tf_name: str) -> List[Dict]:
        """Поиск локальных максимумов"""
        zones = []
        window = 5
        zone_width_pct = self.settings['zone_width_pct'] / 100
        
        for i in range(window, len(df) - window):
            # Проверяем, является ли свеча локальным максимумом
            is_swing_high = all(
                df['high'].iloc[i] > df['high'].iloc[j] 
                for j in range(i - window, i + window + 1) if j != i
            )
            
            if not is_swing_high:
                continue
            
            price = df['high'].iloc[i]
            
            # Считаем касания этого уровня
            touches = 0
            volume_sum = 0
            
            for k in range(max(0, i - 100), min(len(df), i + 100)):
                if k == i:
                    continue
                if abs(df['high'].iloc[k] - price) / price < zone_width_pct:
                    touches += 1
                    volume_sum += df['volume'].iloc[k]
            
            if touches < self.settings['min_touches']:
                continue
            
            # Рассчитываем силу зоны
            strength = self._calculate_zone_strength(touches, volume_sum, df, tf_name)
            
            zones.append({
                'type': 'resistance',
                'price': price,
                'price_low': price * (1 - zone_width_pct),
                'price_high': price * (1 + zone_width_pct),
                'touches': touches,
                'strength': strength,
                'timeframe': tf_name,
                'volume_sum': volume_sum
            })
        
        return zones
    
    def _find_swing_lows(self, df: pd.DataFrame, tf_name: str) -> List[Dict]:
        """Поиск локальных минимумов"""
        zones = []
        window = 5
        zone_width_pct = self.settings['zone_width_pct'] / 100
        
        for i in range(window, len(df) - window):
            # Проверяем, является ли свеча локальным минимумом
            is_swing_low = all(
                df['low'].iloc[i] < df['low'].iloc[j] 
                for j in range(i - window, i + window + 1) if j != i
            )
            
            if not is_swing_low:
                continue
            
            price = df['low'].iloc[i]
            
            # Считаем касания этого уровня
            touches = 0
            volume_sum = 0
            
            for k in range(max(0, i - 100), min(len(df), i + 100)):
                if k == i:
                    continue
                if abs(df['low'].iloc[k] - price) / price < zone_width_pct:
                    touches += 1
                    volume_sum += df['volume'].iloc[k]
            
            if touches < self.settings['min_touches']:
                continue
            
            # Рассчитываем силу зоны
            strength = self._calculate_zone_strength(touches, volume_sum, df, tf_name)
            
            zones.append({
                'type': 'support',
                'price': price,
                'price_low': price * (1 - zone_width_pct),
                'price_high': price * (1 + zone_width_pct),
                'touches': touches,
                'strength': strength,
                'timeframe': tf_name,
                'volume_sum': volume_sum
            })
        
        return zones
    
    def _calculate_zone_strength(self, touches: int, volume_sum: float, 
                                  df: pd.DataFrame, tf_name: str) -> int:
        """Расчет силы зоны"""
        weights = self.settings['strength_weights']
        
        # Базовый вес от касаний
        touch_strength = min(100, touches * weights.get('touches', 15))
        
        # Вес от объема
        avg_volume = df['volume'].mean()
        volume_ratio = volume_sum / avg_volume if avg_volume > 0 else 1
        volume_strength = min(100, volume_ratio * weights.get('volume', 10))
        
        # Вес от таймфрейма
        tf_weights = {
            '1m': 1,
            '5m': 2,
            '15m': 3,
            '30m': 4,
            '1h': 5,
            '4h': 7,
            '1d': 10,
        }
        tf_strength = tf_weights.get(tf_name, 3) * weights.get('timeframe', 20) / 10
        
        # Итоговая сила
        strength = (touch_strength + volume_strength + tf_strength) / 3
        
        return min(100, int(strength))
    
    def analyze_multi_timeframe(self, dataframes: Dict[str, pd.DataFrame]) -> Dict:
        """
        Анализ зон ликвидности на всех таймфреймах
        """
        result = {
            'has_zones': False,
            'zones': [],
            'signals': [],
            'strength': 0
        }
        
        tfs = self.settings.get('timeframes', ['15m', '1h', '4h', '1d'])
        tf_map = {
            '15m': 'current',
            '1h': 'hourly',
            '4h': 'four_hourly',
            '1d': 'daily'
        }
        
        for tf_display in tfs:
            tf_key = tf_map.get(tf_display, tf_display)
            if tf_key not in dataframes or dataframes[tf_key] is None:
                continue
            
            df = dataframes[tf_key]
            zones = self.find_liquidity_zones(df, tf_display)
            
            for zone in zones:
                result['has_zones'] = True
                result['zones'].append(zone)
                result['strength'] += zone['strength'] / len(zones) if zones else 0
                
                # Формируем сигнал
                zone_type = "сопротивление" if zone['type'] == 'resistance' else "поддержка"
                signal_text = (f"📍 Зона ликвидности ({zone_type}) на {zone['timeframe']}: "
                             f"{zone['price']:.4f} (сила {zone['strength']}%, {zone['touches']} касаний)")
                result['signals'].append(signal_text)
        
        if result['strength'] > 100:
            result['strength'] = 100
        
        return result
    
    def check_price_near_zone(self, current_price: float, zones: List[Dict], 
                               distance_threshold: float = 0.5) -> Optional[Dict]:
        """Проверка, находится ли цена рядом с зоной ликвидности"""
        for zone in zones:
            if zone['type'] == 'resistance':
                distance = ((zone['price'] - current_price) / current_price) * 100
                if 0 < distance <= distance_threshold:
                    return {
                        'zone': zone,
                        'distance': distance,
                        'type': 'resistance',
                        'action': 'breakout_or_rejection'
                    }
            else:  # support
                distance = ((current_price - zone['price']) / current_price) * 100
                if 0 < distance <= distance_threshold:
                    return {
                        'zone': zone,
                        'distance': distance,
                        'type': 'support',
                        'action': 'breakdown_or_bounce'
                    }
        return None

class PatternAnalyzer:
    """Анализ графических паттернов"""
    
    def __init__(self, settings: Dict = None):
        from config import PATTERN_SETTINGS
        self.settings = settings or PATTERN_SETTINGS
        self.tolerance = self.settings.get('double_top_bottom', {}).get('max_price_diff_pct', 1.0) / 100
    
    def _apply_aging(self, result: Dict, age_bars: int, df: pd.DataFrame = None, 
                    pattern_type: str = None, idx1: int = None, idx2: int = None, 
                    tf_name: str = None) -> Dict:
        """
        Применяет мягкую инвалидацию к паттерну
        """
        if not result.get('has_pattern'):
            return result
        
        # Получаем настройки (могут быть числом или словарём)
        max_age_config = self.settings.get('max_age_bars', 50)
        reduce_after_config = self.settings.get('reduce_strength_after', 25)
        reduce_factor = self.settings.get('reduce_factor', 0.5)
        
        # Определяем значения для текущего ТФ
        if isinstance(max_age_config, dict) and tf_name:
            max_age = max_age_config.get(tf_name, 50)
        else:
            max_age = max_age_config
        
        if isinstance(reduce_after_config, dict) and tf_name:
            reduce_after = reduce_after_config.get(tf_name, 25)
        else:
            reduce_after = reduce_after_config
        
        original_strength = result.get('original_strength', result['strength'])
        
        # 1. Проверка возраста
        if age_bars > max_age:
            # Слишком старый — не показываем
            return {'has_pattern': False}
        elif age_bars > reduce_after:
            # Снижаем силу
            result['strength'] = int(original_strength * reduce_factor)
            result['description'] += f" ⏳ (устаревает, {age_bars} св)"
        elif age_bars > reduce_after // 2:
            result['strength'] = int(original_strength * 0.75)
            result['description'] += f" ⏳ (стареет, {age_bars} св)"
        
        # 2. Проверка на ложный пробой (для двойной вершины/дна)
        if pattern_type in ['double_top', 'double_bottom'] and df is not None and idx1 is not None and idx2 is not None:
            if pattern_type == 'double_top':
                neckline = min(df['low'].iloc[idx1:idx2])
                recent_highs = df['high'].tail(5)
                if any(h > neckline * 1.01 for h in recent_highs):
                    result['strength'] = int(result['strength'] * 0.7)
                    result['description'] += " ⚠️ (ложный?)"
            else:
                neckline = max(df['high'].iloc[idx1:idx2])
                recent_lows = df['low'].tail(5)
                if any(l < neckline * 0.99 for l in recent_lows):
                    result['strength'] = int(result['strength'] * 0.7)
                    result['description'] += " ⚠️ (ложный?)"
        
        # 3. Сохраняем возраст
        result['age_bars'] = age_bars
        result['original_strength'] = original_strength
        
        return result

    def find_double_top_bottom(self, df: pd.DataFrame, tf_name: str) -> Dict:
        """
        Поиск двойной вершины или двойного дна
        """
        result = {'has_pattern': False, 'type': None, 'direction': None, 
                  'level': 0, 'strength': 0, 'description': ''}
        
        cfg = self.settings.get('double_top_bottom', {})
        if not cfg.get('enabled', True):
            return result
        
        min_distance = cfg.get('min_distance_bars', 5)
        min_drop_pct = cfg.get('min_drop_pct', 2.0) / 100
        
        # Поиск двойной вершины
        highs = df['high'].values
        for i in range(min_distance, len(highs) - min_distance):
            # Первая вершина
            if not self._is_swing_high(df, i):
                continue
            
            # Ищем вторую вершину на том же уровне
            for j in range(i + min_distance, len(highs) - min_distance):
                if not self._is_swing_high(df, j):
                    continue
                
                # Проверяем, что цены близки
                price_diff = abs(highs[i] - highs[j]) / highs[i]
                if price_diff > self.tolerance:
                    continue
                
                # Проверяем, что между ними было падение
                mid_low = min(df['low'].iloc[i:j])
                drop_pct = (highs[i] - mid_low) / highs[i]
                if drop_pct < min_drop_pct:
                    continue
                
                # Нашли двойную вершину!
                result['has_pattern'] = True
                result['type'] = 'double_top'
                result['direction'] = 'SHORT'
                result['level'] = highs[i]
                result['strength'] = cfg.get('strength', 70)
                result['description'] = f"🔻 ДВОЙНАЯ ВЕРШИНА на {tf_name}: {highs[i]:.4f}"
                # Рассчитываем возраст паттерна
                age_bars = len(df) - j  # j - индекс второй вершины
                result = self._apply_aging(result, age_bars, df, result['type'], i, j, tf_name)
                return result
        
        # Поиск двойного дна
        lows = df['low'].values
        for i in range(min_distance, len(lows) - min_distance):
            if not self._is_swing_low(df, i):
                continue
            
            for j in range(i + min_distance, len(lows) - min_distance):
                if not self._is_swing_low(df, j):
                    continue
                
                price_diff = abs(lows[i] - lows[j]) / lows[i]
                if price_diff > self.tolerance:
                    continue
                
                mid_high = max(df['high'].iloc[i:j])
                rise_pct = (mid_high - lows[i]) / lows[i]
                if rise_pct < min_drop_pct:
                    continue
                
                result['has_pattern'] = True
                result['type'] = 'double_bottom'
                result['direction'] = 'LONG'
                result['level'] = lows[i]
                result['strength'] = cfg.get('strength', 70)
                result['description'] = f"🟢 ДВОЙНОЕ ДНО на {tf_name}: {lows[i]:.4f}"
                # Рассчитываем возраст паттерна
                age_bars = len(df) - j  # j - индекс второго дна
                result = self._apply_aging(result, age_bars, df, result['type'], i, j, tf_name)
                return result
        
        return result
    
    def _is_swing_high(self, df: pd.DataFrame, idx: int, window: int = 3) -> bool:
        """Проверка, является ли свеча локальным максимумом"""
        if idx < window or idx >= len(df) - window:
            return False
        high = df['high'].iloc[idx]
        return all(high > df['high'].iloc[idx - i] for i in range(1, window + 1)) and \
               all(high > df['high'].iloc[idx + i] for i in range(1, window + 1))
    
    def _is_swing_low(self, df: pd.DataFrame, idx: int, window: int = 3) -> bool:
        """Проверка, является ли свеча локальным минимумом"""
        if idx < window or idx >= len(df) - window:
            return False
        low = df['low'].iloc[idx]
        return all(low < df['low'].iloc[idx - i] for i in range(1, window + 1)) and \
               all(low < df['low'].iloc[idx + i] for i in range(1, window + 1))
    
    def analyze_multi_timeframe(self, dataframes: Dict[str, pd.DataFrame]) -> Dict:
        """Анализ паттернов на всех таймфреймах"""
        result = {'has_pattern': False, 'patterns': [], 'strength': 0, 'direction': None}
        
        timeframes = self.settings.get('timeframes', ['current', 'hourly'])
        
        # Получаем настройки возраста (могут быть числом или словарём)
        max_age_config = self.settings.get('max_age_bars', 50)
        reduce_after_config = self.settings.get('reduce_strength_after', 25)
        
        tf_short = {
            'current': '15м',
            'hourly': '1ч',
            'four_hourly': '4ч',
            'daily': '1д',
            '5m': '5м',
            '3m': '3м',
        }
        
        for tf_name in timeframes:
            if tf_name not in dataframes or dataframes[tf_name] is None:
                continue
            
            df = dataframes[tf_name]
            
            # Определяем max_age для этого ТФ
            if isinstance(max_age_config, dict):
                max_age = max_age_config.get(tf_name, 20)
            else:
                max_age = max_age_config
            
            # Двойная вершина/дно
            pattern = self.find_double_top_bottom(df, tf_short.get(tf_name, tf_name))
            if pattern.get('has_pattern') and pattern.get('age_bars', 0) <= max_age:
                result['has_pattern'] = True
                result['patterns'].append(pattern)
                result['strength'] = max(result['strength'], pattern['strength'])
                result['direction'] = pattern['direction']
            
            # Флаг
            pattern = self.find_flag(df, tf_short.get(tf_name, tf_name))
            if pattern.get('has_pattern') and pattern.get('age_bars', 0) <= max_age:
                result['has_pattern'] = True
                result['patterns'].append(pattern)
                result['strength'] = max(result['strength'], pattern['strength'])
                result['direction'] = pattern['direction']
            
            # Клин
            pattern = self.find_wedge(df, tf_short.get(tf_name, tf_name))
            if pattern.get('has_pattern') and pattern.get('age_bars', 0) <= max_age:
                result['has_pattern'] = True
                result['patterns'].append(pattern)
                result['strength'] = max(result['strength'], pattern['strength'])
                result['direction'] = pattern['direction']
            
            # Голова и плечи
            pattern = self.find_head_shoulders(df, tf_short.get(tf_name, tf_name))
            if pattern.get('has_pattern') and pattern.get('age_bars', 0) <= max_age:
                result['has_pattern'] = True
                result['patterns'].append(pattern)
                result['strength'] = max(result['strength'], pattern['strength'])
                result['direction'] = pattern['direction']
        
        return result

    def find_flag(self, df: pd.DataFrame, tf_name: str) -> Dict:
        """
        Поиск паттерна Флаг (Bull/Bear Flag)
        """
        result = {'has_pattern': False, 'type': None, 'direction': None,
                'entry_price': 0, 'pole_pct': 0, 'strength': 0, 'description': ''}
        
        cfg = self.settings.get('flag', {})
        if not cfg.get('enabled', True):
            return result
        
        min_pole_pct = cfg.get('min_pole_pct', 3.0) / 100
        min_cons_bars = cfg.get('min_consolidation_bars', 5)
        max_cons_bars = cfg.get('max_consolidation_bars', 15)
        max_slope_pct = cfg.get('max_slope_pct', 0.5) / 100
        
        # Поиск бычьего флага (LONG)
        # 1. Ищем резкое движение вверх (древко)
        for i in range(10, len(df) - max_cons_bars):
            # Древко (Импульс): рост за 3-5 свечей
            pole_start = i - 5
            pole_end = i
            pole_low = min(df['low'].iloc[pole_start:pole_end])
            pole_high = max(df['high'].iloc[pole_start:pole_end])
            pole_change = (pole_high - pole_low) / pole_low
            
            if pole_change < min_pole_pct:
                continue
            
            # 2. Ищем консолидацию после древка (тело флага)
            for j in range(i + min_cons_bars, min(i + max_cons_bars, len(df))):
                cons_highs = df['high'].iloc[i:j]
                cons_lows = df['low'].iloc[i:j]
                
                # Проверяем наклон флага (должен быть против тренда)
                # Для бычьего флага — наклон вниз (нисходящая консолидация)
                slope = (cons_highs.iloc[-1] - cons_highs.iloc[0]) / cons_highs.iloc[0] / (j - i)
                
                if slope > max_slope_pct:  # наклон вверх — не подходит
                    continue
                
                # 3. Проверяем пробой вверх
                current_price = df['close'].iloc[-1]
                breakout_level = cons_highs.max()
                
                if current_price > breakout_level:
                    avg_volume = df['volume'].iloc[i:j].mean()
                    last_volume = df['volume'].iloc[-1]
                    volume_ratio = last_volume / avg_volume if avg_volume > 0 else 1
                    
                    if volume_ratio < cfg.get('volume_confirmation', 1.3):
                        continue
                    
                    result['has_pattern'] = True
                    result['type'] = 'bull_flag'
                    result['direction'] = 'LONG'
                    result['entry_price'] = breakout_level
                    result['pole_pct'] = pole_change * 100
                    result['strength'] = cfg.get('strength', 65)
                    result['description'] = f"🚩 БЫЧИЙ ФЛАГ на {tf_name}: импульс +{pole_change*100:.1f}%, консолидация {j-i} свечей"
                    
                    # Рассчитываем возраст паттерна
                    age_bars = len(df) - j  # j - индекс пробоя
                    result = self._apply_aging(result, age_bars, tf_name=tf_name)
                    return result
        
        # Поиск медвежьего флага (SHORT)
        for i in range(10, len(df) - max_cons_bars):
            # Древко: падение за 3-5 свечей
            pole_start = i - 5
            pole_end = i
            pole_high = max(df['high'].iloc[pole_start:pole_end])
            pole_low = min(df['low'].iloc[pole_start:pole_end])
            pole_change = (pole_high - pole_low) / pole_high
            
            if pole_change < min_pole_pct:
                continue
            
            # Консолидация
            for j in range(i + min_cons_bars, min(i + max_cons_bars, len(df))):
                cons_highs = df['high'].iloc[i:j]
                cons_lows = df['low'].iloc[i:j]
                
                # Для медвежьего флага — наклон вверх
                slope = (cons_lows.iloc[-1] - cons_lows.iloc[0]) / cons_lows.iloc[0] / (j - i)
                
                if slope < -max_slope_pct:  # наклон вниз — не подходит
                    continue
                
                # Проверяем пробой вниз
                current_price = df['close'].iloc[-1]
                breakout_level = cons_lows.min()
                
                if current_price < breakout_level:
                    avg_volume = df['volume'].iloc[i:j].mean()
                    last_volume = df['volume'].iloc[-1]
                    volume_ratio = last_volume / avg_volume if avg_volume > 0 else 1
                    
                    if volume_ratio < cfg.get('volume_confirmation', 1.3):
                        continue
                    
                    result['has_pattern'] = True
                    result['type'] = 'bear_flag'
                    result['direction'] = 'SHORT'
                    result['entry_price'] = breakout_level
                    result['pole_pct'] = pole_change * 100
                    result['strength'] = cfg.get('strength', 65)
                    result['description'] = f"🚩 МЕДВЕЖИЙ ФЛАГ на {tf_name}: импульс -{pole_change*100:.1f}%, консолидация {j-i} свечей"
                    
                    # Рассчитываем возраст паттерна
                    age_bars = len(df) - j  # j - индекс пробоя
                    result['age_bars'] = age_bars
                    result = self._apply_aging(result, age_bars, tf_name=tf_name)
                    return result
        
        return result

    def find_wedge(self, df: pd.DataFrame, tf_name: str) -> Dict:
        """
        Поиск паттерна Клин (Rising/Falling Wedge)
        """
        result = {'has_pattern': False, 'type': None, 'direction': None,
                'entry_price': 0, 'narrowing_pct': 0, 'strength': 0, 'description': ''}
        
        cfg = self.settings.get('wedge', {})
        if not cfg.get('enabled', True):
            return result
        
        min_bars = cfg.get('min_bars', 10)
        min_narrowing_pct = cfg.get('min_narrowing_pct', 30.0) / 100
        
        # Скользящее окно для поиска клина
        for start in range(0, len(df) - min_bars):
            end = min(start + min_bars + 10, len(df))
            window = df.iloc[start:end]
            
            if len(window) < min_bars:
                continue
            
            # Находим локальные максимумы и минимумы в окне
            highs = []
            lows = []
            
            for i in range(len(window)):
                if self._is_swing_high(window, i):
                    highs.append((i, window['high'].iloc[i]))
                if self._is_swing_low(window, i):
                    lows.append((i, window['low'].iloc[i]))
            
            if len(highs) < 3 or len(lows) < 3:
                continue
            
            # Линия тренда по максимумам
            high_slope, high_start = self._calc_trendline(highs)
            # Линия тренда по минимумам
            low_slope, low_start = self._calc_trendline(lows)
            
            if high_slope is None or low_slope is None:
                continue
            
            # Проверяем, сходятся ли линии
            slope_diff = abs(high_slope - low_slope)
            max_slope_diff = cfg.get('max_slope_diff_pct', 0.3) / 100
            
            if slope_diff > max_slope_diff:
                continue
            
            # Проверяем сужение диапазона
            start_width = (high_start - low_start) / low_start
            end_width = self._get_current_width(window, high_slope, low_slope)
            narrowing_pct = (start_width - end_width) / start_width
            
            if narrowing_pct < min_narrowing_pct:
                continue
            
            # Определяем тип клина
            current_price = df['close'].iloc[-1]
            entry_price = None
            direction = None
            wedge_type = None
            
            # Восходящий клин (линии вверх) → пробой вниз
            if high_slope > 0 and low_slope > 0:
                wedge_type = 'rising_wedge'
                direction = 'SHORT'
                entry_price = self._get_lower_line_value(highs, lows, len(window))
                
                if current_price < entry_price:  # пробой вниз
                    result['has_pattern'] = True
                    result['type'] = 'rising_wedge'
                    result['direction'] = 'SHORT'
                    result['narrowing_pct'] = narrowing_pct * 100
            
            # Нисходящий клин (линии вниз) → пробой вверх
            elif high_slope < 0 and low_slope < 0:
                wedge_type = 'falling_wedge'
                direction = 'LONG'
                entry_price = self._get_upper_line_value(highs, lows, len(window))
                
                if current_price > entry_price:  # пробой вверх
                    result['has_pattern'] = True
                    result['type'] = 'falling_wedge'
                    result['direction'] = 'LONG'
                    result['narrowing_pct'] = narrowing_pct * 100
            
            if result['has_pattern']:
                result['entry_price'] = entry_price
                result['strength'] = cfg.get('strength', 75)
                wedge_name = "ВОСХОДЯЩИЙ" if wedge_type == 'rising_wedge' else "НИСХОДЯЩИЙ"
                result['description'] = f"📐 {wedge_name} КЛИН на {tf_name}: сужение {narrowing_pct*100:.0f}%, пробой {direction}"
                
                # Рассчитываем возраст паттерна
                age_bars = len(df) - end  # используйте нужный индекс
                result['age_bars'] = age_bars
                result = self._apply_aging(result, age_bars, tf_name=tf_name)
                return result
    
        return result

    def _calc_trendline(self, points: List[Tuple[int, float]]) -> Tuple[float, float]:
        """Расчёт линии тренда (наклон и начальное значение)"""
        if len(points) < 2:
            return None, None
        
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return None, None
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        
        return slope, intercept

    def _get_current_width(self, window: pd.DataFrame, high_slope: float, low_slope: float) -> float:
        """Текущая ширина клина"""
        last_idx = len(window) - 1
        high_value = window['high'].iloc[-1] if len(window) > 0 else 0
        low_value = window['low'].iloc[-1] if len(window) > 0 else 0
        return (high_value - low_value) / low_value

    def _get_lower_line_value(self, highs: List, lows: List, current_idx: int) -> float:
        """Значение нижней линии тренда"""
        if len(lows) < 2:
            return 0
        slope, intercept = self._calc_trendline(lows)
        if slope is None:
            return 0
        return slope * current_idx + intercept

    def _get_upper_line_value(self, highs: List, lows: List, current_idx: int) -> float:
        """Значение верхней линии тренда"""
        if len(highs) < 2:
            return 0
        slope, intercept = self._calc_trendline(highs)
        if slope is None:
            return 0
        return slope * current_idx + intercept

    def find_head_shoulders(self, df: pd.DataFrame, tf_name: str) -> Dict:
        """
        Поиск паттерна Голова и плечи (Head and Shoulders)
        """
        result = {'has_pattern': False, 'type': None, 'direction': None,
                'neckline': 0, 'head_price': 0, 'strength': 0, 'description': ''}
        
        cfg = self.settings.get('head_shoulders', {})
        if not cfg.get('enabled', True):
            return result
        
        min_distance = cfg.get('min_shoulder_distance', 5)
        max_price_diff = cfg.get('max_price_diff_pct', 1.5) / 100
        head_mult = cfg.get('head_multiplier', 1.02)
        head_mult_bottom = cfg.get('head_multiplier_bottom', 0.98)
        min_neck_touches = cfg.get('min_neck_touches', 2)
        
        # Поиск всех локальных экстремумов
        highs = []
        lows = []
        
        for i in range(min_distance, len(df) - min_distance):
            if self._is_swing_high(df, i, window=3):
                highs.append({'idx': i, 'price': df['high'].iloc[i]})
            if self._is_swing_low(df, i, window=3):
                lows.append({'idx': i, 'price': df['low'].iloc[i]})
        
        if len(highs) < 3:
            return result
        
        # Поиск головы и плечей (SHORT)
        for i in range(1, len(highs) - 1):
            left_shoulder = highs[i-1]
            head = highs[i]
            right_shoulder = highs[i+1]
            
            # Проверяем расстояние между точками
            if (head['idx'] - left_shoulder['idx'] < min_distance or
                right_shoulder['idx'] - head['idx'] < min_distance):
                continue
            
            # Голова должна быть выше плеч
            if not (head['price'] > left_shoulder['price'] * head_mult and
                    head['price'] > right_shoulder['price'] * head_mult):
                continue
            
            # Плечи должны быть на одном уровне
            if abs(left_shoulder['price'] - right_shoulder['price']) / left_shoulder['price'] > max_price_diff:
                continue
            
            # Находим линию шеи (минимумы между плечами и головой)
            neckline_points = []
            
            # Минимум между левым плечом и головой
            left_neck = min(df['low'].iloc[left_shoulder['idx']:head['idx']])
            neckline_points.append(left_neck)
            
            # Минимум между головой и правым плечом
            right_neck = min(df['low'].iloc[head['idx']:right_shoulder['idx']])
            neckline_points.append(right_neck)
            
            # Усредняем линию шеи
            neckline = (left_neck + right_neck) / 2
            
            # Проверяем, сколько раз цена касалась линии шеи
            neck_touches = 0
            for j in range(left_shoulder['idx'], right_shoulder['idx']):
                if abs(df['low'].iloc[j] - neckline) / neckline < 0.005:  # 0.5% допуск
                    neck_touches += 1
            
            if neck_touches < min_neck_touches:
                continue
            
            # Проверяем пробой линии шеи
            current_price = df['close'].iloc[-1]
            
            if current_price < neckline:  # пробой вниз
                result['has_pattern'] = True
                result['type'] = 'head_shoulders'
                result['direction'] = 'SHORT'
                result['neckline'] = neckline
                result['head_price'] = head['price']
                result['strength'] = cfg.get('strength', 85)
                result['description'] = f"🧠 ГОЛОВА И ПЛЕЧИ на {tf_name}: шея {neckline:.4f}, цель {head['price'] - (head['price'] - neckline):.4f}"
                
                # Рассчитываем возраст паттерна
                age_bars = len(df) - right_shoulder['idx']
                result['age_bars'] = age_bars
                result = self._apply_aging(result, age_bars, tf_name=tf_name)
                return result
        
        # Поиск перевернутой головы и плечей (LONG)
        if len(lows) < 3:
            return result
        
        for i in range(1, len(lows) - 1):
            left_shoulder = lows[i-1]
            head = lows[i]
            right_shoulder = lows[i+1]
            
            if (head['idx'] - left_shoulder['idx'] < min_distance or
                right_shoulder['idx'] - head['idx'] < min_distance):
                continue
            
            # Голова должна быть ниже плеч
            if not (head['price'] < left_shoulder['price'] * head_mult_bottom and
                    head['price'] < right_shoulder['price'] * head_mult_bottom):
                continue
            
            # Плечи должны быть на одном уровне
            if abs(left_shoulder['price'] - right_shoulder['price']) / left_shoulder['price'] > max_price_diff:
                continue
            
            # Находим линию шеи (максимумы между плечами и головой)
            neckline_points = []
            
            left_neck = max(df['high'].iloc[left_shoulder['idx']:head['idx']])
            neckline_points.append(left_neck)
            
            right_neck = max(df['high'].iloc[head['idx']:right_shoulder['idx']])
            neckline_points.append(right_neck)
            
            neckline = (left_neck + right_neck) / 2
            
            neck_touches = 0
            for j in range(left_shoulder['idx'], right_shoulder['idx']):
                if abs(df['high'].iloc[j] - neckline) / neckline < 0.005:
                    neck_touches += 1
            
            if neck_touches < min_neck_touches:
                continue
            
            current_price = df['close'].iloc[-1]
            
            if current_price > neckline:  # пробой вверх
                result['has_pattern'] = True
                result['type'] = 'inverse_head_shoulders'
                result['direction'] = 'LONG'
                result['neckline'] = neckline
                result['head_price'] = head['price']
                result['strength'] = cfg.get('strength', 85)
                result['description'] = f"🧠 ПЕРЕВЕРНУТАЯ ГОЛОВА И ПЛЕЧИ на {tf_name}: шея {neckline:.4f}, цель {neckline + (neckline - head['price']):.4f}"
                
                # Рассчитываем возраст паттерна
                age_bars = len(df) - right_shoulder['idx']  # индекс правого плеча
                result['age_bars'] = age_bars
                result = self._apply_aging(result, age_bars, tf_name=tf_name)
                return result
        
        return result

# ============== ГЕНЕРАТОР ГРАФИКОВ ==============

class ChartGenerator:
    """Генератор графиков для сигналов"""
    
    def __init__(self):
        self.figsize = (12, 6)
        self.dpi = 100
        self.style = 'dark_background'
        
    def create_chart(self, df: pd.DataFrame, signal: Dict, coin: str, timeframe: str = '15m') -> BytesIO:
        """Создание графика с ценой, индикаторами и целями"""

        import matplotlib.font_manager as fm
        # Подавляем предупреждения о шрифтах
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

        plt.style.use(self.style)
        
        plot_df = df.tail(100).copy()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, 
                                    gridspec_kw={'height_ratios': [3, 1]})
        
        # ===== ВЕРХНИЙ ГРАФИК =====
        ax1.plot(plot_df.index, plot_df['close'], 
                color='white', linewidth=2, label='Цена')
        
        # EMA линии - ВСЕ ДОЛЖНЫ БЫТЬ В ЛЕГЕНДЕ
        if 'ema_9' in plot_df.columns:
            ax1.plot(plot_df.index, plot_df['ema_9'], 
                    color='#00ff88', linewidth=1.5, alpha=0.7, label='EMA 9')
        if 'ema_21' in plot_df.columns:
            ax1.plot(plot_df.index, plot_df['ema_21'], 
                    color='#ff8800', linewidth=1.5, alpha=0.7, label='EMA 21')
        if 'ema_50' in plot_df.columns:
            ax1.plot(plot_df.index, plot_df['ema_50'], 
                    color='#8888ff', linewidth=1, alpha=0.5, label='EMA 50')
        if 'ema_200' in plot_df.columns:
            ax1.plot(plot_df.index, plot_df['ema_200'], 
                    color='#ff4444', linewidth=1, alpha=0.5, label='EMA 200')
        
        # Bollinger Bands - ДОЛЖНЫ БЫТЬ В ЛЕГЕНДЕ
        if 'BBL_20_2.0' in plot_df.columns and 'BBU_20_2.0' in plot_df.columns:
            ax1.fill_between(plot_df.index, 
                            plot_df['BBL_20_2.0'], 
                            plot_df['BBU_20_2.0'],
                            alpha=0.2, color='gray', label='Bollinger Bands')
        
        # Текущая цена
        current_price = signal['price']
        ax1.axhline(y=current_price, color='#ffff00', # ← ЖЕЛТЫЙ
                linestyle='--', linewidth=2.0, alpha=0.9, # потолще и ярче
                label=f'Текущая: {current_price:.4f}')
        
        # Цели
        if signal.get('target_1'):
            ax1.axhline(y=signal['target_1'], color='#ffaa00', 
                    linestyle='--', linewidth=1.5, alpha=0.8,
                    label=f'Цель 1: {signal["target_1"]}')
        if signal.get('target_2'):
            ax1.axhline(y=signal['target_2'], color='#00ff00', # ← ЗЕЛЕНЫЙ 
                    linestyle='--', linewidth=1.5, alpha=0.8,
                    label=f'Цель 2: {signal["target_2"]}')
        if signal.get('stop_loss'):
            ax1.axhline(y=signal['stop_loss'], color='#ff0000', 
                    linestyle='--', linewidth=1.5, alpha=0.8,
                    label=f'Стоп: {signal["stop_loss"]}')
        
        # ===== FVG ЗОНЫ - ТОЛЬКО 2 БЛИЖАЙШИЕ =====
        if 'fvg_zones' in signal and signal['fvg_zones']:
            # Берем ТОЛЬКО 2 ближайшие зоны
            fvg_to_show = signal['fvg_zones'][:2]
            logger.info(f"  🎨 Рисую {len(fvg_to_show)} FVG зон на графике")
            
            for zone in fvg_to_show:
                # Определяем цвет в зависимости от типа
                color = '#00ff00' if zone['type'] == 'bullish' else '#ff0000'
                alpha = 0.2
                
                # Рисуем зону
                ax1.axhspan(zone['min'], zone['max'], 
                        alpha=alpha, color=color, linewidth=0)
                
                # Добавляем границы зоны
                ax1.axhline(y=zone['min'], color=color, linestyle=':', 
                        linewidth=1, alpha=0.5)
                ax1.axhline(y=zone['max'], color=color, linestyle=':', 
                        linewidth=1, alpha=0.5)
                
                # Добавляем метку с таймфреймом и размером
                mid_price = (zone['min'] + zone['max']) / 2
                label = f"FVG {zone.get('tf_short', '?')} ({zone.get('size', 0):.1f}%)"
                ax1.text(plot_df.index[-1], mid_price, label, 
                        color='white',  # ← только это поменяли с color=color на color='white'
                        fontsize=8,     # ← оставляем как было
                        alpha=0.8,      # ← оставляем как было
                        verticalalignment='center',
                        horizontalalignment='right',
                        bbox=dict(boxstyle="round,pad=0.2", 
                                facecolor='black', 
                                alpha=0.5))  # ← оставляем как было

        # Добавляем отрисовку зон дисперсии
        if 'dispersion_zones' in signal and signal['dispersion_zones']:
            for zone in signal['dispersion_zones'][:3]:
                ax1.axhspan(zone['min'], zone['max'], 
                        alpha=0.2, color='orange', linewidth=0)
                
                # Добавляем метку
                mid_price = (zone['min'] + zone['max']) / 2
                ax1.text(plot_df.index[-1], mid_price, 
                        f"📊 ДИСПЕРСИЯ {zone['strength']:.0f}%", 
                        color='orange', fontsize=7, alpha=0.7,
                        verticalalignment='center',
                        horizontalalignment='right')
        
        # Заголовок
        ax1.set_title(f'{coin} - {signal["direction"]} (TF: {timeframe}, уверенность {signal["confidence"]}%)', 
                    fontsize=14, fontweight='bold', color='white')
        ax1.set_ylabel('Price (USDT)', color='white')
        ax1.legend(loc='upper left', fontsize=8, facecolor='#222222')
        ax1.grid(True, alpha=0.2, linestyle='--')
        ax1.tick_params(colors='white')
        ax1.set_facecolor('#111111')
        
        # ===== НИЖНИЙ ГРАФИК =====
        if 'rsi' in plot_df.columns:
            ax2.plot(plot_df.index, plot_df['rsi'], 
                    color='purple', linewidth=2, label='RSI 14')
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
            ax2.fill_between(plot_df.index, 30, 70, alpha=0.1, color='gray')
            
            if pd.notna(plot_df['rsi'].iloc[-1]):
                current_rsi = plot_df['rsi'].iloc[-1]
                ax2.scatter(plot_df.index[-1], current_rsi, 
                        color='yellow', s=50, zorder=5)
        
        ax2.set_ylabel('RSI', color='white')
        ax2.set_xlabel('Time', color='white')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.2, linestyle='--')
        ax2.tick_params(colors='white')
        ax2.set_facecolor('#111111')
        ax2.legend(loc='upper left', fontsize=8, facecolor='#222222')
        
        # Форматирование времени
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='PNG', dpi=self.dpi, 
                bbox_inches='tight', facecolor='#111111')
        buf.seek(0)
        plt.close()
        
        return buf

# ============== АНАЛИЗАТОР ДИВЕРГЕНЦИЙ ==============

class DivergenceAnalyzer:
    def __init__(self):
        self.lookback = 30
    
    def find_swings(self, df: pd.DataFrame, column: str = 'close', window: int = 5) -> Tuple[List, List]:
        highs = []
        lows = []
        for i in range(window, len(df) - window):
            if all(df[column].iloc[i] > df[column].iloc[j] 
                   for j in range(i - window, i + window + 1) if j != i):
                highs.append((i, df[column].iloc[i]))
            if all(df[column].iloc[i] < df[column].iloc[j] 
                   for j in range(i - window, i + window + 1) if j != i):
                lows.append((i, df[column].iloc[i]))
        return highs, lows
    
    def detect_rsi_divergence(self, df: pd.DataFrame, timeframe: str) -> Dict:
        result = {
            'bullish': False,
            'bearish': False,
            'strength': 0,
            'description': '',
            'timeframe': timeframe
        }
        if 'rsi' not in df.columns:
            return result
            
        price_highs, price_lows = self.find_swings(df, 'close')
        rsi_highs, rsi_lows = self.find_swings(df, 'rsi')
        
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            last_price_low = price_lows[-1]
            prev_price_low = price_lows[-2]
            last_rsi_low = rsi_lows[-1]
            prev_rsi_low = rsi_lows[-2]
            if (last_price_low[1] < prev_price_low[1] and 
                last_rsi_low[1] > prev_rsi_low[1]):
                result['bullish'] = True
                result['strength'] = min(100, abs(last_price_low[1] - prev_price_low[1]) / prev_price_low[1] * 500)
                result['description'] = f"Бычья дивергенция RSI ({timeframe})"
        
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            last_price_high = price_highs[-1]
            prev_price_high = price_highs[-2]
            last_rsi_high = rsi_highs[-1]
            prev_rsi_high = rsi_highs[-2]
            if (last_price_high[1] > prev_price_high[1] and 
                last_rsi_high[1] < prev_rsi_high[1]):
                result['bearish'] = True
                result['strength'] = min(100, abs(last_price_high[1] - prev_price_high[1]) / prev_price_high[1] * 500)
                result['description'] = f"Медвежья дивергенция RSI ({timeframe})"
        return result
    
    def detect_macd_divergence(self, df: pd.DataFrame, timeframe: str) -> Dict:
        result = {
            'bullish': False,
            'bearish': False,
            'strength': 0,
            'description': '',
            'timeframe': timeframe
        }
        if 'MACD_12_26_9' not in df.columns:
            return result
            
        price_highs, price_lows = self.find_swings(df, 'close')
        macd_highs, macd_lows = self.find_swings(df, 'MACD_12_26_9')
        
        if len(price_lows) >= 2 and len(macd_lows) >= 2:
            last_price_low = price_lows[-1]
            prev_price_low = price_lows[-2]
            last_macd_low = macd_lows[-1]
            prev_macd_low = macd_lows[-2]
            if (last_price_low[1] < prev_price_low[1] and 
                last_macd_low[1] > prev_macd_low[1]):
                result['bullish'] = True
                result['strength'] = min(100, abs(last_price_low[1] - prev_price_low[1]) / prev_price_low[1] * 500)
                result['description'] = f"Бычья дивергенция MACD ({timeframe})"
        
        if len(price_highs) >= 2 and len(macd_highs) >= 2:
            last_price_high = price_highs[-1]
            prev_price_high = price_highs[-2]
            last_macd_high = macd_highs[-1]
            prev_macd_high = macd_highs[-2]
            if (last_price_high[1] > prev_price_high[1] and 
                last_macd_high[1] < prev_macd_high[1]):
                result['bearish'] = True
                result['strength'] = min(100, abs(last_price_high[1] - prev_price_high[1]) / prev_price_high[1] * 500)
                result['description'] = f"Медвежья дивергенция MACD ({timeframe})"
        return result
    
    def analyze(self, df: pd.DataFrame, timeframe: str) -> Dict:
        rsi_div = self.detect_rsi_divergence(df, timeframe)
        macd_div = self.detect_macd_divergence(df, timeframe)
        
        result = {
            'has_divergence': rsi_div['bullish'] or rsi_div['bearish'] or macd_div['bullish'] or macd_div['bearish'],
            'bullish': rsi_div['bullish'] or macd_div['bullish'],
            'bearish': rsi_div['bearish'] or macd_div['bearish'],
            'strength': max(rsi_div.get('strength', 0), macd_div.get('strength', 0)),
            'signals': []
        }
        
        unique_signals = set()
        if rsi_div['bullish']:
            unique_signals.add(rsi_div['description'])
        if rsi_div['bearish']:
            unique_signals.add(rsi_div['description'])
        if macd_div['bullish']:
            unique_signals.add(macd_div['description'])
        if macd_div['bearish']:
            unique_signals.add(macd_div['description'])
        
        result['signals'] = list(unique_signals)
        return result

# ============== SMART MONEY АНАЛИЗАТОР ==============

class SmartMoneyAnalyzer:
    def __init__(self, settings: Dict = None):
        self.settings = settings or SMC_SETTINGS
        self.order_blocks = []
        self.fair_value_gaps = []
        
    def _is_swing_high(self, df: pd.DataFrame, idx: int, window: int = 5) -> bool:
        """Проверка, является ли свеча локальным максимумом"""
        if idx < window or idx >= len(df) - window:
            return False
        high = df['high'].iloc[idx]
        for i in range(1, window + 1):
            if high <= df['high'].iloc[idx - i]:
                return False
            if high <= df['high'].iloc[idx + i]:
                return False
        return True

    def _is_swing_low(self, df: pd.DataFrame, idx: int, window: int = 5) -> bool:
        """Проверка, является ли свеча локальным минимумом"""
        if idx < window or idx >= len(df) - window:
            return False
        low = df['low'].iloc[idx]
        for i in range(1, window + 1):
            if low >= df['low'].iloc[idx - i]:
                return False
            if low >= df['low'].iloc[idx + i]:
                return False
        return True
    
    def find_order_blocks_improved(self, df: pd.DataFrame, tf_name: str) -> List[Dict]:
        """
        Улучшенный поиск ордер-блоков с фильтром по ATR
        Как в индикаторе SMC
        """
        blocks = []
        
        # Расчёт ATR для фильтра
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else (df['high'] - df['low']).mean()
        
        for i in range(10, len(df) - 5):
            # Движение цены
            price_move = (df['close'].iloc[i+2] - df['close'].iloc[i]) / df['close'].iloc[i] * 100
            
            if abs(price_move) < 2.0:
                continue
            
            # Фильтр по волатильности (пропускаем слишком широкие свечи)
            candle_range = df['high'].iloc[i] - df['low'].iloc[i]
            if candle_range > 2 * atr:
                continue
            
            if price_move > 0:  # Бычий OB
                # Ищем последнюю красную свечу перед импульсом
                for j in range(i, max(0, i-5), -1):
                    if df['close'].iloc[j] < df['open'].iloc[j]:
                        blocks.append({
                            'type': 'bullish',
                            'price_min': df['low'].iloc[j],
                            'price_max': df['high'].iloc[j],
                            'strength': min(100, abs(price_move) * 12),
                            'timeframe': tf_name,
                            'description': f"📦 Бычий OB на {tf_name}: {df['low'].iloc[j]:.4f}-{df['high'].iloc[j]:.4f}"
                        })
                        break
            else:  # Медвежий OB
                for j in range(i, max(0, i-5), -1):
                    if df['close'].iloc[j] > df['open'].iloc[j]:
                        blocks.append({
                            'type': 'bearish',
                            'price_min': df['low'].iloc[j],
                            'price_max': df['high'].iloc[j],
                            'strength': min(100, abs(price_move) * 12),
                            'timeframe': tf_name,
                            'description': f"📦 Медвежий OB на {tf_name}: {df['low'].iloc[j]:.4f}-{df['high'].iloc[j]:.4f}"
                        })
                        break
        
        return blocks[:5]  # последние 5 блоков
    
    def find_fair_value_gaps(self, df: pd.DataFrame) -> List[Dict]:
        fvg_list = []
        for i in range(1, len(df) - 1):
            candle1 = df.iloc[i-1]
            candle2 = df.iloc[i]
            candle3 = df.iloc[i+1]
            
            if candle3['low'] > candle1['high']:
                gap_size = (candle3['low'] - candle1['high']) / candle1['high'] * 100
                fvg_list.append({
                    'type': 'bullish',
                    'price_min': candle1['high'],
                    'price_max': candle3['low'],
                    'size': gap_size,
                    'strength': min(100, gap_size * 20),
                    'description': f"Бычий FVG ({gap_size:.2f}%)"
                })
            elif candle3['high'] < candle1['low']:
                gap_size = (candle1['low'] - candle3['high']) / candle3['high'] * 100
                fvg_list.append({
                    'type': 'bearish',
                    'price_min': candle3['high'],
                    'price_max': candle1['low'],
                    'size': gap_size,
                    'strength': min(100, gap_size * 20),
                    'description': f"Медвежий FVG ({gap_size:.2f}%)"
                })
        return fvg_list[-10:]
    
    def analyze(self, df: pd.DataFrame, current_price: float) -> Dict:
        result = {
            'has_signal': False,
            'signals': [],
            'strength': 0
        }
        
        order_blocks = self.find_order_blocks(df)
        for ob in order_blocks:
            if ob['price_min'] <= current_price <= ob['price_max']:
                result['has_signal'] = True
                result['signals'].append(ob['description'])
                result['strength'] = max(result['strength'], ob['strength'])
        
        fvg_list = self.find_fair_value_gaps(df)
        for fvg in fvg_list:
            if fvg['price_min'] <= current_price <= fvg['price_max']:
                result['has_signal'] = True
                result['signals'].append(f"📐 FVG: {fvg['description']}")
                result['strength'] = max(result['strength'], fvg['strength'])
        
        return result

    def detect_choch(self, df: pd.DataFrame, tf_name: str) -> Dict:
        """
        Детектор CHoCH (Change of Character) - смена тренда
        Как в индикаторе SMC
        """
        result = {
            'has_choch': False,
            'direction': None,
            'strength': 0,
            'description': '',
            'timeframe': tf_name
        }
        
        # Находим локальные экстремумы
        highs = []
        lows = []
        window = 5
        
        for i in range(window, len(df) - window):
            if self._is_swing_high(df, i, window):
                highs.append({'idx': i, 'price': df['high'].iloc[i]})
            if self._is_swing_low(df, i, window):
                lows.append({'idx': i, 'price': df['low'].iloc[i]})
        
        if len(highs) < 2 or len(lows) < 2:
            return result
        
        # Получаем последние два экстремума
        last_high = highs[-1]['price']
        prev_high = highs[-2]['price']
        last_low = lows[-1]['price']
        prev_low = lows[-2]['price']
        
        current_price = df['close'].iloc[-1]
        
        # CHoCH вверх (смена тренда с нисходящего на восходящий)
        if current_price > prev_high and last_high < prev_high:
            result['has_choch'] = True
            result['direction'] = 'LONG'
            result['strength'] = 85
            result['description'] = f"🔄 CHoCH (смена тренда) на {tf_name}: пробой уровня {prev_high:.4f}"
            return result
        
        # CHoCH вниз (смена тренда с восходящего на нисходящий)
        elif current_price < prev_low and last_low > prev_low:
            result['has_choch'] = True
            result['direction'] = 'SHORT'
            result['strength'] = 85
            result['description'] = f"🔄 CHoCH (смена тренда) на {tf_name}: пробой уровня {prev_low:.4f}"
            return result
        
        return result

    def calculate_premium_discount_zones(self, df: pd.DataFrame, tf_name: str) -> Dict:
        """
        Расчёт Premium/Discount/Equilibrium зон как в SMC
        """
        result = {
            'has_zone': False,
            'zone_type': None,
            'direction': None,
            'premium': 0,
            'discount': 0,
            'equilibrium': 0,
            'description': ''
        }
        
        lookback = min(100, len(df))
        swing_high = df['high'].tail(lookback).max()
        swing_low = df['low'].tail(lookback).min()
        
        if swing_high == swing_low:
            return result
        
        current_price = df['close'].iloc[-1]
        
        # Расчёт зон (как в индикаторе SMC)
        premium_zone = swing_high * 0.95 + swing_low * 0.05
        discount_zone = swing_low * 0.95 + swing_high * 0.05
        equilibrium = (swing_high + swing_low) / 2
        
        result['premium'] = premium_zone
        result['discount'] = discount_zone
        result['equilibrium'] = equilibrium
        
        # Определяем зону
        if current_price > premium_zone:
            result['has_zone'] = True
            result['zone_type'] = 'premium'
            result['direction'] = 'SHORT'
            result['description'] = f"📊 Premium Zone на {tf_name}: цена {current_price:.4f} (перекупленность)"
        elif current_price < discount_zone:
            result['has_zone'] = True
            result['zone_type'] = 'discount'
            result['direction'] = 'LONG'
            result['description'] = f"📊 Discount Zone на {tf_name}: цена {current_price:.4f} (перепроданность)"
        elif abs(current_price - equilibrium) / equilibrium * 100 < 1.0:
            result['has_zone'] = True
            result['zone_type'] = 'equilibrium'
            result['direction'] = None
            result['description'] = f"⚖️ Equilibrium Zone на {tf_name}: цена {current_price:.4f} (равновесие, возможна цель)"
        
        return result

    def find_equal_highs_lows(self, df: pd.DataFrame, tf_name: str, current_price: float, signal_type: str = 'regular') -> Dict:
        """
        Поиск равных максимумов (EQH) и равных минимумов (EQL)
        Это уровни, где толпа держит стоп-лоссы
        """
        from config import EQUAL_HIGH_LOW_SETTINGS
        
        result = {
            'has_equal': False,
            'type': None,
            'price': 0,
            'strength': 0,
            'description': ''
        }
        
        if not EQUAL_HIGH_LOW_SETTINGS.get('enabled', True):
            return result
        
        threshold_pct = EQUAL_HIGH_LOW_SETTINGS.get('threshold_pct', 0.1)
        confirmation_bars = EQUAL_HIGH_LOW_SETTINGS.get('confirmation_bars', 3)
        
        # Выбираем макс расстояние в зависимости от типа сигнала
        if signal_type in ['PUMP', 'DUMP', 'pump']:
            max_distance = EQUAL_HIGH_LOW_SETTINGS.get('max_distance_pct', 5.0)
        elif signal_type == 'accumulation':
            max_distance = EQUAL_HIGH_LOW_SETTINGS.get('max_distance_accumulation_pct', 25.0)
        else:
            max_distance = EQUAL_HIGH_LOW_SETTINGS.get('max_distance_regular_pct', 15.0)
        
        # Поиск равных максимумов (EQH)
        highs = df['high'].values
        for i in range(confirmation_bars, len(highs) - confirmation_bars):
            current_high = highs[i]
            
            # Ищем похожий максимум в прошлом
            for j in range(i - confirmation_bars, max(0, i - 50), -1):
                prev_high = highs[j]
                diff_pct = abs(current_high - prev_high) / prev_high * 100
                
                if diff_pct <= threshold_pct:
                    # Проверяем подтверждение
                    if all(h < current_high for h in highs[i+1:i+confirmation_bars+1]):
                        result['has_equal'] = True
                        result['type'] = 'EQH'
                        result['price'] = current_high
                        result['strength'] = 70
                        result['description'] = f"📐 EQH (Equal High) на {tf_name}: {current_high:.4f} — уровень ликвидности"
                        break
            if result['has_equal']:
                break
        
        # Поиск равных минимумов (EQL)
        if not result['has_equal']:
            lows = df['low'].values
            for i in range(confirmation_bars, len(lows) - confirmation_bars):
                current_low = lows[i]
                
                for j in range(i - confirmation_bars, max(0, i - 50), -1):
                    prev_low = lows[j]
                    diff_pct = abs(current_low - prev_low) / prev_low * 100
                    
                    if diff_pct <= threshold_pct:
                        if all(l > current_low for l in lows[i+1:i+confirmation_bars+1]):
                            result['has_equal'] = True
                            result['type'] = 'EQL'
                            result['price'] = current_low
                            result['strength'] = 70
                            result['description'] = f"📐 EQL (Equal Low) на {tf_name}: {current_low:.4f} — уровень ликвидности"
                            break
                if result['has_equal']:
                    break
        
        # Проверяем расстояние до цены (только если нашли)
        if result['has_equal']:
            distance_pct = abs(result['price'] - current_price) / current_price * 100
            if distance_pct > max_distance:
                return {'has_equal': False, 'type': None, 'price': 0, 'strength': 0, 'description': ''}
            result['distance'] = distance_pct
            result['description'] += f" ({distance_pct:.1f}% от цены)"
        
        return result

# ============== ФРАКТАЛЬНЫЙ АНАЛИЗАТОР ==============

class FractalAnalyzer:
    def __init__(self, settings: Dict = None):
        self.settings = settings or FRACTAL_SETTINGS
        self.window = self.settings.get('window', 5)
        
    def analyze(self, df: pd.DataFrame) -> Dict:
        fractal_up = 0
        fractal_down = 0
        
        for i in range(self.window, len(df) - self.window):
            if all(df['high'].iloc[i] > df['high'].iloc[j] 
                   for j in range(i - self.window, i + self.window + 1) if j != i):
                fractal_up += 1
            if all(df['low'].iloc[i] < df['low'].iloc[j] 
                   for j in range(i - self.window, i + self.window + 1) if j != i):
                fractal_down += 1
        
        result = {
            'has_fractal': False,
            'signals': [],
            'strength': 0
        }
        
        total = fractal_up + fractal_down
        if total > 0:
            if fractal_up > fractal_down * 2:
                result['has_fractal'] = True
                result['signals'].append(f"Преобладание бычьих фракталов ({fractal_up}/{fractal_down})")
                result['strength'] = 70
            elif fractal_down > fractal_up * 2:
                result['has_fractal'] = True
                result['signals'].append(f"Преобладание медвежьих фракталов ({fractal_down}/{fractal_up})")
                result['strength'] = 70
        
        return result

# ============== СБОРЩИК ВСЕХ УРОВНЕЙ ==============

class Level:
    """Универсальный класс для любого уровня"""
    
    def __init__(self, level_type: str, price: float, strength: int, tf: str, 
                 source: str, touches: int = 0, is_dynamic: bool = False):
        self.level_type = level_type  # 'horizontal', 'trendline', 'fvg', 'fib', 'ema'
        self.price = price
        self.strength = strength  # 0-100
        self.tf = tf  # '15m', '1h', '4h', '1d', '1w', '1M'
        self.source = source  # описание
        self.touches = touches  # сколько раз касались
        self.is_dynamic = is_dynamic  # динамический уровень (EMA) или статический
        self.min_price = price if level_type != 'fvg' else None
        self.max_price = price if level_type != 'fvg' else None
        self.is_broken = False
        self.breakout_time = None
        self.breakout_price = None

class LevelCollector:
    """Сбор всех сильных уровней со всех таймфреймов"""
    
    def __init__(self):
        self.levels = []
    
    def collect_levels(self, dataframes: Dict[str, pd.DataFrame], current_price: float) -> List[Level]:
        """Сбор уровней со всех таймфреймов"""
        all_levels = []
        
        # Приоритет таймфреймов (старшие важнее)
        tf_priority = ['monthly', 'weekly', 'daily', 'four_hourly', 'hourly', 'current']
        tf_weights = {'monthly': 4.0, 'weekly': 3.5, 'daily': 3.0, 
                     'four_hourly': 2.5, 'hourly': 2.0, 'current': 1.0}
        
        for tf in tf_priority:
            if tf not in dataframes or dataframes[tf] is None:
                continue
            
            df = dataframes[tf]
            tf_weight = tf_weights.get(tf, 1.0)
            
            # 1. Горизонтальные уровни (локальные максимумы/минимумы)
            levels_h = self._find_horizontal_levels(df, tf, tf_weight)
            all_levels.extend(levels_h)
            
            # 2. EMA уровни (EMA 200, EMA 50)
            levels_ema = self._find_ema_levels(df, tf, tf_weight)
            all_levels.extend(levels_ema)
            
            # 3. FVG зоны (из вашего FVG анализа)
            if 'fvg_analysis' in dataframes[tf]:
                levels_fvg = self._convert_fvg_to_levels(dataframes[tf]['fvg_analysis'], tf, tf_weight)
                all_levels.extend(levels_fvg)
            
            # 4. Уровни Фибоначчи
            if 'fib_analysis' in dataframes[tf]:
                levels_fib = self._convert_fib_to_levels(dataframes[tf]['fib_analysis'], tf, tf_weight)
                all_levels.extend(levels_fib)
        
        # Сортируем по расстоянию до цены
        for level in all_levels:
            if level.min_price <= current_price <= level.max_price:
                level.distance = 0
            elif level.min_price > current_price:
                level.distance = (level.min_price - current_price) / current_price
            else:
                level.distance = (current_price - level.max_price) / current_price
        
        all_levels.sort(key=lambda x: x.distance)
        
        return all_levels[:20]  # топ-20 ближайших уровней
    
    def _find_horizontal_levels(self, df: pd.DataFrame, tf: str, weight: float) -> List[Level]:
        """Поиск горизонтальных уровней"""
        levels = []
        window = 20
        
        # Ищем локальные максимумы (сопротивление)
        for i in range(window, len(df) - window):
            if df['high'].iloc[i] == max(df['high'].iloc[i-window:i+window]):
                price = df['high'].iloc[i]
                strength = self.calculate_level_strength('horizontal', tf, 0, size=0)
                
                # Считаем касания
                touches = sum(1 for j in range(i, len(df)) 
                            if abs(df['high'].iloc[j] - price) / price < 0.003)
                
                # Пересчитываем силу с учетом касаний
                strength = self.calculate_level_strength('horizontal', tf, touches, size=0)
                
                level = Level(
                    level_type='horizontal_resistance',
                    price=price,
                    strength=strength,
                    tf=tf,
                    source=f"Локальный максимум ({tf})",
                    touches=touches
                )
                level.min_price = price * 0.995
                level.max_price = price * 1.005
                levels.append(level)
        
        # Ищем локальные минимумы (поддержка)
        for i in range(window, len(df) - window):
            if df['low'].iloc[i] == min(df['low'].iloc[i-window:i+window]):
                price = df['low'].iloc[i]
                strength = self.calculate_level_strength('horizontal', tf, 0, size=0)
                
                touches = sum(1 for j in range(i, len(df)) 
                            if abs(df['low'].iloc[j] - price) / price < 0.003)
                
                strength = self.calculate_level_strength('horizontal', tf, touches, size=0)
                
                level = Level(
                    level_type='horizontal_support',
                    price=price,
                    strength=strength,
                    tf=tf,
                    source=f"Локальный минимум ({tf})",
                    touches=touches
                )
                level.min_price = price * 0.995
                level.max_price = price * 1.005
                levels.append(level)
        
        return levels
    
    def _find_ema_levels(self, df: pd.DataFrame, tf: str, weight: float) -> List[Level]:
        """Поиск EMA уровней"""
        levels = []
        
        for period, color, importance in [(200, '#ff4444', 3.0), (50, '#8888ff', 2.0)]:
            ema_col = f'ema_{period}'
            if ema_col not in df.columns:
                continue
            
            current_ema = df[ema_col].iloc[-1]
            
            # Считаем касания EMA с защитой от выхода за границы
            touches = 0
            df_len = len(df)
            start_idx = max(0, df_len - 50)  # последние 50 свечей
            
            for i in range(start_idx, df_len):
                try:
                    close_price = df['close'].iloc[i]
                    ema_value = df[ema_col].iloc[i]
                    
                    if pd.isna(close_price) or pd.isna(ema_value) or ema_value == 0:
                        continue
                    
                    if abs(close_price - ema_value) / ema_value < 0.003:
                        touches += 1
                except IndexError:
                    continue
            
            strength = self.calculate_level_strength(f'ema_{period}', tf, touches, size=0)
            
            level = Level(
                level_type=f'ema_{period}',
                price=current_ema,
                strength=strength,
                tf=tf,
                source=f"EMA {period} ({tf})",
                touches=touches,
                is_dynamic=True
            )
            level.min_price = current_ema * 0.99
            level.max_price = current_ema * 1.01
            levels.append(level)
        
        return levels    
    
    def calculate_level_strength(self, level_type: str, tf: str, touches: int, **kwargs) -> int:
        """Расчет силы уровня по настройкам"""
        
        weights = LEVEL_ANALYSIS_SETTINGS['weights']
        
        if level_type == 'horizontal' or level_type in ['horizontal_resistance', 'horizontal_support']:
            base = weights['horizontal']['base']
            per_touch = weights['horizontal']['per_touch']
            tf_mult = weights['horizontal']['tf_multiplier'].get(tf, 1.0)
            
            strength = (base + touches * per_touch) * tf_mult
            
        elif level_type == 'fvg' or 'fvg' in level_type:
            base = weights['fvg']['base']
            size_mult = weights['fvg']['size_multiplier']
            tf_mult = weights['fvg']['tf_multiplier'].get(tf, 1.0)
            
            strength = (base + kwargs.get('size', 0) * size_mult) * tf_mult
            
        elif 'ema' in level_type:
            if '200' in level_type:
                base = weights['ema']['ema_200']
            else:
                base = weights['ema']['ema_50']
            tf_mult = weights['ema']['tf_multiplier'].get(tf, 1.0)
            
            strength = (base + touches * 2) * tf_mult  # +2 за каждое касание EMA
            
        else:
            # По умолчанию
            strength = 50 * tf_multiplier.get(tf, 1.0)
        
        return min(100, int(strength))

    def find_confluence_levels(self, all_levels: List[Level], current_price: float, tolerance: float = 0.5) -> List[Dict]:
        """
        Поиск уровней, которые совпадают на разных таймфреймах
        tolerance: допустимое отклонение в процентах (0.5%)
        """
        confluence_zones = []
        
        # Группируем уровни по цене (с допуском)
        for level in all_levels:
            if level.distance > 20:  # не учитываем слишком далекие
                continue
                
            # Ищем другие уровни в этой же ценовой зоне
            matching_levels = []
            for other in all_levels:
                if other == level:
                    continue
                # Проверяем, совпадают ли ценовые зоны
                price_diff = abs(level.price - other.price) / level.price * 100
                if price_diff <= tolerance:
                    matching_levels.append(other)
            
            if matching_levels:
                # Собираем все таймфреймы, где есть уровни
                tfs = [level.tf] + [l.tf for l in matching_levels]
                unique_tfs = list(set(tfs))
                
                # Считаем силу конвергенции
                strength = len(unique_tfs) * 20  # базовый вес за каждый ТФ
                for l in [level] + matching_levels:
                    strength += l.strength / 10
                
                # Определяем тип (поддержка или сопротивление)
                if level.price < current_price:
                    zone_type = "поддержка"
                    direction = "LONG"
                else:
                    zone_type = "сопротивление"
                    direction = "SHORT"
                
                confluence_zones.append({
                    'price': level.price,
                    'zone_type': zone_type,
                    'direction': direction,
                    'timeframes': unique_tfs,
                    'strength': min(100, strength),
                    'levels': [level] + matching_levels,
                    'distance': level.distance,
                    'source': f"Конвергенция на {', '.join(unique_tfs)}"
                })
        
        # Сортируем по силе и расстоянию
        confluence_zones.sort(key=lambda x: (-x['strength'], x['distance']))
        return confluence_zones[:5]  # топ-5 сильных зон

    def calculate_level_strength_score(self, level: Dict, df: pd.DataFrame, last: pd.Series, alignment: Dict) -> Dict:
        """
        Оценка силы уровня и вероятности разворота/пробоя
        """
        result = {
            'strength': 0,           # 0-100
            'probability': 0,        # 0-100 (вероятность разворота)
            'direction': None,
            'signals': [],
            'action': 'наблюдение'    # 'разворот', 'пробой', 'ложный_пробой'
        }
        
        # ===== 1. КОНВЕРГЕНЦИЯ ТАЙМФРЕЙМОВ =====
        if 'timeframes' in level:
            tf_count = len(level['timeframes'])
        else:
            tf_count = 1
        
        if tf_count >= 4:
            result['strength'] += 40
            result['signals'].append(f"🔥 СУПЕР-КОНВЕРГЕНЦИЯ: {tf_count} ТФ")
            result['probability'] += 30
        elif tf_count >= 3:
            result['strength'] += 30
            result['signals'].append(f"⭐ Сильная конвергенция: {tf_count} ТФ")
            result['probability'] += 20
        elif tf_count >= 2:
            result['strength'] += 20
            result['signals'].append(f"📊 Конвергенция: {tf_count} ТФ")
            result['probability'] += 10
        
        # ===== 2. КАСАНИЯ УРОВНЯ =====
        touches = level.get('touches', 1)
        if touches >= 7:
            result['strength'] += 25
            result['signals'].append(f"🎯 Уровень тестирован {touches} раз (очень сильный)")
            result['probability'] += 20
        elif touches >= 5:
            result['strength'] += 20
            result['signals'].append(f"✅ Уровень тестирован {touches} раз")
            result['probability'] += 15
        elif touches >= 3:
            result['strength'] += 12
            result['signals'].append(f"📌 Уровень тестирован {touches} раза")
            result['probability'] += 8
        
        # ===== 3. ОБЪЕМ НА ПОДХОДЕ =====
        volume_ratio = last.get('volume_ratio', 1.0)
        if volume_ratio > 3.0:
            result['strength'] += 25
            result['signals'].append(f"🔥 Аномальный объем x{volume_ratio:.1f}")
            result['probability'] += 15
        elif volume_ratio > 2.0:
            result['strength'] += 18
            result['signals'].append(f"⚡ Высокий объем x{volume_ratio:.1f}")
            result['probability'] += 10
        elif volume_ratio > 1.5:
            result['strength'] += 10
            result['signals'].append(f"📊 Повышенный объем x{volume_ratio:.1f}")
            result['probability'] += 5
        
        # ===== 4. RSI ЭКСТРЕМУМ =====
        rsi = last.get('rsi', 50)
        if rsi > 85:
            result['strength'] += 15
            result['signals'].append(f"🔴 RSI сильно перекуплен ({rsi:.1f})")
            result['probability'] += 15
        elif rsi > 75:
            result['strength'] += 10
            result['signals'].append(f"🟡 RSI перекуплен ({rsi:.1f})")
            result['probability'] += 8
        elif rsi < 15:
            result['strength'] += 15
            result['signals'].append(f"🟢 RSI сильно перепродан ({rsi:.1f})")
            result['probability'] += 15
        elif rsi < 25:
            result['strength'] += 10
            result['signals'].append(f"📉 RSI перепродан ({rsi:.1f})")
            result['probability'] += 8
        
        # ===== 5. ТРЕНД =====
        weekly_trend = alignment.get('weekly_trend', '')
        daily_trend = alignment.get('daily_trend', '')
        zone_type = level.get('zone_type', '')
        
        if zone_type == 'поддержка':
            if weekly_trend == 'ВОСХОДЯЩИЙ' or daily_trend == 'ВОСХОДЯЩИЙ':
                result['strength'] += 20
                result['signals'].append("📈 Тренд вверх (поддержка усилена)")
                result['probability'] += 15
                result['direction'] = 'LONG'
            else:
                result['direction'] = 'LONG' if result['probability'] > 50 else 'NEUTRAL'
        else:  # сопротивление
            if weekly_trend == 'НИСХОДЯЩИЙ' or daily_trend == 'НИСХОДЯЩИЙ':
                result['strength'] += 20
                result['signals'].append("📉 Тренд вниз (сопротивление усилено)")
                result['probability'] += 15
                result['direction'] = 'SHORT'
            else:
                result['direction'] = 'SHORT' if result['probability'] > 50 else 'NEUTRAL'
        
        # ===== 6. ИМПУЛЬС ПОДХОДА =====
        distance = level.get('distance', 100)
        if distance < 1.0:
            result['strength'] += 15
            result['signals'].append(f"⚡ Цена у уровня ({distance:.1f}%)")
            result['probability'] += 20
        elif distance < 2.0:
            result['strength'] += 10
            result['signals'].append(f"🎯 Цена близко к уровню ({distance:.1f}%)")
            result['probability'] += 10
        
        # ===== 7. ОПРЕДЕЛЕНИЕ ДЕЙСТВИЯ =====
        result['strength'] = min(100, result['strength'])
        result['probability'] = min(100, result['probability'])
        
        if result['probability'] >= 70:
            result['action'] = 'разворот'
        elif result['probability'] >= 50:
            result['action'] = 'вероятный_разворот'
        elif result['probability'] >= 30:
            result['action'] = 'наблюдение'
        else:
            result['action'] = 'возможный_пробой'
        
        return result

# ============== УНИВЕРСАЛЬНЫЙ ДЕТЕКТОР ПРОБОЕВ ==============

class UniversalBreakoutDetector:
    """Детектор пробоев для любых уровней"""
    
    def __init__(self):
        self.tracked_breakouts = {}  # отслеживаем все пробои
        self.fakeouts = set()  # ложные пробои
    
    def analyze_level(self, level: Level, current_price: float, 
                     df: pd.DataFrame, required_candles: int = 3) -> Dict:
        """Анализ одного уровня"""
        
        key = f"{level.tf}_{level.level_type}_{level.price}"
        
        # Определяем статус пробоя
        if level.level_type in ['horizontal_resistance', 'trendline_resistance', 'fib_resistance']:
            is_breakout = current_price > level.max_price
            breakout_direction = "вверх"
            target_direction = "LONG"
        else:
            is_breakout = current_price < level.min_price
            breakout_direction = "вниз"
            target_direction = "SHORT"
        
        # Нет пробоя
        if not is_breakout:
            if key in self.tracked_breakouts:
                # Проверяем, не был ли это ложный пробой
                tracker = self.tracked_breakouts[key]
                if tracker['status'] == 'potential' and tracker['max_price'] > level.max_price * 1.01:
                    # Был пробой, но цена вернулась - ЭТО ЛОЖНЫЙ!
                    self.fakeouts.add(key)
                    return {
                        'type': 'fakeout',
                        'level': level,
                        'direction': breakout_direction,
                        'target': target_direction,
                        'message': f"🚨 ЛОЖНЫЙ ПРОБОЙ {breakout_direction} на {level.tf}!",
                        'confidence': level.strength * 1.5
                    }
                del self.tracked_breakouts[key]
            return None
        
        # Есть пробой
        if key not in self.tracked_breakouts:
            # Первый раз видим пробой
            self.tracked_breakouts[key] = {
                'level': level,
                'breakout_price': current_price,
                'max_price': current_price,
                'min_price': current_price,
                'time': datetime.now(),
                'candles': 1,
                'status': 'potential'
            }
            return {
                'type': 'breakout_alert',
                'level': level,
                'direction': breakout_direction,
                'message': f"⚠️ ПРОБОЙ {breakout_direction} на {level.tf}! (ожидание подтверждения)",
                'confidence': level.strength
            }
        
        # Отслеживаем пробой
        tracker = self.tracked_breakouts[key]
        tracker['candles'] += 1
        tracker['max_price'] = max(tracker['max_price'], current_price)
        tracker['min_price'] = min(tracker['min_price'], current_price)
        
        # Проверяем подтверждение
        if tracker['candles'] >= required_candles:
            # Проверяем размер движения
            if breakout_direction == "вверх":
                move_percent = ((tracker['max_price'] - level.max_price) / level.max_price) * 100
            else:
                move_percent = ((level.min_price - tracker['min_price']) / level.min_price) * 100
            
            if move_percent >= 1.0:  # подтвержденный пробой минимум на 1%
                del self.tracked_breakouts[key]
                return {
                    'type': 'confirmed_breakout',
                    'level': level,
                    'direction': breakout_direction,
                    'target': target_direction,
                    'move_percent': move_percent,
                    'message': f"✅ ПРОБОЙ {breakout_direction} на {level.tf} ПОДТВЕРЖДЕН! (+{move_percent:.1f}%)",
                    'confidence': level.strength * 1.3
                }
        
        return None

# ============== АНАЛИЗАТОР ФИБОНАЧЧИ ==============

class FibonacciAnalyzer:
    """
    Анализ уровней Фибоначчи с продвинутой логикой:
    - Точки A и B определяются по паттерну 3 откатных свечей + 1 подтверждающая
    - Автоматическое перестроение при пробое -0.618
    - Смена направления при пробое 1.0
    - Поддержка разных таймфреймов с возможностью отключения
    """
    
    def __init__(self, settings: Dict = None):
        from config import FIBONACCI_SETTINGS
        self.settings = settings or FIBONACCI_SETTINGS
        
        # Уровни Фибоначчи
        self.levels = self.settings.get('levels', {
            'retracement': [0.236, 0.382, 0.5, 0.618, 0.786, 0.86],
            'extension': [-0.18, -0.27, -0.618]
        })
        
        # Все уровни (для расчетов)
        self.all_levels = self.levels['retracement'] + self.levels['extension'] + [1.0, 0]
        
        # Зоны
        self.zones = self.settings.get('zones', {
            'accumulation': [1, 0.86, 0.786, 0.618],
            'correction': [0, -0.18, -0.27, -0.618]
        })
        
        # Сила уровней
        self.level_strength = self.settings.get('level_strength', {
            0.618: 95, 0.786: 90, 0.86: 85, 0.5: 80,
            0.382: 70, 0.236: 65, -0.27: 85, -0.618: 95, -0.18: 75, 1.0: 100, 0: 60
        })
        
        self.lookback_candles = self.settings.get('lookback_candles', 3)
        self.min_distance = self.settings.get('min_distance_pct', 0.5)
        self.touch_tolerance = self.settings.get('touch_tolerance', 0.003)
        
        # Какие ТФ анализировать
        self.enabled_timeframes = {
            tf: cfg.get('enabled', True) 
            for tf, cfg in self.settings.get('timeframes', {}).items()
        }
        
        # Веса ТФ
        self.tf_weights = self.settings.get('weights', {
            'monthly': 3.0, 'weekly': 2.5, 'daily': 2.0,
            'four_hourly': 1.8, 'hourly': 1.5, '30m': 1.2, 'current': 1.0
        })
        
        # Словари для перевода
        self.tf_short = {
            'monthly': '1М', 'weekly': '1н', 'daily': '1д',
            'four_hourly': '4ч', 'hourly': '1ч', '30m': '30м', 'current': '15м'
        }
        
        logger.info("✅ FibonacciAnalyzer инициализирован (продвинутая версия)")
    
    def find_three_candle_pattern(self, df: pd.DataFrame, direction: str) -> Optional[Dict]:
        """
        Поиск паттерна: 3 откатные свечи + 1 подтверждающая
        
        direction: 'up' (ищем максимум) или 'down' (ищем минимум)
        
        Возвращает: {'price': цена, 'index': индекс, 'type': 'bullish'/'bearish'}
        """
        if direction == 'up':
            # Ищем 3 красные откатные свечи (каждая ниже предыдущей по закрытию)
            # + 1 зеленая свеча слева, которая закрылась выше
            for i in range(self.lookback_candles + 2, len(df)):
                # Проверяем 3 откатные свечи
                red1 = df['close'].iloc[i-3] < df['open'].iloc[i-3]  # красная
                red2 = df['close'].iloc[i-2] < df['open'].iloc[i-2]
                red3 = df['close'].iloc[i-1] < df['open'].iloc[i-1]
                
                if not (red1 and red2 and red3):
                    continue
                
                # Проверяем, что каждая ниже предыдущей по закрытию
                if not (df['close'].iloc[i-2] < df['close'].iloc[i-3] and
                        df['close'].iloc[i-1] < df['close'].iloc[i-2]):
                    continue
                
                # Ищем зеленую подтверждающую свечу слева
                green = df['close'].iloc[i-4] > df['open'].iloc[i-4]
                if not green:
                    continue
                
                # Берем максимум из зеленой свечи (с учетом тени)
                green_high = df['high'].iloc[i-4]
                return {
                    'price': green_high,
                    'index': i-4,
                    'type': 'bullish'
                }
        else:  # direction == 'down'
            # Ищем 3 зеленые откатные свечи (каждая выше предыдущей по закрытию)
            # + 1 красная свеча слева, которая закрылась ниже
            for i in range(self.lookback_candles + 2, len(df)):
                # Проверяем 3 откатные свечи
                green1 = df['close'].iloc[i-3] > df['open'].iloc[i-3]
                green2 = df['close'].iloc[i-2] > df['open'].iloc[i-2]
                green3 = df['close'].iloc[i-1] > df['open'].iloc[i-1]
                
                if not (green1 and green2 and green3):
                    continue
                
                # Проверяем, что каждая выше предыдущей по закрытию
                if not (df['close'].iloc[i-2] > df['close'].iloc[i-3] and
                        df['close'].iloc[i-1] > df['close'].iloc[i-2]):
                    continue
                
                # Ищем красную подтверждающую свечу слева
                red = df['close'].iloc[i-4] < df['open'].iloc[i-4]
                if not red:
                    continue
                
                # Берем минимум из красной свечи (с учетом тени)
                red_low = df['low'].iloc[i-4]
                return {
                    'price': red_low,
                    'index': i-4,
                    'type': 'bearish'
                }
        
        return None
    
    def find_swing_point(self, df: pd.DataFrame, direction: str, start_idx: int = None) -> Optional[Dict]:
        """
        Поиск точки A (ближайший минимум/максимум)
        direction: 'up' — ищем минимум, 'down' — ищем максимум
        """
        window = 5
        if start_idx is None:
            start_idx = len(df) - 1
        
        end_idx = max(0, start_idx - 100)
        
        if direction == 'up':
            # Ищем локальный минимум
            for i in range(start_idx, end_idx, -1):
                if i < window or i >= len(df) - window:
                    continue
                is_swing = all(
                    df['low'].iloc[i] < df['low'].iloc[j]
                    for j in range(i - window, i + window + 1) if j != i and 0 <= j < len(df)
                )
                if is_swing:
                    return {'price': df['low'].iloc[i], 'index': i, 'type': 'bullish'}
        else:
            # Ищем локальный максимум
            for i in range(start_idx, end_idx, -1):
                if i < window or i >= len(df) - window:
                    continue
                is_swing = all(
                    df['high'].iloc[i] > df['high'].iloc[j]
                    for j in range(i - window, i + window + 1) if j != i and 0 <= j < len(df)
                )
                if is_swing:
                    return {'price': df['high'].iloc[i], 'index': i, 'type': 'bearish'}
        
        return None
    
    def calculate_fib_levels(self, point_a: float, point_b: float) -> Dict:
        """Расчет всех уровней Фибоначчи"""
        diff = point_b - point_a
        levels = {}
        
        # Ретрейсмент уровни (между A и B)
        for level in self.levels['retracement']:
            price = point_b - diff * level
            levels[f'{level:.3f}'] = {
                'price': price,
                'type': 'retracement',
                'level': level,
                'description': f'{level*100:.1f}%'
            }
        
        # Уровень 1.0 (A)
        levels['1.000'] = {
            'price': point_a,
            'type': 'retracement',
            'level': 1.0,
            'description': '100%'
        }
        
        # Уровень 0 (B)
        levels['0.000'] = {
            'price': point_b,
            'type': 'retracement',
            'level': 0,
            'description': '0%'
        }
        
        # Расширения (за точкой B)
        for level in self.levels['extension']:
            price = point_b + diff * level
            levels[f'{level:.3f}'] = {
                'price': price,
                'type': 'extension',
                'level': level,
                'description': f'{level*100:.1f}%'
            }
        
        return levels
    
    def check_price_reaction(self, current_price: float, levels: Dict) -> List[Dict]:
        """Проверка реакции цены на уровни Фибоначчи"""
        reactions = []
        
        for key, level_data in levels.items():
            level_price = level_data['price']
            distance = abs(current_price - level_price) / current_price * 100
            
            if distance < self.min_distance:
                level_val = level_data['level']
                strength = self.level_strength.get(level_val, 50)
                
                direction = 'support' if current_price < level_price else 'resistance'
                direction_ru = 'поддержка' if direction == 'support' else 'сопротивление'
                
                reactions.append({
                    'level': key,
                    'price': level_price,
                    'strength': strength,
                    'description': f"{level_data['description']}",
                    'direction': direction,
                    'direction_ru': direction_ru,
                    'level_val': level_val
                })
        
        return reactions
    
    def analyze(self, df: pd.DataFrame, timeframe: str = 'current') -> Dict:
        """Анализ Фибоначчи на таймфрейме"""
        result = {
            'has_signal': False, 
            'signals': [], 
            'strength': 0, 
            'levels': {}, 
            'timeframe': timeframe,
            'trend_direction': None
        }
        
        # Пропускаем, если ТФ отключен
        if not self.enabled_timeframes.get(timeframe, True):
            return result
        
        # Определяем направление по EMA 9/21
        if 'ema_9' in df.columns and 'ema_21' in df.columns:
            if df['ema_9'].iloc[-1] > df['ema_21'].iloc[-1]:
                trend = 'up'
            else:
                trend = 'down'
        else:
            trend = 'up' if df['close'].iloc[-1] > df['close'].iloc[-20] else 'down'
        
        # Ищем точку B по паттерну
        pattern = self.find_three_candle_pattern(df, trend)
        if not pattern:
            return result
        
        point_b = pattern['price']
        point_b_idx = pattern['index']
        
        # Ищем точку A (ближайший минимум/максимум перед точкой B)
        point_a_data = self.find_swing_point(df, trend, point_b_idx)
        if not point_a_data:
            return result
        
        point_a = point_a_data['price']
        
        # Проверяем направление: для восходящего тренда A < B, для нисходящего A > B
        is_bullish = point_a < point_b
        if (trend == 'up' and not is_bullish) or (trend == 'down' and is_bullish):
            return result
        
        # Рассчитываем уровни
        levels = self.calculate_fib_levels(point_a, point_b)
        
        # Проверяем реакцию цены
        current_price = df['close'].iloc[-1]
        reactions = self.check_price_reaction(current_price, levels)
        
        for r in reactions:
            result['has_signal'] = True
            tf_short = self.tf_short.get(timeframe, timeframe)
            
            # Формируем сигнал с ценой
            signal_text = (f"Фибо {tf_short}: {r['description']} "
                          f"({r['direction_ru']} {r['price']:.4f})")
            result['signals'].append(signal_text)
            result['strength'] = max(result['strength'], r['strength'])
            result['levels'] = levels
            result['trend_direction'] = 'up' if is_bullish else 'down'
        
        return result
    
    def analyze_multi_timeframe(self, dataframes: Dict[str, pd.DataFrame]) -> Dict:
        """Мультитаймфреймовый анализ Фибоначчи"""
        result = {
            'has_confluence': False,
            'signals': [],
            'strength': 0,
            'levels': {},
            'trend_directions': {}
        }
        
        tf_priority = ['monthly', 'weekly', 'daily', 'four_hourly', 'hourly', '30m', 'current']
        
        for tf_name in tf_priority:
            if tf_name not in dataframes or dataframes[tf_name] is None:
                continue
            
            df = dataframes[tf_name]
            tf_result = self.analyze(df, tf_name)
            
            if tf_result['has_signal']:
                weight = self.tf_weights.get(tf_name, 1.0)
                
                result['signals'].extend(tf_result['signals'])
                result['strength'] += tf_result['strength'] * weight
                result['levels'][tf_name] = tf_result['levels']
                result['trend_directions'][tf_name] = tf_result['trend_direction']
                result['has_confluence'] = True
        
        if result['strength'] > 100:
            result['strength'] = 100
        
        return result

# ============== АНАЛИЗАТОР VOLUME PROFILE ==============

class VolumeProfileAnalyzer:
    """Анализ Volume Profile для определения ключевых уровней"""
    
    def __init__(self, settings: Dict = None):
        self.settings = settings or VOLUME_PROFILE_SETTINGS
        self.lookback = self.settings.get('lookback_bars', 100)
        self.va_pct = self.settings.get('value_area_pct', 70)
    
    def calculate_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Расчет Volume Profile"""
        if df is None or len(df) < 20:
            return {}
        
        recent = df.tail(self.lookback).copy()
        price_precision = 6 if recent['close'].max() < 0.1 else 4 if recent['close'].max() < 10 else 2
        
        volume_by_price = {}
        for _, row in recent.iterrows():
            price_step = 10 ** (-price_precision)
            price_levels = np.arange(round(row['low'], price_precision), round(row['high'], price_precision) + price_step, price_step)
            vol_per_level = row['volume'] / len(price_levels) if len(price_levels) > 0 else 0
            for price in price_levels:
                price_key = round(price, price_precision)
                volume_by_price[price_key] = volume_by_price.get(price_key, 0) + vol_per_level
        
        if not volume_by_price:
            return {}
        
        sorted_prices = sorted(volume_by_price.keys())
        volumes = [volume_by_price[p] for p in sorted_prices]
        max_vol_idx = np.argmax(volumes)
        poc_price = sorted_prices[max_vol_idx]
        
        total_volume = sum(volumes)
        price_vol_pairs = list(zip(sorted_prices, volumes))
        price_vol_pairs.sort(key=lambda x: x[1], reverse=True)
        
        cum_vol, value_area_prices = 0, []
        for price, vol in price_vol_pairs:
            cum_vol += vol
            value_area_prices.append(price)
            if cum_vol >= total_volume * self.va_pct / 100:
                break
        
        val, vah = min(value_area_prices), max(value_area_prices)
        avg_volume = total_volume / len(volumes)
        hvn_threshold = avg_volume * self.settings.get('min_hvn_strength', 2.0)
        hvn_levels = [p for p, v in zip(sorted_prices, volumes) if v > hvn_threshold]
        
        return {
            'poc': poc_price,
            'val': val,
            'vah': vah,
            'hvn': hvn_levels[:5],
            'total_volume': total_volume
        }
    
    def check_price_reaction(self, current_price: float, vp_data: Dict) -> List[Dict]:
        """Проверка реакции цены на уровни Volume Profile"""
        reactions, distance = [], self.settings.get('confluence_distance', 0.5)
        if not vp_data:
            return reactions
        
        poc_dist = abs(current_price - vp_data['poc']) / current_price * 100
        if poc_dist < distance:
            reactions.append({'level': 'POC', 'strength': 90, 'description': f"Цена у POC ({vp_data['poc']:.2f})"})
        
        val_dist = abs(current_price - vp_data['val']) / current_price * 100
        if val_dist < distance:
            reactions.append({'level': 'VAL', 'strength': 75, 'description': f"Цена у VAL ({vp_data['val']:.2f})"})
        
        vah_dist = abs(current_price - vp_data['vah']) / current_price * 100
        if vah_dist < distance:
            reactions.append({'level': 'VAH', 'strength': 75, 'description': f"Цена у VAH ({vp_data['vah']:.2f})"})
        
        for hvn in vp_data['hvn'][:1]:
            hvn_dist = abs(current_price - hvn) / current_price * 100
            if hvn_dist < distance:
                reactions.append({'level': 'HVN', 'strength': 80, 'description': f"Цена в зоне HVN ({hvn:.2f})"})
                break
        
        return reactions
    
    def analyze_multi_timeframe(self, dataframes: Dict[str, pd.DataFrame]) -> Dict:
        """Мультитаймфреймовый анализ Volume Profile"""
        result = {'has_confluence': False, 'signals': [], 'strength': 0, 'levels': {}}
        target_tfs = self.settings.get('timeframes', ['daily', 'weekly', 'monthly'])
        
        for tf_name in target_tfs:
            if tf_name not in dataframes or dataframes[tf_name] is None:
                continue
            
            df = dataframes[tf_name]
            vp_data = self.calculate_volume_profile(df)
            if vp_data:
                reactions = self.check_price_reaction(df['close'].iloc[-1], vp_data)
                weight = 3.0 if tf_name == 'monthly' else 2.5 if tf_name == 'weekly' else 2.0 if tf_name == 'daily' else 1.0
                
                for r in reactions:
                    result['has_confluence'] = True
                    result['signals'].append(f"📊 {tf_name}: {r['description']}")
                    result['strength'] += r['strength'] * weight
                    result['levels'][tf_name] = vp_data
        
        if result['strength'] > 100:
            result['strength'] = 100
        
        return result

# ============== АНАЛИЗАТОР ИМБАЛАНСОВ ==============

class ImbalanceAnalyzer:
    def __init__(self, settings: Dict = None):
        self.settings = settings or IMBALANCE_SETTINGS
        self.threshold_buy = self.settings.get('threshold_buy', 0.3)
        self.threshold_sell = self.settings.get('threshold_sell', -0.3)
        
    def analyze(self, dataframes: Dict[str, pd.DataFrame]) -> Dict:
        result = {
            'has_imbalance': False,
            'signals': [],
            'strength': 0
        }
        
        for tf_name, df in dataframes.items():
            if df is None or df.empty:
                continue
            
            df['buy_volume'] = np.where(
                df['close'] > df['open'],
                df['volume'] * 0.7,
                df['volume'] * 0.3
            )
            df['sell_volume'] = df['volume'] - df['buy_volume']
            df['imbalance'] = (df['buy_volume'] - df['sell_volume']) / df['volume']
            
            last_imbalance = df['imbalance'].iloc[-1]
            
            if last_imbalance > self.threshold_buy:
                result['has_imbalance'] = True
                result['signals'].append(f"Бычий имбаланс на {tf_name}")
                result['strength'] = max(result['strength'], abs(last_imbalance) * 100)
            elif last_imbalance < self.threshold_sell:
                result['has_imbalance'] = True
                result['signals'].append(f"Медвежий имбаланс на {tf_name}")
                result['strength'] = max(result['strength'], abs(last_imbalance) * 100)
        
        return result

# ============== АНАЛИЗАТОР ЛИКВИДНОСТИ ==============

class LiquidityAnalyzer:
    def __init__(self, settings: Dict = None):
        self.settings = settings or LIQUIDITY_SETTINGS
    
    def analyze(self, symbol: str, df: pd.DataFrame) -> Dict:
        result = {
            'has_signal': False,
            'signals': [],
            'strength': 0
        }
        return result

# ============== БАЗОВЫЙ КЛАСС ДЛЯ БИРЖ ==============

class BaseExchangeFetcher:
    def __init__(self, name: str):
        self.name = name
    
    async def fetch_all_pairs(self) -> List[str]:
        return []
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
        return None
    
    async def fetch_funding_rate(self, symbol: str) -> Optional[float]:
        return 0.0
    
    async def fetch_ticker(self, symbol: str) -> Dict:
        return {}
    
    async def fetch_contract_info(self, symbol: str) -> Dict:
        return {
            'max_leverage': 100,
            'min_amount': 5.0,
            'max_amount': 2_000_000
        }
    
    async def fetch_open_interest(self, symbol: str) -> Optional[float]:
        return None
    
    async def close(self):
        pass

# ============== BINGX FUTURES ==============

class BingxFetcher(BaseExchangeFetcher):
    def __init__(self):
        super().__init__("BingX")
        self.exchange = ccxt.bingx({
            'apiKey': os.getenv('BINGX_API_KEY'),
            'secret': os.getenv('BINGX_SECRET_KEY'),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',
                'adjustForTimeDifference': True
            }
        })
        
        # Инициализация кэша плеч
        try:
            from leverage_cache import LeverageCache
            self.leverage_cache = LeverageCache(
                os.getenv('BINGX_API_KEY'),
                os.getenv('BINGX_SECRET_KEY')
            )
            logger.info("✅ BingX Leverages клиент инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ LeverageCache не инициализирован: {e}")
            self.leverage_cache = None
        
        logger.info("✅ BingX Futures инициализирован")
    
    async def fetch_all_pairs(self) -> List[str]:
        try:
            markets = await self.exchange.load_markets()
            usdt_pairs = []
            
            for symbol, market in markets.items():
                if (market['quote'] == 'USDT' and 
                    market['active'] and 
                    market['type'] in ['swap', 'future']):
                    usdt_pairs.append(symbol)
            
            logger.info(f"📊 BingX Futures: загружено {len(usdt_pairs)} фьючерсных пар")
            return usdt_pairs
        except Exception as e:
            logger.error(f"❌ BingX ошибка: {e}")
            return []
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < 20:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            error_msg = str(e)
            if "pause currently" in error_msg or "109415" in error_msg:
                logger.debug(f"⏭️ Пропускаю приостановленную пару {symbol}")
                return None
            if "404" not in error_msg:
                logger.error(f"Ошибка BingX {symbol}: {e}")
            return None
    
    async def fetch_funding_rate(self, symbol: str) -> Optional[float]:
        try:
            funding = await self.exchange.fetch_funding_rate(symbol)
            if funding and 'fundingRate' in funding:
                return funding['fundingRate']
            return 0.0
        except:
            return 0.0
    
    async def fetch_ticker(self, symbol: str) -> Dict:
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return {
                'volume_24h': ticker.get('quoteVolume'),
                'price_change_24h': ticker.get('percentage'),
                'last': ticker.get('last')
            }
        except:
            return {}
    
    async def fetch_contract_info(self, symbol: str) -> Dict:
        """Получение информации о контракте с защитой от некорректных данных"""
        try:
            markets = await self.exchange.load_markets()
            market = markets.get(symbol, {})
            limits = market.get('limits', {})
            
            # Получаем реальное плечо из кэша если доступно
            max_leverage = 100
            if self.leverage_cache:
                try:
                    max_leverage = await self.leverage_cache.get_leverage(symbol)
                except Exception as e:
                    logger.debug(f"Ошибка получения плеча из кэша для {symbol}: {e}")
            
            # Если плечо некорректное - определяем по монете
            if max_leverage > 200 or max_leverage < 1:
                coin = symbol.split('/')[0].upper()
                if coin in ['BTC', 'ETH']:
                    max_leverage = 125
                elif coin in ['BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'DOT', 'LINK']:
                    max_leverage = 75
                elif coin in ['SHIB', 'PEPE', 'DOGS', 'NOT', 'BONK', 'WIF']:
                    max_leverage = 50
                else:
                    max_leverage = 50
            
            # Минимальная сумма входа - защита от дурака
            min_amount = 5.0
            if limits.get('amount'):
                raw_min = limits['amount'].get('min', 5.0)
                # Если минималка слишком большая (>500$) - игнорируем
                if raw_min < 500:
                    min_amount = raw_min
                else:
                    logger.debug(f"Слишком большая минималка {raw_min} для {symbol}, использую 5$")
            
            # Для мемкоинов иногда минималка выше
            coin = symbol.split('/')[0].upper()
            if coin in ['SHIB', 'PEPE', 'DOGS', 'BONK'] and min_amount < 10:
                min_amount = 10.0
            
            # Максимальная сумма
            max_amount = 2_000_000
            if limits.get('amount') and limits['amount'].get('max'):
                raw_max = limits['amount'].get('max', 2_000_000)
                # Ограничиваем разумными пределами
                if raw_max < 50_000_000:
                    max_amount = raw_max
            
            # Получаем лимиты позиций из tiers если доступно
            if self.leverage_cache:
                try:
                    position_limits = await self.leverage_cache.get_position_limits(symbol)
                    if position_limits.get('max_position'):
                        max_amount = min(position_limits['max_position'], max_amount)
                except Exception as e:
                    logger.debug(f"Ошибка получения position limits для {symbol}: {e}")
            
            return {
                'max_leverage': max_leverage,
                'min_amount': round(min_amount, 2),
                'max_amount': int(max_amount),
                'has_data': True
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения контракта {symbol}: {e}")
            
            # Fallback значения с защитой
            coin = symbol.split('/')[0].upper()
            
            if coin in ['BTC', 'ETH']:
                max_leverage = 125
            elif coin in ['BNB', 'SOL', 'XRP', 'ADA']:
                max_leverage = 75
            else:
                max_leverage = 50
            
            return {
                'max_leverage': max_leverage,
                'min_amount': 5.0,
                'max_amount': 2_000_000,
                'has_data': False
            }
    
    async def fetch_open_interest(self, symbol: str) -> Optional[float]:
        try:
            oi = await self.exchange.fetch_open_interest(symbol)
            return oi.get('openInterestAmount', 0)
        except:
            return None
    
    async def close(self):
        await self.exchange.close()

# ============== ДЛЯ ПОДДЕРЖКИ BYBIT, MEXC ==============
class MultiExchangeFetcher:
    """Универсальный фетчер для всех бирж"""
    
    def __init__(self, exchange_id: str, api_key: str = None, api_secret: str = None):
        self.exchange_id = exchange_id
        self.name = exchange_id.capitalize()
        
        exchange_class = getattr(ccxt, exchange_id)
        config = {'enableRateLimit': True, 'options': {'adjustForTimeDifference': True}}
        
        if api_key and api_secret:
            config['apiKey'] = api_key
            config['secret'] = api_secret
        
        self.exchange = exchange_class(config)
        
        if exchange_id == 'mexc':
            self.exchange.options['defaultType'] = 'future'
        elif exchange_id == 'bybit':
            self.exchange.options['defaultType'] = 'linear'
    
    async def fetch_all_pairs(self) -> List[str]:
        await self.exchange.load_markets()
        usdt_pairs = []
        for symbol, market in self.exchange.markets.items():
            if market['quote'] == 'USDT' and market['active'] and market['type'] in ['swap', 'future', 'linear']:
                usdt_pairs.append(symbol)
        return usdt_pairs
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> Optional[pd.DataFrame]:
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv or len(ohlcv) < 20:
                return None
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            return None
    
    async def fetch_funding_rate(self, symbol: str) -> Optional[float]:
        try:
            funding = await self.exchange.fetch_funding_rate(symbol)
            return funding.get('fundingRate', 0.0) if funding else 0.0
        except:
            return 0.0
    
    async def fetch_ticker(self, symbol: str) -> Dict:
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return {'volume_24h': ticker.get('quoteVolume'), 'price_change_24h': ticker.get('percentage')}
        except:
            return {}
    
    async def fetch_contract_info(self, symbol: str) -> Dict:
        return {'max_leverage': 100, 'min_amount': 5.0, 'max_amount': 2_000_000}
    
    async def close(self):
        await self.exchange.close()

class SMCFvgAnalyzer:
    """
    Анализатор Fair Value Gaps как в индикаторе Smart Money Concepts
    """    
    def format_price(self, price):
        if price < 0.0001:
            return f"{price:.8f}".rstrip('0').rstrip('.')
        elif price < 0.001:
            return f"{price:.6f}".rstrip('0').rstrip('.')
        elif price < 0.01:
            return f"{price:.5f}".rstrip('0').rstrip('.')
        elif price < 0.1:
            return f"{price:.4f}".rstrip('0').rstrip('.')
        elif price < 1:
            return f"{price:.3f}".rstrip('0').rstrip('.')
        else:
            return f"{price:.2f}".rstrip('0').rstrip('.')

    def __init__(self, settings: Dict = None):
        from config import FVG_SETTINGS
        self.settings = settings or FVG_SETTINGS
    
    def analyze(self, df: pd.DataFrame, tf_name: str) -> Dict:
        """
        Анализ FVG на одном таймфрейме
        """
        result = {
            'has_fvg': False,
            'zones': [],
            'signals': [],
            'strength': 0
        }
        
        mode = self.settings.get('mode', 'advanced')
        min_gap_size = self.settings.get('min_gap_size_pct', 0.3)
        
        # Расчёт автоматического порога (только для advanced режима)
        threshold = 0
        if mode == 'advanced' and self.settings.get('use_threshold', True):
            lookback = self.settings.get('threshold_lookback', 50)
            delta_changes = (df['close'] - df['open']).abs()
            cumulative_delta = delta_changes.rolling(lookback).sum() / lookback
            multiplier = self.settings.get('threshold_multiplier', 0.5)
            threshold = cumulative_delta.iloc[-1] * multiplier if len(cumulative_delta) > 0 else 0
        
        # Поиск FVG
        for i in range(2, len(df) - 1):
            candle1 = df.iloc[i-2]
            candle2 = df.iloc[i-1]
            candle3 = df.iloc[i]
            
            current_close = candle3['close']
            prev_close = candle2['close']
            
            # Процент изменения свечи (barDeltaPercent)
            bar_delta = abs(current_close - prev_close) / prev_close * 100 if prev_close > 0 else 0
            
            # Бычий FVG
            if candle3['low'] > candle1['high']:
                # Применяем фильтры
                if not self._should_include_fvg(mode, bar_delta, threshold, current_close, candle1['high']):
                    continue
                
                gap_size = (candle3['low'] - candle1['high']) / candle1['high'] * 100
                
                if gap_size < min_gap_size:
                    continue
                
                zone = {
                    'type': 'bullish',
                    'min': candle1['high'],
                    'max': candle3['low'],
                    'size': gap_size,
                    'strength': min(100, gap_size * 20),
                    'confirmed': mode == 'advanced',
                    'tf': tf_name,                    
                    'description': f"📈 FVG ({tf_name}) бычий: {self.format_price(candle1['high'])}-{self.format_price(candle3['low'])} ({gap_size:.2f}%)"
                }
                result['zones'].append(zone)
                result['has_fvg'] = True
            
            # Медвежий FVG
            elif candle3['high'] < candle1['low']:
                if not self._should_include_fvg(mode, bar_delta, threshold, current_close, candle1['low'], is_bullish=False):
                    continue
                
                gap_size = (candle1['low'] - candle3['high']) / candle3['high'] * 100
                
                if gap_size < min_gap_size:
                    continue
                
                zone = {
                    'type': 'bearish',
                    'min': candle3['high'],
                    'max': candle1['low'],
                    'size': gap_size,
                    'strength': min(100, gap_size * 20),
                    'confirmed': mode == 'advanced',
                    'tf': tf_name,
                    'description': f"📉 FVG ({tf_name}) медвежий: {self.format_price(candle3['high'])}-{self.format_price(candle1['low'])} ({gap_size:.2f}%)"
                }
                result['zones'].append(zone)
                result['has_fvg'] = True
        
        # Сортируем по расстоянию до текущей цены
        current_price = df['close'].iloc[-1]
        max_distance = self.settings.get('max_distance_pct', 15.0)
        
        filtered_zones = []
        for zone in result['zones']:
            if current_price < zone['min']:
                distance = (zone['min'] - current_price) / current_price * 100
            elif current_price > zone['max']:
                distance = (current_price - zone['max']) / current_price * 100
            else:
                distance = 0
            
            zone['distance'] = distance
            
            if distance <= max_distance:
                filtered_zones.append(zone)
                result['signals'].append(zone['description'])
        
        result['zones'] = filtered_zones
        result['zones'].sort(key=lambda x: x['distance'])
        
        if result['zones']:
            result['strength'] = sum(z['strength'] for z in result['zones']) / len(result['zones'])
        
        return result
    
    def _should_include_fvg(self, mode: str, bar_delta: float, threshold: float, 
                            current_close: float, level_price: float, is_bullish: bool = True) -> bool:
        """
        Проверка фильтров для FVG
        """
        if mode != 'advanced':
            return True
        
        # Фильтр по bar_delta (проценту изменения свечи)
        if self.settings.get('use_bar_delta_filter', True):
            if bar_delta < threshold:
                return False
        
        # Фильтр по подтверждению закрытием
        if self.settings.get('use_close_confirmation', True):
            if is_bullish:
                if current_close <= level_price:
                    return False
            else:
                if current_close >= level_price:
                    return False
        
        return True
    
    def analyze_multi_timeframe(self, dataframes: Dict[str, pd.DataFrame]) -> Dict:
        """
        Анализ FVG на всех таймфреймах
        """
        result = {
            'has_fvg': False,
            'zones': [],
            'signals': [],
            'strength': 0,
            'zones_by_tf': {}
        }
        
        timeframes = self.settings.get('timeframes', ['15m', '1h', '4h'])
        
        tf_map = {
            '15m': 'current',
            '1h': 'hourly',
            '4h': 'four_hourly',
            '1d': 'daily',
            '1w': 'weekly'
        }
        
        for tf_display in timeframes:
            tf_key = tf_map.get(tf_display, tf_display)
            if tf_key not in dataframes or dataframes[tf_key] is None:
                continue
            
            df = dataframes[tf_key]
            fvg_result = self.analyze(df, tf_display)
            
            if fvg_result['has_fvg']:
                result['has_fvg'] = True
                result['zones'].extend(fvg_result['zones'])
                result['signals'].extend(fvg_result['signals'])
                result['strength'] = max(result['strength'], fvg_result['strength'])
                result['zones_by_tf'][tf_display] = fvg_result['zones']
        
        # Сортируем все зоны по расстоянию
        result['zones'].sort(key=lambda x: x.get('distance', 999))
        
        # Получаем текущую цену
        current_price = dataframes['current']['close'].iloc[-1]
        
        # 1. Находим FVG, в котором находится цена
        fvg_current = [z for z in result['zones'] if z['min'] <= current_price <= z['max']]
        
        # 2. Находим ближайший FVG сверху и снизу
        fvg_above = [z for z in result['zones'] if z['min'] > current_price]
        fvg_below = [z for z in result['zones'] if z['max'] < current_price]
        
        fvg_above.sort(key=lambda x: x['min'])
        fvg_below.sort(key=lambda x: x['max'], reverse=True)
        
        # 3. Формируем итоговый список (максимум 2 зоны)
        filtered_zones = []
        
        # Сначала добавляем FVG, в котором цена (если есть)
        if fvg_current:
            filtered_zones.append(fvg_current[0])
        
        # Затем добавляем ближайший сверху (если есть и ещё нет 2 зон)
        if fvg_above and len(filtered_zones) < 2:
            filtered_zones.append(fvg_above[0])
        
        # Затем добавляем ближайший снизу (если есть и ещё нет 2 зон)
        if fvg_below and len(filtered_zones) < 2:
            filtered_zones.append(fvg_below[0])
        
        result['zones'] = filtered_zones

        return result

# ============== МУЛЬТИТАЙМФРЕЙМ АНАЛИЗАТОР ==============

class MultiTimeframeAnalyzer:
    def __init__(self):
        self.divergence = DivergenceAnalyzer() if FEATURES['advanced']['divergence'] else None
        self.smc = SmartMoneyAnalyzer(SMC_SETTINGS) if FEATURES['advanced']['smart_money'] else None
        self.fractal = FractalAnalyzer(FRACTAL_SETTINGS) if FEATURES['advanced']['fractals'] else None
        self.fibonacci = None
        self.volume_profile = None
        self.accumulation = None
        self.imbalance = ImbalanceAnalyzer(IMBALANCE_SETTINGS) if FEATURES['advanced']['imbalance'] else None
        self.breakout_tracker = BreakoutTracker()
        # Инициализация истории Фибоначчи
        self.fib_history = FibHistoryTracker()
        logger.info("✅ FibHistoryTracker добавлен в MultiTimeframeAnalyzer")

        # Словарь для перевода таймфреймов
        self.tf_translation = {
            'monthly': 'месячный',
            'weekly': 'недельный',
            'daily': 'дневной',
            'hourly': 'часовой',
            'current': 'текущий'
        }
        self.stop_hunt_detector = StopHuntDetector()
        self.liquidity_zone_detector = LiquidityZoneDetector()
        self.pattern_analyzer = PatternAnalyzer()
        self.smc_fvg_analyzer = SMCFvgAnalyzer()
        self.smart_money = SmartMoneyAnalyzer()

    def set_fibonacci(self, fib_analyzer):
        self.fibonacci = fib_analyzer
    
    def set_volume_profile(self, vp_analyzer):
        self.volume_profile = vp_analyzer
    
    def set_accumulation(self, acc_analyzer):
        self.accumulation = acc_analyzer
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет всех технических индикаторов"""
        df['rsi'] = calculate_rsi(df['close'], INDICATOR_SETTINGS['rsi_period'])
        
        macd_line, signal_line, hist = calculate_macd(
            df['close'], 
            INDICATOR_SETTINGS['macd_fast'],
            INDICATOR_SETTINGS['macd_slow'],
            INDICATOR_SETTINGS['macd_signal']
        )
        df['MACD_12_26_9'] = macd_line
        df['MACDs_12_26_9'] = signal_line
        df['MACDh_12_26_9'] = hist
        
        for period in INDICATOR_SETTINGS['ema_periods']:
            df[f'ema_{period}'] = calculate_ema(df['close'], period)
        
        df['sma_50'] = calculate_sma(df['close'], 50)
        df['sma_200'] = calculate_sma(df['close'], 200)
        
        sma, upper, lower = calculate_bollinger_bands(
            df['close'], 
            INDICATOR_SETTINGS['bollinger_period'],
            INDICATOR_SETTINGS['bollinger_std']
        )
        df['BBL_20_2.0'] = lower
        df['BBM_20_2.0'] = sma
        df['BBU_20_2.0'] = upper
        
        df['atr'] = calculate_atr(df['high'], df['low'], df['close'], INDICATOR_SETTINGS['atr_period'])
        df['volume_sma'] = calculate_sma(df['volume'], INDICATOR_SETTINGS['volume_sma_period'])
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        if FEATURES['advanced']['vwap']:
            df['vwap'] = calculate_vwap(df)
        
        return df
    
    def analyze_timeframe_alignment(self, dataframes: Dict[str, pd.DataFrame]) -> Dict:        
        """Анализ согласованности трендов на разных таймфреймах"""
        alignment = {
            'trend_alignment': 0,
            'signals': [],
            'trends': {},
            'trend_strength': {},
            'current_trend': None,
            'hourly_trend': None,
            'four_hourly_trend': None,
            'daily_trend': None,
            'weekly_trend': None,
            'monthly_trend': None,
        }
        
        # Словарь для перевода
        tf_names = {
            '1m': '1м',
            '3m': '3м',
            '5m': '5м',
            'current': '15м',
            '30m': '30м',
            'hourly': '1ч',
            'four_hourly': '4ч',
            'daily': '1д',
            'weekly': '1н',
            'monthly': '1м'
        }
        
        # Собираем все таймфреймы в правильном порядке
        tf_order = ['1m', '3m', '5m', 'current', '30m', 'hourly', 'four_hourly', 'daily', 'weekly', 'monthly']
        
        trends_list = []
        minor_bullish = 0
        minor_bearish = 0

        from config import TREND_STRENGTH_SETTINGS
        strength_settings = TREND_STRENGTH_SETTINGS
        
        for tf_name in tf_order:
            if tf_name not in dataframes or dataframes[tf_name] is None or dataframes[tf_name].empty:
                continue
            
            df = dataframes[tf_name]
            last = df.iloc[-1]
            
            # Определяем тренд по EMA 9 и 21
            if pd.notna(last.get('ema_9')) and pd.notna(last.get('ema_21')):
                trend = 'ВОСХОДЯЩИЙ' if last['ema_9'] > last['ema_21'] else 'НИСХОДЯЩИЙ'
                alignment['trends'][tf_name] = trend
                
                # ===== ОЦЕНКА СИЛЫ ТРЕНДА =====
                strength_score = 0
                strength_desc = []
                
                if strength_settings.get('enabled', True):
                    # 1. Оценка по EMA 50
                    if strength_settings.get('use_ema_50', True) and pd.notna(last.get('ema_50')):
                        if trend == 'ВОСХОДЯЩИЙ':
                            if last['ema_9'] > last['ema_21'] > last['ema_50']:
                                strength_score += strength_settings['weights']['strong_trend']
                                strength_desc.append("сильный")
                            elif last['ema_9'] > last['ema_21'] and last['ema_21'] < last['ema_50']:
                                strength_score += strength_settings['weights']['weak_trend']
                                strength_desc.append("слабый")
                        else:  # НИСХОДЯЩИЙ
                            if last['ema_9'] < last['ema_21'] < last['ema_50']:
                                strength_score += strength_settings['weights']['strong_trend']
                                strength_desc.append("сильный")
                            elif last['ema_9'] < last['ema_21'] and last['ema_21'] > last['ema_50']:
                                strength_score += strength_settings['weights']['weak_trend']
                                strength_desc.append("слабый")
                    
                    # 2. Оценка по объему
                    if strength_settings.get('use_volume', True) and pd.notna(last.get('volume_ratio')):
                        if last['volume_ratio'] > strength_settings['volume_ratio_threshold']:
                            strength_score += strength_settings['weights']['volume_confirmation']
                            strength_desc.append(f"объем x{last['volume_ratio']:.1f}")
                    
                    # 3. Оценка по RSI
                    if strength_settings.get('use_rsi', True) and pd.notna(last.get('rsi')):
                        if trend == 'ВОСХОДЯЩИЙ' and last['rsi'] < strength_settings['rsi_oversold']:
                            strength_score += strength_settings['weights']['rsi_extreme']
                            strength_desc.append(f"RSI перепродан ({last['rsi']:.1f})")
                        elif trend == 'НИСХОДЯЩИЙ' and last['rsi'] > strength_settings['rsi_overbought']:
                            strength_score += strength_settings['weights']['rsi_extreme']
                            strength_desc.append(f"RSI перекуплен ({last['rsi']:.1f})")
                
                alignment['trend_strength'][tf_name] = {
                    'score': strength_score,
                    'desc': ', '.join(strength_desc) if strength_desc else 'обычный'
                }

                # Сохраняем в отдельные поля для совместимости
                if tf_name == 'current':
                    alignment['current_trend'] = trend
                elif tf_name == 'hourly':
                    alignment['hourly_trend'] = trend
                elif tf_name == 'four_hourly':
                    alignment['four_hourly_trend'] = trend
                elif tf_name == 'daily':
                    alignment['daily_trend'] = trend
                elif tf_name == 'weekly':
                    alignment['weekly_trend'] = trend
                elif tf_name == 'monthly':
                    alignment['monthly_trend'] = trend
                
                # Для младших ТФ (1м, 3м, 5м) — считаем для группировки
                if tf_name in ['1m', '3m', '5m']:
                    if trend == 'ВОСХОДЯЩИЙ':
                        minor_bullish += 1
                    else:
                        minor_bearish += 1
                
                # Для остальных ТФ — добавляем в согласованность
                elif tf_name not in ['1m', '3m', '5m']:
                    trends_list.append(trend)
                    
                # Добавляем сигнал для сильных трендов с указанием силы
                if tf_name in ['weekly', 'monthly']:
                    strength = alignment['trend_strength'].get(tf_name, {})
                    strength_text = f" ({strength.get('desc', '')})" if strength.get('desc') and strength.get('desc') != 'обычный' else ""
                    
                    if last['ema_9'] > last['ema_200']:
                        alignment['signals'].append(f"{tf_names[tf_name]} ТРЕНД ВОСХОДЯЩИЙ (выше EMA 200){strength_text}")
                    elif last['ema_9'] < last['ema_200']:
                        alignment['signals'].append(f"{tf_names[tf_name]} ТРЕНД НИСХОДЯЩИЙ (ниже EMA 200){strength_text}")
        
        # Добавляем группировку младших ТФ в причины (без дублирования с 1м)
        # Определяем, какие ТФ входят в группу младших
        minor_tfs_present = []
        if '1m' in alignment['trends']:
            minor_tfs_present.append('1м')
        if '3m' in alignment['trends']:
            minor_tfs_present.append('3м')
        if '5m' in alignment['trends']:
            minor_tfs_present.append('5м')
        
        # Проверяем, есть ли уже отдельная строка для 1м с EMA 200
        has_1m_ema = False
        for s in alignment['signals']:
            if '1м ТРЕНД' in s and 'EMA 200' in s:
                has_1m_ema = True
                break
        
        if minor_bullish > 0 or minor_bearish > 0:
            # Формируем текст только для ТФ, которых нет в отдельной строке с EMA 200
            tfs_to_show = []
            for tf in minor_tfs_present:
                if tf == '1м' and has_1m_ema:
                    continue  # пропускаем 1м, если уже есть в EMA 200 строке
                tfs_to_show.append(tf)
            
            if tfs_to_show:
                tfs_str = ', '.join(tfs_to_show)
                if minor_bullish > minor_bearish:
                    alignment['signals'].append(f"{tfs_str} тренд восходящий (подтверждение)")
                else:
                    alignment['signals'].append(f"{tfs_str} тренд нисходящий (подтверждение)")
        
        # Собираем старшие ТФ с EMA 200 в одну группу
        ema_signals = []
        tf_names_display = {
            'weekly': '1н',
            'monthly': '1м',
            'daily': '1д',
            'four_hourly': '4ч',
            'hourly': '1ч',
            'current': '15м'
        }
        
        for tf_name in ['weekly', 'monthly']:
            if tf_name in alignment['trends']:
                trend = alignment['trends'][tf_name]
                df = dataframes.get(tf_name)
                if df is not None and not df.empty:
                    last = df.iloc[-1]
                    if pd.notna(last.get('ema_9')) and pd.notna(last.get('ema_200')):
                        if trend == 'ВОСХОДЯЩИЙ' and last['ema_9'] > last['ema_200']:
                            ema_signals.append(tf_names_display.get(tf_name, tf_name))
                        elif trend == 'НИСХОДЯЩИЙ' and last['ema_9'] < last['ema_200']:
                            ema_signals.append(tf_names_display.get(tf_name, tf_name))
        
        # Добавляем также 1м если есть EMA 200
        if '1m' in alignment['trends']:
            df = dataframes.get('1m')
            if df is not None and not df.empty:
                last = df.iloc[-1]
                if pd.notna(last.get('ema_9')) and pd.notna(last.get('ema_200')):
                    trend = alignment['trends']['1m']
                    if trend == 'ВОСХОДЯЩИЙ' and last['ema_9'] > last['ema_200']:
                        ema_signals.append('1м')
                    elif trend == 'НИСХОДЯЩИЙ' and last['ema_9'] < last['ema_200']:
                        ema_signals.append('1м')
        
        if ema_signals:
            tfs_str = ', '.join(ema_signals)
            # Определяем направление по старшему ТФ (weekly или monthly)
            main_trend = alignment.get('weekly_trend') or alignment.get('monthly_trend') or alignment.get('daily_trend')
            if main_trend == 'ВОСХОДЯЩИЙ':
                alignment['signals'].append(f"{tfs_str} ТРЕНД ВОСХОДЯЩИЙ (выше EMA 200)")
            elif main_trend == 'НИСХОДЯЩИЙ':
                alignment['signals'].append(f"{tfs_str} ТРЕНД НИСХОДЯЩИЙ (ниже EMA 200)")
        
        # Считаем согласованность (только для основных ТФ)
        if trends_list:
            bullish = trends_list.count('ВОСХОДЯЩИЙ')
            bearish = trends_list.count('НИСХОДЯЩИЙ')
            alignment['trend_alignment'] = (max(bullish, bearish) / len(trends_list)) * 100
            
            # Логируем согласованность
            logger.info(f"  📊 Согласованность трендов: {alignment['trend_alignment']:.1f}% ({len(trends_list)} ТФ)")

            if alignment['trend_alignment'] >= 80 and len(trends_list) >= 3:
                direction = "бычий" if bullish > bearish else "медвежий"
                alignment['signals'].append(f"Тренды согласованы: {alignment['trend_alignment']:.0f}% ({direction}, {len(trends_list)} ТФ)")
        
        return alignment
    
    def check_tf_alignment(self, dataframes: Dict[str, pd.DataFrame], signal_type: str = 'regular') -> Dict:
        """
        Проверка согласованности таймфреймов для принятия решения о сигнале
        signal_type: 'regular', 'pump', 'accumulation'
        """
        result = {
            'status': 'rejected',
            'direction': None,
            'reasons': [],
            'confidence_modifier': 0,
            'percentage': 0,
            'aligned_count': 0,
            'total_count': 0,
            'trend_type': None,           # 'trend' или 'correction' для пампов
        }
        
        settings = TF_ALIGNMENT_SETTINGS
        if not settings.get('enabled', True):
            result['status'] = 'perfect'
            result['percentage'] = 100
            return result
        
        # Выбираем режим и ТФ в зависимости от типа сигнала
        mode = settings.get(f'{signal_type}_mode', 'info')
        tfs_config = settings.get(f'{signal_type}_tfs', settings.get('regular_tfs'))
        
        # Если режим 'off' — выключаем проверку
        if mode == 'off':
            result['status'] = 'perfect'
            result['percentage'] = 100
            return result
        
        # Получаем тренды из ранее проанализированных данных
        alignment = self.analyze_timeframe_alignment(dataframes)
        
        # Собираем все ТФ для анализа
        all_tfs = tfs_config.get('major', []) + tfs_config.get('minor', []) + tfs_config.get('ultra_minor', [])
        all_tfs = list(set(all_tfs))  # убираем дубликаты
        
        logger.info(f"  🔍 Все ТФ для анализа ({signal_type}): {all_tfs}")
        
        # Определяем направление каждого ТФ (только для ТФ с данными)
        tf_directions = []
        available_tfs = 0
        available_tfs_list = []
        for tf in all_tfs:
            trend = alignment.get(f'{tf}_trend')
            if trend:
                available_tfs += 1
                available_tfs_list.append(tf)
                if trend == 'ВОСХОДЯЩИЙ':
                    tf_directions.append('LONG')
                elif trend == 'НИСХОДЯЩИЙ':
                    tf_directions.append('SHORT')
                else:
                    tf_directions.append(None)
            else:
                tf_directions.append(None)

        logger.info(f"  🔍 Доступные ТФ для согласованности ({signal_type}): {available_tfs_list}")
        
        # Если нет данных ни для одного ТФ — возвращаем
        if available_tfs == 0:
            result['status'] = 'perfect'
            result['percentage'] = 100
            return result
        
        # Определяем направление по старшим ТФ (для пампов это важно)
        major_tfs = tfs_config.get('major', [])
        major_directions = []
        for i, tf in enumerate(all_tfs):
            if tf in major_tfs and tf_directions[i]:
                major_directions.append(tf_directions[i])
        
        major_direction = None
        if major_directions:
            bullish = major_directions.count('LONG')
            bearish = major_directions.count('SHORT')
            if bullish > bearish:
                major_direction = 'LONG'
            elif bearish > bullish:
                major_direction = 'SHORT'
        
        # Определяем направление по младшим ТФ (для пампов это импульс)
        minor_tfs = tfs_config.get('minor', []) + tfs_config.get('ultra_minor', [])
        minor_directions = []
        for i, tf in enumerate(all_tfs):
            if tf in minor_tfs and tf_directions[i]:
                minor_directions.append(tf_directions[i])
        
        minor_direction = None
        if minor_directions:
            bullish = minor_directions.count('LONG')
            bearish = minor_directions.count('SHORT')
            if bullish > bearish:
                minor_direction = 'LONG'
            elif bearish > bullish:
                minor_direction = 'SHORT'
        
        # Для памп-дамп определяем тип движения
        if signal_type == 'pump':
            if major_direction and minor_direction and major_direction == minor_direction:
                result['trend_type'] = 'trend'  # памп по тренду
                result['reasons'].append(f"✅ Памп по тренду (старшие ТФ {major_direction})")
            elif major_direction and minor_direction and major_direction != minor_direction:
                result['trend_type'] = 'correction'  # памп против тренда
                result['reasons'].append(f"⚠️ Памп против тренда (старшие {major_direction}, импульс {minor_direction})")
            elif minor_direction and not major_direction:
                result['trend_type'] = 'impulse'  # импульс без тренда
                result['reasons'].append(f"📊 Импульс без явного тренда")
        
        # Для обычных сигналов и накопления — используем старшие ТФ для направления
        if signal_type != 'pump':
            result['direction'] = major_direction
        else:
            # Для пампов направление определяется младшими ТФ (импульс)
            result['direction'] = minor_direction
        
        # Считаем количество согласованных ТФ с направлением
        aligned_count = 0
        if result['direction']:
            for d in tf_directions:
                if d == result['direction']:
                    aligned_count += 1
        
        result['aligned_count'] = aligned_count
        result['total_count'] = available_tfs
        result['percentage'] = round((aligned_count / available_tfs) * 100) if available_tfs > 0 else 0
        
        # Определяем бонус/штраф к уверенности
        if result['percentage'] >= 100:
            result['confidence_modifier'] = settings.get('bonus_perfect', 20)
        elif result['percentage'] >= 66:
            result['confidence_modifier'] = settings.get('bonus_high', 10)
        elif result['percentage'] >= 33:
            result['confidence_modifier'] = settings.get('penalty_low', -5)
        else:
            result['confidence_modifier'] = settings.get('penalty_very_low', -15)
        
        # Добавляем причину о согласованности (только одну строку)
        if settings.get('show_percentage', True):
            if result['percentage'] >= 100:
                result['reasons'].append(f"📊 Согласованность ТФ: {result['percentage']}% ({aligned_count}/{available_tfs}) — идеально")
            elif result['percentage'] >= 66:
                result['reasons'].append(f"📊 Согласованность ТФ: {result['percentage']}% ({aligned_count}/{available_tfs}) — хорошо")
            elif result['percentage'] >= 33:
                result['reasons'].append(f"📊 Согласованность ТФ: {result['percentage']}% ({aligned_count}/{available_tfs}) — средняя")
            else:
                result['reasons'].append(f"📊 Согласованность ТФ: {result['percentage']}% ({aligned_count}/{available_tfs}) — низкая")
        
        # Определяем статус на основе режима
        threshold = settings.get('thresholds', {}).get(mode, 0)
        
        if mode == 'info' and settings.get('send_all_in_info_mode', True):
            result['status'] = 'perfect' if result['percentage'] >= 100 else 'warning'
        elif result['percentage'] >= threshold:
            result['status'] = 'perfect' if result['percentage'] >= 100 else 'warning'
        else:
            result['status'] = 'rejected'
            if mode != 'info':
                result['reasons'].append(f"❌ Сигнал отклонен (согласованность {result['percentage']}% < {threshold}%)")
        
        return result
    
    # ============== Старый метод анализа, не используется сейчас ==============
    def analyze_fvg_multi_timeframe(self, dataframes: Dict[str, pd.DataFrame], current_price: float) -> Dict:
        """
        Анализ FVG на всех таймфреймах с фильтрацией по расстоянию
        """

        result = {'has_fvg': False, 'signals': [], 'strength': 0, 'zones': []}
        all_zones = []
        
        # Максимальное расстояние для отображения в причинах (например, 15%)
        MAX_DISTANCE_PCT = 15.0
        """
        Анализ FVG на всех таймфреймах с фильтрацией для графика
        """
        result = {'has_fvg': False, 'signals': [], 'strength': 0, 'zones': []}
        all_zones = []  # временный список всех найденных зон
        
        # Приоритет таймфреймов
        tf_priority = ['monthly', 'weekly', 'daily', 'four_hourly', 'hourly', 'current']
        
        # Словари для форматирования
        tf_short = {
            'monthly': '1М',
            'weekly': '1н',
            'daily': '1д',
            'four_hourly': '4ч',
            'hourly': '1ч',
            'current': '15м'
        }
        
        tf_weights = {
            'monthly': 4.0,
            'weekly': 3.5,
            'daily': 2.5,
            'four_hourly': 2.0,
            'hourly': 1.5,
            'current': 1.0
        }
        
        tf_names_ru = {
            'monthly': 'месячный',
            'weekly': 'недельный',
            'daily': 'дневной',
            'four_hourly': '4-часовой',
            'hourly': 'часовой',
            'current': '15-минутный'
        }
        
        dir_emoji = {
            'bullish': '📈',
            'bearish': '📉'
        }
        
        # Анализируем каждый таймфрейм
        for tf_name in tf_priority:
            if tf_name not in dataframes or dataframes[tf_name] is None:
                continue
            
            df = dataframes[tf_name]

            if df is None or df.empty or len(df) < 20:
                logger.info(f"    ⏭️ {tf_name}: недостаточно данных ({len(df) if df is not None else 0} свечей)")
                continue
            
            # Создаем временный SMC анализатор для этого ТФ
            smc_temp = SmartMoneyAnalyzer(SMC_SETTINGS)
            fvg_list = smc_temp.find_fair_value_gaps(df)

            logger.info(f"    🔍 {tf_name}: найдено {len(fvg_list)} FVG кандидатов")

            for fvg in fvg_list:
                try:
                    # Проверяем, не закрыта ли зона
                    #if self._is_fvg_closed(df, fvg):
                    #    logger.info(f"    ⏭️ {tf_name} FVG пропущен (закрыт)")
                    #    continue
                    logger.info(f"    ✅ {tf_name} FVG (всегда показываю): {fvg['price_min']:.6f}-{fvg['price_max']:.6f}")

                    # Проверяем, находится ли текущая цена в зоне FVG
                    in_zone = (fvg['price_min'] <= current_price <= fvg['price_max'])
                    
                    # Рассчитываем расстояние до зоны
                    if in_zone:
                        distance_pct = 0
                        distance_text = "в зоне"
                        zone_type = "тест"
                    elif current_price < fvg['price_min']:
                        distance_pct = ((fvg['price_min'] - current_price) / current_price) * 100
                        distance_text = f"выше на {distance_pct:.1f}%"
                        zone_type = "сопротивление сверху"
                    else:  # current_price > fvg['price_max']
                        distance_pct = ((current_price - fvg['price_max']) / current_price) * 100
                        distance_text = f"ниже на {distance_pct:.1f}%"
                        zone_type = "поддержка снизу"
                    
                    # Логируем найденный FVG
                    logger.info(f"    ✅ {tf_name} FVG: {fvg['price_min']:.6f}-{fvg['price_max']:.6f}, {distance_text}")
                    
                    # Форматируем цены зоны
                    if fvg['price_min'] < 0.001:
                        zone_str = f"{fvg['price_min']:.6f}-{fvg['price_max']:.6f}"
                    elif fvg['price_min'] < 0.1:
                        zone_str = f"{fvg['price_min']:.4f}-{fvg['price_max']:.4f}"
                    else:
                        zone_str = f"{fvg['price_min']:.2f}-{fvg['price_max']:.2f}"
                    
                    # Формируем сигнал
                    size_pct = fvg.get('size', 0)
                    tf_ru = tf_names_ru.get(tf_name, tf_name)
                    direction = "бычий" if fvg['type'] == 'bullish' else "медвежий"
                    
                    signal_text = (f"FVG {tf_short[tf_name]}: {zone_str} "
                                f"(размер {size_pct:.2f}% {dir_emoji[fvg['type']]} {zone_type}, {distance_text})")
                    
                    result['has_fvg'] = True
                    result['signals'].append(signal_text)
                    
                    # Сохраняем зону для графиков и анализа
                    all_zones.append({
                        'tf': tf_name,
                        'tf_short': tf_short[tf_name],
                        'tf_ru': tf_ru,
                        'min': fvg['price_min'],
                        'max': fvg['price_max'],
                        'type': fvg['type'],
                        'dir_emoji': dir_emoji[fvg['type']],
                        'size': size_pct,
                        'distance_pct': distance_pct,
                        'in_zone': in_zone,
                        'zone_type': zone_type,
                        'distance_text': distance_text,
                        'weight': tf_weights.get(tf_name, 1.0),
                        'strength': fvg['strength']
                    })
                    
                    # Увеличиваем силу с весом таймфрейма
                    result['strength'] += fvg['strength'] * tf_weights.get(tf_name, 1.0)
                    
                except Exception as e:
                    logger.error(f"    ❌ Ошибка при обработке FVG для {tf_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                        
        # ===== ФИЛЬТРАЦИЯ ПО РАССТОЯНИЮ ДЛЯ ПРИЧИН =====
        MAX_DISTANCE_PCT = 15.0
        DISTANCE_THRESHOLDS = {
            'monthly': 50.0,
            'weekly': 30.0,
            'daily': 20.0,
            'four_hourly': 15.0,
            'hourly': 10.0,
            'current': 5.0
        }

        filtered_zones = []
        for zone in all_zones:
            threshold = DISTANCE_THRESHOLDS.get(zone['tf'], MAX_DISTANCE_PCT)
            
            if zone['distance_pct'] > threshold:
                logger.info(f"    ⏭️ FVG {zone['tf']} пропущен - слишком далеко ({zone['distance_pct']:.1f}% > {threshold}%)")
                continue
            
            filtered_zones.append(zone)
            
            # Добавляем в причины
            signal_text = (f"FVG {zone['tf_short']}: {zone['min']:.4f}-{zone['max']:.4f} "
                        f"(размер {zone['size']:.2f}% {zone['type']}, {zone['distance_text']})")
            result['signals'].append(signal_text)
            result['strength'] += zone['strength'] * zone['weight']

        logger.info(f"  📊 Добавлено {len(result['signals'])} FVG в причины (из {len(all_zones)} найденных)")
                       
        # Ограничиваем силу 100%
        if result['strength'] > 100:
            result['strength'] = 100
        
        return result

    def _is_fvg_closed(self, df: pd.DataFrame, fvg: Dict) -> bool:
        try:
            # Защита от некорректных данных
            if not fvg or 'price_min' not in fvg or 'price_max' not in fvg:
                return True
            
            last_idx = len(df) - 1
            start_idx = max(0, last_idx - 100)
            
            close_count = 0
            for i in range(start_idx, last_idx):
                candle = df.iloc[i]
                
                if fvg['type'] == 'bullish':
                    if candle['close'] < fvg['price_min']:
                        close_count += 1
                        if close_count >= 2:
                            return True
                    else:
                        close_count = 0
                else:
                    if candle['close'] > fvg['price_max']:
                        close_count += 1
                        if close_count >= 2:
                            return True
                    else:
                        close_count = 0
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка в _is_fvg_closed: {e}")
            return True

    def analyze_ema_touch(self, df: pd.DataFrame, last: pd.Series) -> Dict:
        """Анализ касаний цены EMA уровней"""
        result = {'touches': [], 'signals': []}
        
        # Защита от пустых данных
        if df is None or df.empty or last is None:
            return result
        
        for period in [9, 14, 21, 28, 50, 100]:
            ema_col = f'ema_{period}'
            if ema_col not in df.columns:
                continue
            
            ema_value = last.get(ema_col)
            if pd.isna(ema_value):
                continue
            
            price = last.get('close')
            if pd.isna(price):
                continue
            
            distance = abs(price - ema_value) / price * 100
            
            if distance < 0.5:
                result['touches'].append(period)
                
                if price > ema_value:
                    result['signals'].append(f"📊 Цена у EMA {period} (поддержка)")
                else:
                    result['signals'].append(f"📊 Цена у EMA {period} (сопротивление)")
        
        return result

    def analyze_ema_touch_multi_timeframe(self, dataframes: Dict[str, pd.DataFrame], current_price: float) -> Dict:
        """
        Анализ касаний EMA на всех таймфреймах с настройками из конфига
        """
        from config import EMA_TOUCH_SETTINGS, TIMEFRAME_WEIGHTS
        
        result = {'touches': [], 'signals': [], 'strength': 0}
        
        if not EMA_TOUCH_SETTINGS.get('enabled', True):
            return result
        
        periods = EMA_TOUCH_SETTINGS.get('periods', [9, 14, 21, 28, 50, 100])
        distance_threshold = EMA_TOUCH_SETTINGS.get('distance_threshold', 0.5)
        max_signals = EMA_TOUCH_SETTINGS.get('max_signals', 3)
        weights = EMA_TOUCH_SETTINGS.get('weights', {})
        
        # Словарь для перевода таймфреймов
        tf_short = {
            '1m': '1м',
            '3m': '3м',
            '5m': '5м',
            'current': '15м',
            '30m': '30м',
            'hourly': '1ч',
            'four_hourly': '4ч',
            'daily': '1д',
            'weekly': '1н',
            'monthly': '1М'
        }
        
        # Приоритет таймфреймов (от старших к младшим)
        tf_order = ['monthly', 'weekly', 'daily', 'four_hourly', 'hourly', '30m', 'current', '5m', '3m', '1m']
        
        for tf_name in tf_order:
            if tf_name not in dataframes or dataframes[tf_name] is None:
                continue
            
            df = dataframes[tf_name]
            if df.empty:
                continue
            
            last = df.iloc[-1]
            tf_short_name = tf_short.get(tf_name, tf_name)
            weight = weights.get(tf_name, TIMEFRAME_WEIGHTS.get(tf_name, 5))
            
            for period in periods:
                ema_col = f'ema_{period}'
                if ema_col not in df.columns:
                    continue
                
                ema_value = last[ema_col]
                price = last['close']
                distance = abs(price - ema_value) / price * 100
                
                if distance < distance_threshold:
                    if price > ema_value:
                        touch_type = "поддержка"
                        direction_bias = "LONG"
                    else:
                        touch_type = "сопротивление"
                        direction_bias = "SHORT"
                    
                    # Сила касания (чем больше период, тем сильнее)
                    strength = int(weight * (period / 50))
                    
                    result['touches'].append({
                        'tf': tf_short_name,
                        'period': period,
                        'type': touch_type,
                        'distance': distance,
                        'strength': strength,
                        'price': price,
                        'ema': ema_value
                    })
                    
                    signal_text = f"📊 Цена у EMA {period} на {tf_short_name} ({touch_type}, {distance:.1f}%)"
                    result['signals'].append(signal_text)
                    result['strength'] += strength
                    
                    logger.info(f"    📊 {tf_name} EMA {period}: {touch_type} (дистанция {distance:.2f}%)")
        
        # Ограничиваем количество сигналов в причинах
        if len(result['signals']) > max_signals:
            result['signals'] = result['signals'][:max_signals]
        
        # Ограничиваем силу
        if result['strength'] > 100:
            result['strength'] = 100
        
        return result
    
    def calculate_volume_spike(self, df: pd.DataFrame) -> Dict:
        """Поиск свечей с аномальным объемом"""
        from config import VOLUME_ANALYSIS_SETTINGS
        
        settings = VOLUME_ANALYSIS_SETTINGS['spike_detector']
        if not settings['enabled']:
            return {'spike': False}
        
        lookback = settings['lookback']
        threshold = settings['threshold']
        
        if len(df) < lookback + 1:
            return {'spike': False}
        
        last_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].iloc[-lookback-1:-1].mean()
        ratio = last_volume / avg_volume if avg_volume > 0 else 1
        
        if ratio > threshold:
            return {
                'spike': True,
                'ratio': ratio,
                'price': df['close'].iloc[-1],
                'direction': 'UP' if df['close'].iloc[-1] > df['open'].iloc[-1] else 'DOWN'
            }
        return {'spike': False}

    def calculate_volume_dispersion(self, df: pd.DataFrame, hours: int = 2) -> Dict:
        """Расчет дисперсии объема за указанный период"""
        from config import VOLUME_ANALYSIS_SETTINGS
        
        settings = VOLUME_ANALYSIS_SETTINGS['volume_dispersion']
        if not settings['enabled']:
            return {'dispersion': 1.0, 'interpretation': ''}
        
        # 4 свечи в час для 15м таймфрейма
        periods = hours * 4
        if len(df) < periods:
            periods = len(df)
        
        period_df = df.tail(periods)
        
        # Дисперсия объема
        volume_std = period_df['volume'].std()
        volume_mean = period_df['volume'].mean()
        volume_dispersion = volume_std / volume_mean if volume_mean > 0 else 1.0
        
        # Интерпретация
        high_threshold = settings['high_threshold']
        low_threshold = settings['low_threshold']
        
        if volume_dispersion > high_threshold:
            interpretation = f"🔥 Высокая дисперсия объема x{volume_dispersion:.1f}"
        elif volume_dispersion < low_threshold:
            interpretation = f"📊 Низкая дисперсия объема (накопление)"
        else:
            interpretation = f"📈 Средняя дисперсия объема x{volume_dispersion:.1f}"
        
        return {
            'dispersion': volume_dispersion,
            'interpretation': interpretation
        }

    def calculate_price_dispersion(self, df: pd.DataFrame, hours: int = 2) -> Dict:
        """Расчет ценовой дисперсии за указанный период"""
        from config import DISPERSION_ANALYSIS_SETTINGS
        
        if not DISPERSION_ANALYSIS_SETTINGS['enabled']:
            return {'dispersion': 0, 'interpretation': '', 'zones': []}
        
        # 4 свечи в час для 15м таймфрейма
        periods = hours * 4
        if len(df) < periods:
            periods = len(df)
        
        period_df = df.tail(periods)
        
        # Ценовая дисперсия
        price_std = period_df['close'].std()
        price_mean = period_df['close'].mean()
        price_dispersion = (price_std / price_mean) * 100 if price_mean > 0 else 0
        
        # Поиск зон высокой дисперсии
        zones = []
        if DISPERSION_ANALYSIS_SETTINGS['show_zones_on_chart']:
            window = 10  # 10 свечей для анализа
            for i in range(0, len(period_df) - window, window):
                window_df = period_df.iloc[i:i+window]
                window_std = window_df['close'].std()
                window_mean = window_df['close'].mean()
                window_disp = (window_std / window_mean) * 100 if window_mean > 0 else 0
                
                if window_disp > DISPERSION_ANALYSIS_SETTINGS['thresholds']['high']:
                    zones.append({
                        'min': window_df['low'].min(),
                        'max': window_df['high'].max(),
                        'strength': window_disp,
                        'start': window_df.index[0],
                        'end': window_df.index[-1]
                    })
        
        # Интерпретация
        high_threshold = DISPERSION_ANALYSIS_SETTINGS['thresholds']['high']
        low_threshold = DISPERSION_ANALYSIS_SETTINGS['thresholds']['low']
        
        if price_dispersion > high_threshold:
            interpretation = f"🔥 ВЫСОКАЯ ДИСПЕРСИЯ ({price_dispersion:.1f}%)"
        elif price_dispersion < low_threshold:
            interpretation = f"📊 НИЗКАЯ ДИСПЕРСИЯ ({price_dispersion:.1f}%)"
        else:
            interpretation = f"📈 СРЕДНЯЯ ДИСПЕРСИЯ ({price_dispersion:.1f}%)"
        
        return {
            'dispersion': price_dispersion,
            'interpretation': interpretation,
            'zones': zones[:DISPERSION_ANALYSIS_SETTINGS['max_zones']]
        }

    def find_sniper_entry(self, levels: List[Dict], current_price: float, df: pd.DataFrame) -> Optional[Dict]:
        """
        Поиск снайперской точки входа (лимитный ордер на уровне)
        """
        from config import SNIPER_ENTRY_SETTINGS
        
        if not SNIPER_ENTRY_SETTINGS['enabled']:
            return None
        
        for level in levels:
            # Пропускаем слабые уровни
            if level.get('strength', 0) < SNIPER_ENTRY_SETTINGS['long']['min_strength']:
                continue
            
            # LONG: покупка на поддержке
            if level['zone_type'] == 'поддержка':
                distance = ((current_price - level['price']) / current_price) * 100
                
                if distance <= SNIPER_ENTRY_SETTINGS['long']['distance_threshold']:
                    # Проверяем подтверждение
                    if SNIPER_ENTRY_SETTINGS['long']['confirmation_volume']:
                        volume_ratio = df['volume_ratio'].iloc[-1] if 'volume_ratio' in df.columns else 1.0
                        if volume_ratio < SNIPER_ENTRY_SETTINGS['long']['confirmation_volume']:
                            continue
                    
                    if SNIPER_ENTRY_SETTINGS['long']['confirmation_rsi']:
                        rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
                        if rsi > SNIPER_ENTRY_SETTINGS['long']['confirmation_rsi']:
                            continue
                    
                    # Расчет цен для лимитного ордера
                    entry_price = level['price'] * (1 + SNIPER_ENTRY_SETTINGS['order']['price_offset'] / 100)
                    stop_loss = level['price'] * (1 - SNIPER_ENTRY_SETTINGS['order']['stop_loss_offset'] / 100)
                    take_profit = entry_price * (1 + SNIPER_ENTRY_SETTINGS['order']['take_profit'] / 100)
                    
                    return {
                        'type': 'LONG',
                        'level_price': level['price'],
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'strength': level.get('strength', 50),
                        'message': f"🎯 Снайперский вход LONG: лимит {entry_price:.4f} (уровень {level['price']:.4f})"
                    }
            
            # SHORT: продажа на сопротивлении
            elif level['zone_type'] == 'сопротивление':
                distance = ((level['price'] - current_price) / current_price) * 100
                
                if distance <= SNIPER_ENTRY_SETTINGS['short']['distance_threshold']:
                    # Проверяем подтверждение
                    if SNIPER_ENTRY_SETTINGS['short']['confirmation_volume']:
                        volume_ratio = df['volume_ratio'].iloc[-1] if 'volume_ratio' in df.columns else 1.0
                        if volume_ratio < SNIPER_ENTRY_SETTINGS['short']['confirmation_volume']:
                            continue
                    
                    if SNIPER_ENTRY_SETTINGS['short']['confirmation_rsi']:
                        rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
                        if rsi < SNIPER_ENTRY_SETTINGS['short']['confirmation_rsi']:
                            continue
                    
                    # Расчет цен для лимитного ордера
                    entry_price = level['price'] * (1 - SNIPER_ENTRY_SETTINGS['order']['price_offset'] / 100)
                    stop_loss = level['price'] * (1 + SNIPER_ENTRY_SETTINGS['order']['stop_loss_offset'] / 100)
                    take_profit = entry_price * (1 - SNIPER_ENTRY_SETTINGS['order']['take_profit'] / 100)
                    
                    return {
                        'type': 'SHORT',
                        'level_price': level['price'],
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'strength': level.get('strength', 50),
                        'message': f"🎯 Снайперский вход SHORT: лимит {entry_price:.4f} (уровень {level['price']:.4f})"
                    }
        
        return None

    def generate_signal(self, dataframes: Dict[str, pd.DataFrame], metadata: Dict, symbol: str, exchange: str) -> Optional[Dict]:
        """
        Генерация торгового сигнала на основе всех индикаторов
        """
        logger.info(f"🔄 generate_signal начал работу для {symbol}")

        global BREAKOUT_CONFIRMATION_SETTINGS
        
        if 'current' not in dataframes or dataframes['current'].empty:
            logger.warning(f"⚠️ Нет current данных для {symbol}")
            return None
        
        df = dataframes['current']
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
               
        logger.info(f"  📊 {symbol} - Цена: {last['close']}, RSI: {last['rsi'] if pd.notna(last['rsi']) else 'N/A'}")

        # ===== ОБЪЯВЛЯЕМ ПЕРЕМЕННЫЕ =====
        confidence = 50
        reasons = []
        direction = 'NEUTRAL'
        signal_type = 'regular'

        # ===== ОПРЕДЕЛЕНИЕ ТАЙМФРЕЙМА ДЛЯ ТЕКУЩЕГО ТИПА СИГНАЛА =====
        from config import SIGNAL_TIMEFRAMES
        
        # Определяем ТФ в зависимости от типа сигнала
        if signal_type == 'pump':
            tf_config = SIGNAL_TIMEFRAMES.get('pump', {})
            main_tf = tf_config.get('timeframe', SIGNAL_TIMEFRAMES['default'])
            secondary_tf = tf_config.get('secondary')
        elif signal_type == 'accumulation':
            tf_config = SIGNAL_TIMEFRAMES.get('accumulation', {})
            main_tf = tf_config.get('timeframe', SIGNAL_TIMEFRAMES['default'])
            secondary_tf = tf_config.get('secondary')
        else:
            tf_config = SIGNAL_TIMEFRAMES.get('regular', {})
            main_tf = tf_config.get('timeframe', SIGNAL_TIMEFRAMES['default'])
            secondary_tf = tf_config.get('secondary')
        
        # Получаем DataFrame для основного ТФ
        if main_tf == 'current':
            df_main = dataframes.get('current')
        else:
            # Конвертируем '15m' -> 'current', '1h' -> 'hourly' и т.д.
            tf_map = {
                '1m': '1m', '3m': '3m', '5m': '5m',
                '15m': 'current', '30m': '30m',
                '1h': 'hourly', '4h': 'four_hourly',
                '1d': 'daily', '1w': 'weekly'
            }
            df_main = dataframes.get(tf_map.get(main_tf, 'current'))
        
        if df_main is None or df_main.empty:
            logger.warning(f"⚠️ Нет данных для ТФ {main_tf}, использую current")
            df_main = dataframes.get('current')
        
        # Используем df_main вместо df для основного анализа
        df = df_main
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        
        logger.info(f"  📊 {symbol} - Цена: {last['close']}, RSI: {last['rsi'] if pd.notna(last['rsi']) else 'N/A'}")
        logger.info(f"  📊 Использую ТФ: {main_tf} (тип сигнала: {signal_type})")
        
        # Если есть secondary ТФ, загружаем его для подтверждения
        if secondary_tf:
            tf_map = {
                '1m': '1m', '3m': '3m', '5m': '5m',
                '15m': 'current', '30m': '30m',
                '1h': 'hourly', '4h': 'four_hourly',
                '1d': 'daily', '1w': 'weekly'
            }
            secondary_key = tf_map.get(secondary_tf, 'current')
            df_secondary = dataframes.get(secondary_key)
            if df_secondary is not None and not df_secondary.empty:
                logger.info(f"  📊 Вторичный ТФ {secondary_tf} загружен")

        # Словарь для перевода таймфреймов
        tf_short = {
            'monthly': '1М',
            'weekly': '1н',
            'daily': '1д',
            'four_hourly': '4ч',
            'hourly': '1ч',
            '30m': '30м',
            'current': '15м',
            '5m': '5м',
            '3m': '3м',
            '1m': '1м'
        }

        # Переменные для отслеживания пробоя
        breakout_confirmed = False
        
        # Загружаем стратегию
        from config import STRATEGY_SETTINGS
        strategy_name = STRATEGY_SETTINGS['selected']
        strategy = STRATEGY_SETTINGS[strategy_name]
        logger.info(f"  📋 Стратегия: {strategy['name']}")

        fvg_analysis = None
        liquidity_zones = None  
        
        # ===== АНАЛИЗ СОГЛАСОВАННОСТИ ТРЕНДОВ =====
        alignment = self.analyze_timeframe_alignment(dataframes)
        logger.info(f"  📊 {symbol} - Согласованность трендов: {alignment['trend_alignment']}%")
        
        if alignment['trend_alignment'] < 30:
            reasons.append(f"⚠️ Низкая согласованность ТФ ({alignment['trend_alignment']:.0f}%) — возможен разворот")
               
        # ===== ПРОВЕРКА СОГЛАСОВАННОСТИ ТАЙМФРЕЙМОВ =====
        # Определяем тип сигнала для выбора режима
        tf_signal_type = 'regular'
        if signal_type == 'accumulation':
            tf_signal_type = 'accumulation'
        elif signal_type in ['PUMP', 'DUMP', 'pump']:
            tf_signal_type = 'pump'
        
        tf_alignment = self.check_tf_alignment(dataframes, tf_signal_type)
        
        # Добавляем причины от согласованности
        for reason in tf_alignment['reasons']:
            reasons.append(reason)
        
        # Корректируем уверенность
        confidence += tf_alignment['confidence_modifier']
        
        # Для памп-дамп добавляем тип движения в причины
        if tf_signal_type == 'pump' and tf_alignment.get('trend_type'):
            if tf_alignment['trend_type'] == 'trend':
                reasons.append(f"📈 Памп по тренду — высокая вероятность продолжения")
            elif tf_alignment['trend_type'] == 'correction':
                reasons.append(f"🔄 Памп против тренда — ожидаем коррекцию")
            elif tf_alignment['trend_type'] == 'impulse':
                reasons.append(f"⚡ Импульс без явного тренда")
        
        # Определяем направление, если оно еще не определено
        if direction == 'NEUTRAL' and tf_alignment['direction']:
            direction = tf_alignment['direction']
        
        # Проверяем, нужно ли отправлять сигнал
        send_signal = False
        if tf_alignment['status'] == 'perfect' and TF_ALIGNMENT_SETTINGS.get('send_on_perfect', True):
            send_signal = True
        elif tf_alignment['status'] == 'warning' and TF_ALIGNMENT_SETTINGS.get('send_on_warning', True):
            send_signal = True
        elif tf_alignment['status'] == 'rejected' and TF_ALIGNMENT_SETTINGS.get('send_on_rejected', False):
            send_signal = True
            reasons.append(f"🔧 СИГНАЛ ОТКЛОНЕН, но отправлен в тестовом режиме")
        
        if not send_signal:
            logger.info(f"⏭️ {symbol} - сигнал отклонен по согласованности ТФ (status={tf_alignment['status']})")
            return None
              
        # ===== RSI =====
        if pd.notna(last['rsi']):
            if last['rsi'] < INDICATOR_SETTINGS['rsi_oversold']:
                reasons.append(f"RSI перепродан ({last['rsi']:.1f})")
                confidence += INDICATOR_WEIGHTS['rsi']
            elif last['rsi'] > INDICATOR_SETTINGS['rsi_overbought']:
                reasons.append(f"RSI перекуплен ({last['rsi']:.1f})")
                confidence += INDICATOR_WEIGHTS['rsi']
        
        # ===== MACD =====
        if pd.notna(last['MACD_12_26_9']) and pd.notna(last['MACDs_12_26_9']):
            if last['MACD_12_26_9'] > last['MACDs_12_26_9'] and prev['MACD_12_26_9'] <= prev['MACDs_12_26_9']:
                reasons.append("Бычье пересечение MACD")
                confidence += INDICATOR_WEIGHTS['macd']
            elif last['MACD_12_26_9'] < last['MACDs_12_26_9'] and prev['MACD_12_26_9'] >= prev['MACDs_12_26_9']:
                reasons.append("Медвежье пересечение MACD")
                confidence += INDICATOR_WEIGHTS['macd']
        
        # ===== EMA =====
        if last['ema_9'] > last['ema_21'] and prev['ema_9'] <= prev['ema_21']:
            reasons.append("Бычье пересечение EMA (9/21)")
            confidence += INDICATOR_WEIGHTS['ema_cross_current']
        elif last['ema_9'] < last['ema_21'] and prev['ema_9'] >= prev['ema_21']:
            reasons.append("Медвежье пересечение EMA (9/21)")
            confidence += INDICATOR_WEIGHTS['ema_cross_current']
        
        # ===== ОБЪЕМ =====
        if last['volume_ratio'] > 1.5:
            reasons.append(f"Объем x{last['volume_ratio']:.1f} от нормы")
            confidence += INDICATOR_WEIGHTS['volume']
        
        # ===== VWAP =====
        if FEATURES['advanced']['vwap'] and 'vwap' in df.columns:
            vwap_value = last['vwap']
            price = last['close']
            
            # Умное форматирование в зависимости от размера числа
            if vwap_value < 0.0001:
                vwap_formatted = f"{vwap_value:.8f}".rstrip('0').rstrip('.')
            elif vwap_value < 0.001:
                vwap_formatted = f"{vwap_value:.6f}".rstrip('0').rstrip('.')
            elif vwap_value < 0.01:
                vwap_formatted = f"{vwap_value:.5f}".rstrip('0').rstrip('.')
            elif vwap_value < 0.1:
                vwap_formatted = f"{vwap_value:.4f}".rstrip('0').rstrip('.')
            elif vwap_value < 1:
                vwap_formatted = f"{vwap_value:.3f}".rstrip('0').rstrip('.')
            else:
                vwap_formatted = f"{vwap_value:.2f}"
            
            reasons.append(f"Цена {'выше' if price > vwap_value else 'ниже'} VWAP ({vwap_formatted})")
            confidence += 10
        
        # ===== СИГНАЛЫ ОТ СТАРШИХ ТАЙМФРЕЙМОВ =====
        for signal in alignment['signals']:
            reasons.append(signal)
            if "НЕДЕЛЬНЫЙ" in signal or "МЕСЯЧНЫЙ" in signal:
                confidence += INDICATOR_WEIGHTS['weekly_trend']
            elif "Дневной" in signal:
                confidence += INDICATOR_WEIGHTS['daily_trend']

        # ===== СОГЛАСОВАННОСТЬ ТРЕНДОВ =====
        if alignment['trend_alignment'] > 70:
            confidence += INDICATOR_WEIGHTS['trend_alignment']
        
        # ===== АНАЛИЗ ФИБОНАЧЧИ =====
        fib_analysis = None
        if self.fibonacci and FEATURES['advanced']['fibonacci']:
            logger.info(f"  🔍 {symbol} - Начинаю анализ Фибоначчи")
            fib_analysis = self.fibonacci.analyze_multi_timeframe(dataframes)
            if fib_analysis['has_confluence']:
                for signal in fib_analysis['signals']:
                    reasons.append(signal)
                confidence += fib_analysis['strength'] / 5
                logger.info(f"  ✅ {symbol} - Фибоначчи: найдено {len(fib_analysis['signals'])} сигналов")
        
        # ===== АНАЛИЗ НАКОПЛЕНИЯ =====
        accumulation_analysis = None
        if self.accumulation and FEATURES['advanced']['accumulation']:
            logger.info(f"  🔍 {symbol} - Начинаю анализ накопления")
            accumulation_analysis = self.accumulation.analyze(df)
            
            if accumulation_analysis.get('has_accumulation'):
                for signal in accumulation_analysis['signals']:
                    reasons.append(f"📦 {signal}")
                confidence += accumulation_analysis.get('strength', 0) / 5
                signal_type = 'accumulation'  # ← это уже есть
                logger.info(f"  ✅ {symbol} - Накопление: {len(accumulation_analysis['signals'])} признаков")
                
                # Расчет потенциала
                potential = self.accumulation.calculate_potential(df, dataframes)
                if potential['has_potential']:
                    accumulation_analysis['potential'] = potential
                    for reason in potential['reasons']:
                        reasons.append(reason)
                    logger.info(f"  📈 {symbol} - Потенциал: {potential['target_pct']}%")
                
                # ✅ ДОБАВИТЬ: предупреждение о сниженном плече
                if ACCUMULATION_SIGNAL_SETTINGS.get('enabled', True):
                    reasons.append(f"⚠️ СНИЖЕННОЕ ПЛЕЧО: рекомендуется {ACCUMULATION_SIGNAL_SETTINGS['max_leverage']}x (вместо 50-100x)")

                if accumulation_analysis.get('direction'):
                    direction = accumulation_analysis['direction']
                logger.info(f"  ✅ {symbol} - Накопление: найдено {len(accumulation_analysis['signals'])} сигналов")
            else:
                logger.info(f"  ⚠️ {symbol} - Накопление НЕ обнаружено")
                # Если накопления нет, но ранее был установлен signal_type = 'accumulation' — сбрасываем
                if signal_type == 'accumulation':
                    signal_type = 'regular'

        # ===== ПОДЖАТИЕ К УРОВНЮ (для сигнала 2) =====
        compression_analysis = None
        if self.accumulation and ACCUMULATION_SETTINGS.get('send_compression_alert', True):
            try:
                compression_analysis = self.accumulation.detect_compression_to_level(df, last['close'])
                if compression_analysis.get('compression'):
                    reasons.append(compression_analysis['description'])
                    confidence += compression_analysis.get('strength', 10)
                    signal_type = 'compression'
                    logger.info(f"  ✅ {symbol} - Обнаружено поджатие к уровню")
            except Exception as e:
                logger.error(f"Ошибка анализа поджатия: {e}")

        # ===== АНАЛИЗ УРОВНЕЙ НА СТАРШИХ ТАЙМФРЕЙМАХ =====
        senior_tf_analysis = {
            'has_senior_level': False,
            'levels': [],
            'signals': [],
            'strength': 0
        }
        
        senior_tfs = ['four_hourly', 'daily', 'weekly']  # 4ч, 1д, 1н
        
        for tf_name in senior_tfs:
            if tf_name not in dataframes or dataframes[tf_name] is None:
                continue
            
            tf_df = dataframes[tf_name]
            current_price = last['close']
            
            # 1. Предыдущие хаи/лои (локальные экстремумы)
            recent_high = tf_df['high'].tail(20).max()
            recent_low = tf_df['low'].tail(20).min()
            
            high_distance = (recent_high - current_price) / current_price * 100
            low_distance = (current_price - recent_low) / current_price * 100
            
            if 0 < high_distance <= 1.0:
                senior_tf_analysis['has_senior_level'] = True
                senior_tf_analysis['levels'].append({
                    'tf': tf_name,
                    'type': 'resistance',
                    'price': recent_high,
                    'distance': high_distance,
                    'source': f'Предыдущий хай на {tf_name}'
                })
                senior_tf_analysis['signals'].append(f"📈 Цена у предыдущего хая на {tf_short.get(tf_name, tf_name)}: +{high_distance:.2f}%")
                senior_tf_analysis['strength'] += 20
            
            if 0 < low_distance <= 1.0:
                senior_tf_analysis['has_senior_level'] = True
                senior_tf_analysis['levels'].append({
                    'tf': tf_name,
                    'type': 'support',
                    'price': recent_low,
                    'distance': low_distance,
                    'source': f'Предыдущий лой на {tf_name}'
                })
                senior_tf_analysis['signals'].append(f"📉 Цена у предыдущего лоя на {tf_short.get(tf_name, tf_name)}: -{low_distance:.2f}%")
                senior_tf_analysis['strength'] += 20
            
            # 2. Уровни Фибоначчи на старших ТФ
            if hasattr(self, 'fibonacci') and self.fibonacci:
                try:
                    fib_result = self.fibonacci.analyze(tf_df, tf_name)
                    if fib_result.get('has_signal'):
                        for signal_text in fib_result.get('signals', [])[:2]:
                            if "сопротивление" in signal_text or "поддержка" in signal_text:
                                senior_tf_analysis['has_senior_level'] = True
                                senior_tf_analysis['signals'].append(signal_text)
                                senior_tf_analysis['strength'] += 15
                except Exception as e:
                    logger.debug(f"Ошибка Фибоначчи для {tf_name}: {e}")
            
            # 3. FVG зоны на старших ТФ
            if fvg_analysis and fvg_analysis.get('zones'):
                for zone in fvg_analysis['zones']:
                    if zone.get('tf') == tf_name:
                        if zone['min'] <= current_price <= zone['max']:
                            senior_tf_analysis['has_senior_level'] = True
                            senior_tf_analysis['levels'].append({
                                'tf': tf_name,
                                'type': 'zone',
                                'price': zone,
                                'distance': 0,
                                'source': f'FVG зона на {tf_name}'
                            })
                            senior_tf_analysis['signals'].append(f"📐 FVG зона на {tf_short.get(tf_name, tf_name)}: {zone['min']:.4f}-{zone['max']:.4f}")
                            senior_tf_analysis['strength'] += 25
            
            # 4. Конфлюенция EMA на старших ТФ
            ema_periods = [7, 14, 21, 28, 50, 100]
            for period in ema_periods:
                col = f'ema_{period}'
                if col in tf_df.columns:
                    ema_price = tf_df[col].iloc[-1]
                    distance = abs(ema_price - current_price) / current_price * 100
                    if distance <= 1.0:
                        level_type = 'resistance' if ema_price > current_price else 'support'
                        senior_tf_analysis['has_senior_level'] = True
                        senior_tf_analysis['levels'].append({
                            'tf': tf_name,
                            'type': level_type,
                            'price': ema_price,
                            'distance': distance,
                            'source': f'EMA {period} на {tf_name}'
                        })
                        senior_tf_analysis['signals'].append(f"📊 EMA {period} на {tf_short.get(tf_name, tf_name)}: {level_type} {ema_price:.4f}")
                        senior_tf_analysis['strength'] += 10
        
        # Добавляем сигналы старших ТФ в причины
        if senior_tf_analysis['has_senior_level']:
            for signal in senior_tf_analysis['signals'][:3]:
                if signal not in reasons:
                    reasons.append(signal)
            confidence += senior_tf_analysis['strength'] / 10
            logger.info(f"  ✅ Найдено {len(senior_tf_analysis['levels'])} уровней на старших ТФ")
        else:
            reasons.append("⚠️ Нет совпадений с уровнями на старших ТФ — возможен дальнейший рост")

        # ===== АНАЛИЗ ТРЕНДОВЫХ ЛИНИЙ =====
        trendline_breakout = False
        trendline_warnings = []
        breakout_level = None  # ← ДОБАВИТЬ

        if FEATURES['advanced']['patterns']:
            logger.info(f"  🔍 {symbol} - Анализ трендовых линий")
            trend_analyzer = TrendLineAnalyzer()
            current_tf = TIMEFRAMES.get('current', '15m')
            
            # 1. Ищем уже случившиеся пробои
            trend_lines = trend_analyzer.find_trend_lines(df, touch_count=3)
            
            best_line = None
            max_touches = 0
            for line in trend_lines:
                if line['is_broken'] and line['touches'] > max_touches:
                    max_touches = line['touches']
                    best_line = line
                    breakout_level = line['current_level']  # ← ЗАПОМИНАЕМ
            
            if best_line:
                reasons.append(f"📈 Пробой наклонного сопротивления на {current_tf} ({best_line['touches']} касаний)")
                confidence += 20
                trendline_breakout = True
                signal_type = 'breakout'
                logger.info(f"  ✅ {symbol} - Обнаружен пробой тренда с {best_line['touches']} касаниями на {current_tf}")
                
                # ===== ДЕТЕКТОР ЛОЖНОГО ПРОБОЯ =====
                if breakout_level and last['close'] < breakout_level * 1.01:  # цена вернулась к уровню
                    reasons.append("⚠️ Цена вернулась к уровню после пробоя - возможен ложный пробой")
                    confidence -= 30
                    # Меняем направление на противоположное
                    if 'LONG' in direction or direction == 'LONG':
                        direction = 'SHORT 📉 (ловушка быков)'
                    logger.info(f"  ⚠️ {symbol} - Обнаружен потенциальный ложный пробой!")
        
        # Проверяем приближение к трендовым линиям
        approaching = trend_analyzer.check_approaching_trendline(
            df, last['close'], touch_count=3, threshold=0.5
        )

        for warning in approaching:
            reasons.append(warning['message'])
            confidence += 10
            
            # ✅ Добавить предупреждение для консервативной стратегии
            if strategy.get('require_breakout_confirmation', False):
                level_type = "поддержки" if "поддержке" in warning['message'] else "сопротивления"
                reasons.append(f"⏳ {strategy['name']} стратегия: ждем ПРОБОЯ наклонного уровня {level_type} на {current_tf}")
        
        # ===== АНАЛИЗ МЛАДШИХ ТАЙМФРЕЙМОВ ДЛЯ ПАМП-ДАМП поджатие и пробой, уровни наклонные и горизонтальные(1м, 3м) =====
        if signal_type in ['PUMP', 'DUMP', 'pump']:
            try:
                # 1. Поджатие на 1м и 3м
                for tf_name in ['1m', '3m']:
                    if tf_name in dataframes and dataframes[tf_name] is not None:
                        df_minor = dataframes[tf_name]
                        if hasattr(self, 'accumulation') and self.accumulation:
                            compression = self.accumulation.detect_compression(df_minor)
                            if compression.get('compression'):
                                reasons.append(f"⚡ Поджатие на {tf_name}: {compression['description']}")
                                confidence += 15
                                logger.info(f"  ✅ {symbol} - Поджатие на {tf_name}")
                
                # 2. Пробой уровней на 1м и 3м
                trend_analyzer = TrendLineAnalyzer()
                for tf_name in ['1m', '3m']:
                    if tf_name in dataframes and dataframes[tf_name] is not None:
                        df_minor = dataframes[tf_name]
                        trend_lines = trend_analyzer.find_trend_lines(df_minor, touch_count=2)
                        for line in trend_lines:
                            if line.get('is_broken', False):
                                level_type = "сопротивления" if line['type'] == 'resistance' else "поддержки"
                                reasons.append(f"⚡ Пробой наклонного {level_type} на {tf_name} ({line['touches']} касаний)")
                                confidence += 20
                                logger.info(f"  ✅ {symbol} - Пробой на {tf_name}")
                                break  # берём первый пробой
                
                # 3. Подход к уровню на 1м и 3м (опционально)
                for tf_name in ['1m', '3m']:
                    if tf_name in dataframes and dataframes[tf_name] is not None:
                        df_minor = dataframes[tf_name]
                        # Проверяем приближение к локальным экстремумам
                        recent_high = df_minor['high'].tail(10).max()
                        recent_low = df_minor['low'].tail(10).min()
                        current_price = last['close']
                        
                        distance_to_high = (recent_high - current_price) / current_price * 100
                        distance_to_low = (current_price - recent_low) / current_price * 100
                        
                        if 0 < distance_to_high <= 0.5:
                            reasons.append(f"🎯 Цена подходит к локальному максимуму на {tf_name} (+{distance_to_high:.1f}%)")
                            confidence += 10
                        elif 0 < distance_to_low <= 0.5:
                            reasons.append(f"🎯 Цена подходит к локальному минимуму на {tf_name} (-{distance_to_low:.1f}%)")
                            confidence += 10

                # 4. Пробой горизонтальных уровней на 1м и 3м
                for tf_name in ['1m', '3m']:
                    if tf_name in dataframes and dataframes[tf_name] is not None:
                        df_minor = dataframes[tf_name]
                        # Ищем локальные максимумы/минимумы
                        recent_high = df_minor['high'].tail(20).max()
                        recent_low = df_minor['low'].tail(20).min()
                        current_price = last['close']
                        
                        if current_price > recent_high:
                            reasons.append(f"⚡ Пробой горизонтального сопротивления на {tf_name}: {recent_high:.4f}")
                            confidence += 15
                        elif current_price < recent_low:
                            reasons.append(f"⚡ Пробой горизонтальной поддержки на {tf_name}: {recent_low:.4f}")
                            confidence += 15
                        
            except Exception as e:
                logger.error(f"❌ Ошибка анализа младших ТФ для {symbol}: {e}")

        # ===== ПОДТВЕРЖДЕНИЕ ПРОБОЕВ =====
        if FEATURES['advanced']['patterns'] and BREAKOUT_CONFIRMATION_SETTINGS['enabled']:
            logger.info(f"  🔍 {symbol} - Проверка подтвержденных пробоев")
            
            from config import BREAKOUT_CONFIRMATION_SETTINGS
            
            # ✅ Используем secondary_tf для подтверждения, если он есть
            if secondary_tf and df_secondary is not None:
                df_confirmation = df_secondary
                current_tf_name = secondary_tf
                logger.info(f"  📊 Использую {secondary_tf} для подтверждения пробоя")
            else:
                df_confirmation = df
                current_tf_name = main_tf
                logger.info(f"  ⚠️ Нет данных {secondary_tf}, использую {main_tf}")
            
            # Уровни ищем на текущем ТФ (15m)
            trend_analyzer = TrendLineAnalyzer()
            trend_lines = trend_analyzer.find_trend_lines(df, touch_count=3)
            
            for line in trend_lines:
                confirmed = self.breakout_tracker.check_breakout_confirmation(
                    symbol, current_tf_name, df_confirmation, line, last['close'],
                    required_candles=BREAKOUT_CONFIRMATION_SETTINGS['required_candles'],
                    required_percent=BREAKOUT_CONFIRMATION_SETTINGS['required_percent'],
                    volume_confirmation=BREAKOUT_CONFIRMATION_SETTINGS['volume_confirmation'],
                    confirmation_mode=BREAKOUT_CONFIRMATION_SETTINGS['confirmation_mode']
                )
                
                if confirmed:
                    # Формируем понятное описание уровня
                    level_type = "сопротивления" if line['type'] == 'resistance' else "поддержки"
                    tf_display = current_tf_name
                    
                    reasons.append(f"✅ {confirmed['message']}")
                    confidence += 30
                    signal_type = 'confirmed_breakout'
                    breakout_confirmed = True
                    
                    # Добавляем понятную причину с указанием ТФ пробоя
                    reasons.append(f"📈 Пробой наклонного {level_type} на {tf_display} ({line['touches']} касаний)")
                    
                    # Определяем направление
                    if confirmed['direction'] == 'вверх':
                        direction = 'LONG 📈 (подтвержденный пробой)'
                    else:
                        direction = 'SHORT 📉 (подтвержденный пробой)'
                    
                    logger.info(f"  ✅ {symbol} - Подтвержденный пробой на {tf_display}! +30 confidence")
                    break  # берем первый подтвержденный пробой

        # ===== АНАЛИЗ НАКЛОННЫХ УРОВНЕЙ НА МЛАДШИХ ТФ =====
        trendline_breakout_5m = False
        trendline_breakout_15m = False
        
        trend_analyzer = TrendLineAnalyzer()
        
        # Проверяем на 5м
        if '5m' in dataframes and dataframes['5m'] is not None:
            df_5m = dataframes['5m']
            try:
                trend_lines_5m = trend_analyzer.find_trend_lines(df_5m, touch_count=3)
                for line in trend_lines_5m:
                    if line.get('is_broken', False):
                        trendline_breakout_5m = True
                        reasons.append(f"📉 Пробой наклонного уровня на 5м ({line['touches']} касаний)")
                        confidence += 15
                        break
            except Exception as e:
                logger.debug(f"Ошибка анализа наклонных на 5м: {e}")
        
        # Проверяем на 15м
        if 'current' in dataframes and dataframes['current'] is not None:
            df_15m = dataframes['current']
            try:
                trend_lines_15m = trend_analyzer.find_trend_lines(df_15m, touch_count=3)
                for line in trend_lines_15m:
                    if line.get('is_broken', False):
                        trendline_breakout_15m = True
                        reasons.append(f"📉 Пробой наклонного уровня на 15м ({line['touches']} касаний)")
                        confidence += 10
                        break
            except Exception as e:
                logger.debug(f"Ошибка анализа наклонных на 15м: {e}")

        # ===== ПРОВЕРКА СТРАТЕГИИ: ТРЕБОВАНИЕ ПРОБОЯ =====
        if strategy['require_breakout_confirmation']:
            if not breakout_confirmed:
                # Определяем направление ожидаемого пробоя
                if direction == 'LONG':
                    expected_breakout = "вверх"
                elif direction == 'SHORT':
                    expected_breakout = "вниз"
                else:
                    expected_breakout = "неизвестно"
                
                reasons.append(f"⏳ {strategy['name']} стратегия: ждем ПРОБОЯ {expected_breakout} на 15m")
                direction = 'NEUTRAL'
                logger.info(f"  ⏳ {symbol} - Сигнал отменен: требуется подтверждение пробоя {expected_breakout}")

        # ===== АНАЛИЗ КОНВЕРГЕНЦИИ УРОВНЕЙ =====
        if FEATURES['advanced']['patterns']:
            logger.info(f"  🔍 {symbol} - Анализ конвергенции уровней")
            level_collector = LevelCollector()
            
            # Собираем все уровни
            all_levels = level_collector.collect_levels(dataframes, last['close'])
            
            # Ищем совпадающие уровни
            confluence_zones = level_collector.find_confluence_levels(all_levels, last['close'], tolerance=0.5)
            
            # Словарь для перевода таймфреймов
            tf_display_map = {
                'current': '15м',
                'hourly': '1ч',
                'four_hourly': '4ч',
                'daily': '1д',
                'weekly': '1н',
                'monthly': '1м'
            }
            
            for zone in confluence_zones:
                # Форматируем таймфреймы для отображения
                tfs_display = []
                for tf in zone['timeframes']:
                    tfs_display.append(tf_display_map.get(tf, tf))
                tfs_str = ', '.join(tfs_display)
                
                # Форматируем цену уровня
                level_price = zone['price']
                if level_price < 0.0001:
                    price_str = f"{level_price:.8f}".rstrip('0').rstrip('.')
                elif level_price < 0.001:
                    price_str = f"{level_price:.6f}".rstrip('0').rstrip('.')
                elif level_price < 0.01:
                    price_str = f"{level_price:.5f}".rstrip('0').rstrip('.')
                elif level_price < 0.1:
                    price_str = f"{level_price:.4f}".rstrip('0').rstrip('.')
                elif level_price < 1:
                    price_str = f"{level_price:.3f}".rstrip('0').rstrip('.')
                else:
                    price_str = f"{level_price:.2f}".rstrip('0').rstrip('.')
                
                reason_text = f"Конвергенция на {tfs_str}: {zone['zone_type']} {price_str} (сила {zone['strength']:.0f}%)"
                
                # Проверяем, нет ли уже такой причины
                if reason_text not in reasons:
                    reasons.append(reason_text)
                
                if zone['direction'] == 'LONG' and direction != 'LONG':
                    # Сильный уровень поддержки снизу
                    confidence += zone['strength'] / 2
                    if zone['distance'] < 5:
                        reasons.append(f"Близкая сильная поддержка ({zone['distance']:.1f}%)")
                elif zone['direction'] == 'SHORT' and direction != 'SHORT':
                    # Сильный уровень сопротивления сверху
                    confidence += zone['strength'] / 2
                    if zone['distance'] < 5:
                        reasons.append(f"Близкое сильное сопротивление ({zone['distance']:.1f}%)")

        # ===== АНАЛИЗ КОНВЕРГЕНЦИИ И СИЛЫ УРОВНЕЙ =====
        if FEATURES['advanced']['patterns']:
            logger.info(f"  🔍 {symbol} - Анализ силы уровней")
            level_collector = LevelCollector()
            
            # Собираем все уровни
            all_levels = level_collector.collect_levels(dataframes, last['close'])
            
            # Ищем совпадающие уровни (конвергенцию)
            confluence_zones = level_collector.find_confluence_levels(all_levels, last['close'], tolerance=0.5)
            
            # Анализируем каждый сильный уровень
            for zone in confluence_zones[:3]:  # топ-3 уровня
                # Оценка силы уровня
                strength_score = level_collector.calculate_level_strength_score(
                    zone, df, last, alignment
                )
                
                # Добавляем в причины
                if strength_score['strength'] >= 70:
                    reasons.append(f"🔥 {zone['source']}: {zone['zone_type']} на {zone['price']:.4f}")
                else:
                    reasons.append(f"⭐ {zone['source']}: {zone['zone_type']} на {zone['price']:.4f}")
                
                for signal in strength_score['signals'][:3]:
                    reasons.append(f"   {signal}")
                
                reasons.append(f"   📊 Вероятность разворота: {strength_score['probability']:.0f}%")
                
                # Корректируем направление
                if strength_score['action'] in ['разворот', 'вероятный_разворот']:
                    if zone['direction'] == 'LONG':
                        if direction != 'LONG':
                            old_dir = direction
                            direction = 'LONG 📈 (разворот от сильного уровня)'
                            confidence += strength_score['strength'] / 3
                            reasons.append(f"🔄 Смена направления: {old_dir} → LONG (сильный уровень)")
                    elif zone['direction'] == 'SHORT':
                        if direction != 'SHORT':
                            old_dir = direction
                            direction = 'SHORT 📉 (разворот от сильного уровня)'
                            confidence += strength_score['strength'] / 3
                            reasons.append(f"🔄 Смена направления: {old_dir} → SHORT (сильный уровень)")
                
                # Если вероятность пробоя выше
                elif strength_score['action'] == 'возможный_пробой':
                    reasons.append(f"⚠️ Возможен пробой уровня (вероятность разворота {strength_score['probability']:.0f}%)")
                    confidence -= 10

                # ===== СНАЙПЕРСКИЕ ТОЧКИ ВХОДА =====
                if SNIPER_ENTRY_SETTINGS['enabled'] and confluence_zones:
                    sniper = self.find_sniper_entry(confluence_zones, last['close'], df)
                    if sniper:
                        reasons.append(sniper['message'])
                        confidence += sniper['strength'] / 5
                        
                        # Добавляем в результат для отображения
                        if 'sniper_entry' not in locals():
                            # Проверяем, что signal является словарем
                            if isinstance(signal, dict):
                                signal['sniper_entry'] = sniper
                            else:
                                logger.warning(f"⚠️ signal не является словарем, тип: {type(signal)}")
                        logger.info(f"  🎯 {symbol} - Найдена снайперская точка входа: {sniper['type']} по {sniper['entry_price']:.4f}")

        # ===== АНАЛИЗ НАКОПЛЕНИЯ ПОСЛЕ ПРОБОЯ =====
                if breakout_level and (last['close'] > breakout_level * 0.99 and last['close'] < breakout_level * 1.01):
                    # Цена тестирует уровень после пробоя
                    if last['volume_ratio'] > 1.5:
                        reasons.append("📊 Накопление на пробитом уровне")
                        confidence += 15  # добавляем уверенности
                        logger.info(f"  📊 {symbol} - Накопление на пробитом уровне (объем x{last['volume_ratio']:.1f})")
                    else:
                        reasons.append("⚠️ Слабое накопление - возможен ложный пробой")
                        confidence -= 20
                        logger.info(f"  ⚠️ {symbol} - Слабое накопление на пробитом уровне")
        
        # ===== FVG МУЛЬТИТАЙМФРЕЙМОВЫЙ АНАЛИЗ (SMC) =====
        fvg_analysis = {'has_fvg': False, 'signals': [], 'zones': []}
        if FEATURES['advanced']['smart_money'] and FVG_SETTINGS.get('enabled', True):
            try:
                logger.info(f"  🔍 {symbol} - Анализ FVG (SMC)")
                fvg_analysis = self.smc_fvg_analyzer.analyze_multi_timeframe(dataframes)
                logger.info(f"  📊 FVG анализ вернул: has_fvg={fvg_analysis.get('has_fvg', False)}, zones={len(fvg_analysis.get('zones', []))}")
                
                if fvg_analysis['has_fvg']:
                    for zone in fvg_analysis['zones']:  # берём из zones, а не из signals
                        reasons.insert(0, zone['description'])
                    # for signal_text in fvg_analysis['signals'][:5]:
                        # reasons.append(signal_text)  # ← было
                        # reasons.insert(0, signal_text)  # ← стало
                    confidence += fvg_analysis['strength'] / 5
                    logger.info(f"  ✅ {symbol} - Найдено FVG: {len(fvg_analysis['zones'])}")
            except Exception as e:
                logger.error(f"❌ Ошибка в FVG анализе: {e}")

        # ===== FVG ПО НАПРАВЛЕНИЮ =====
        if strategy['fvg'].get('direction_filter', True):
            if direction == 'LONG':
                fvg_below = [z for z in fvg_analysis.get('zones', []) if z['max'] < last['close']]
                if not fvg_below:
                    reasons.append("⚠️ Нет FVG поддержки снизу (LONG)")
                    confidence -= 10
                else:
                    logger.info(f"  ✅ Найдено {len(fvg_below)} FVG поддержки снизу")
            elif direction == 'SHORT':
                fvg_above = [z for z in fvg_analysis.get('zones', []) if z['min'] > last['close']]
                if not fvg_above:
                    reasons.append("⚠️ Нет FVG сопротивления сверху (SHORT)")
                    confidence -= 10
                else:
                    logger.info(f"  ✅ Найдено {len(fvg_above)} FVG сопротивления сверху")

        # ===== ПРОВЕРКА СТРАТЕГИИ: ТРЕБОВАНИЕ ЗАКРЫТИЯ FVG =====
        require_close_pct = strategy['fvg'].get('require_close_pct', 0)
        if require_close_pct > 0 and fvg_analysis.get('has_fvg', False):
            try:
                # Проверяем, насколько закрыт ближайший FVG
                current_price = last['close']
                closest_fvg = None
                min_distance = float('inf')
                
                for zone in fvg_analysis.get('zones', []):
                    if zone['min'] > current_price:
                        distance = zone['min'] - current_price
                    elif zone['max'] < current_price:
                        distance = current_price - zone['max']
                    else:
                        # Цена внутри FVG
                        distance = 0
                        # Считаем процент закрытия
                        fvg_range = zone['max'] - zone['min']
                        if fvg_range > 0:
                            if current_price > zone['max']:
                                close_pct = 100
                            elif current_price < zone['min']:
                                close_pct = 0
                            else:
                                close_pct = ((current_price - zone['min']) / fvg_range) * 100
                            
                            if close_pct < require_close_pct:
                                # ✅ ИСПРАВЛЕНО: используем zone.get('tf', 'FVG') вместо zone['tf_short']
                                tf_name = zone.get('tf', 'FVG')
                                reasons.append(f"⚠️ FVG {tf_name} закрыт только на {close_pct:.0f}% (требуется {require_close_pct}%)")
                                confidence -= 15
                                logger.info(f"  ⚠️ FVG закрыт на {close_pct:.0f}% < {require_close_pct}%")
                    
            except Exception as e:
                logger.error(f"❌ Ошибка в проверке закрытия FVG: {e}")

        ## Проверка стратегии: требовать закрытие FVG на X%               #ЗАГЛУШКА
        # require_close_pct = strategy['fvg'].get('require_close_pct', 0)
        # if require_close_pct > 0 and fvg_in_zone > 0:
            ## Здесь нужно получить процент закрытия FVG
            ## Для простоты пока пропустим, добавим позже
            #logger.info(f"  📊 Требуется закрытие FVG на {require_close_pct}%")
        
        # ===== SMC: CHoCH (смена тренда) =====
        choch_analysis = None
        if hasattr(self, 'smart_money') and FEATURES['advanced']['smart_money']:
            try:
                choch_analysis = self.smart_money.detect_choch(df, tf_short.get('current', '15м'))
                if choch_analysis.get('has_choch'):
                    reasons.append(choch_analysis['description'])
                    confidence += choch_analysis['strength'] / 5
                    if choch_analysis['direction']:
                        direction = choch_analysis['direction']
                        logger.info(f"  🎯 Направление от CHoCH: {direction}")
            except Exception as e:
                logger.error(f"Ошибка CHoCH: {e}")

        # ===== SMC: Premium/Discount Zones =====
        pd_analysis = self.smart_money.calculate_premium_discount_zones(df, tf_short.get('current', '15м'))
        if pd_analysis.get('has_zone'):
            # reasons.append(pd_analysis['description'])  # ← было
            reasons.insert(0, pd_analysis['description'])  # ← стало
            confidence += 15
            if pd_analysis['direction'] and direction == 'NEUTRAL':
                direction = pd_analysis['direction']

        # ===== SMC: EQH/EQL =====
        equal_analysis = self.smart_money.find_equal_highs_lows(df, tf_short.get('current', '15м'), last['close'], signal_type)
        if equal_analysis.get('has_equal'):
            # reasons.append(equal_analysis['description'])  # ← было
            reasons.insert(0, equal_analysis['description'])  # ← стало
            confidence += 10

        # ===== ЗОНЫ ЛИКВИДНОСТИ =====              
        if LIQUIDITY_ZONES_SETTINGS.get('enabled', True):
            try:
                logger.info(f"  🔍 {symbol} - Анализ зон ликвидности")
                liquidity_zones = self.liquidity_zone_detector.analyze_multi_timeframe(dataframes)
                
                if liquidity_zones['has_zones']:
                    for signal in liquidity_zones['signals'][:3]:
                        reasons.append(signal)
                    confidence += liquidity_zones['strength'] / 10
                    logger.info(f"  ✅ {symbol} - Найдено {len(liquidity_zones['zones'])} зон ликвидности")
                    
                    # Проверяем, находится ли цена рядом с зоной
                    near_zone = self.liquidity_zone_detector.check_price_near_zone(
                        last['close'], liquidity_zones['zones'], distance_threshold=0.5
                    )
                    if near_zone:
                        reasons.append(f"⚠️ Цена в {near_zone['distance']:.1f}% от зоны {near_zone['type']}")
                        
            except Exception as e:
                logger.error(f"❌ Ошибка в анализе зон ликвидности для {symbol}: {e}")

        # ===== РАСЧЕТ ПОТЕНЦИАЛА (ДЛЯ ВСЕХ ТИПОВ СИГНАЛОВ) =====
        potential_analysis = None
        if self.accumulation and FEATURES['advanced']['accumulation']:
            try:
                # Передаем FVG и зоны ликвидности если они есть
                fvg_for_potential = fvg_analysis if 'fvg_analysis' in locals() else None
                liquidity_for_potential = liquidity_zones.get('zones') if 'liquidity_zones' in locals() and liquidity_zones else None
                
                logger.info(f"  🔍 {symbol} - Передаю в calculate_potential: FVG={fvg_for_potential is not None}, Liquidity={liquidity_for_potential is not None}")

                potential_analysis = self.accumulation.calculate_potential(
                    df, dataframes, fvg_for_potential, liquidity_for_potential
                )
                
                if potential_analysis and potential_analysis.get('has_potential'):
                    logger.info(f"  ✅ {symbol} - ПОТЕНЦИАЛ НАЙДЕН: {potential_analysis}")
                    logger.info(f"  📝 Причины из потенциала: {potential_analysis['reasons']}")
                    for reason in potential_analysis['reasons']:
                        logger.info(f"  ➕ Добавляю причину: {reason}")
                        # Добавляем в начало, чтобы было видно первым
                        if reason not in reasons:
                            reasons.insert(0, reason)
                            logger.info(f"  ✅ Причина добавлена")
                        else:
                            logger.info(f"  ⏭️ Причина уже есть")
                    
                    # ✅ БОНУС ЗА КОНФЛЮЕНЦИЮ (ВЫНЕСЕН ИЗ ELSE)
                    level_count = potential_analysis.get('level_count', 0)

                    # ✅ Проверка стратегии: минимальное количество уровней в конфлюенции
                    if level_count < strategy['min_confluence_levels']:
                        reasons.append(f"⚠️ {strategy['name']} стратегия: требуется {strategy['min_confluence_levels']}+ уровней (есть {level_count})")
                        direction = 'NEUTRAL'
                        logger.info(f"  ⏳ {symbol} - Сигнал отменен: недостаточно уровней конфлюенции")

                    if level_count >= 4:
                        confidence += 25
                        reasons.append(f"🔥 СУПЕР-КОНФЛЮЕНЦИЯ: {level_count} уровней в одной зоне")
                    elif level_count >= 3:
                        confidence += 20
                        reasons.append(f"⭐ СИЛЬНАЯ КОНФЛЮЕНЦИЯ: {level_count} уровней в одной зоне")
                    elif level_count >= 2:
                        confidence += 10
                        reasons.append(f"📊 СХОЖДЕНИЕ УРОВНЕЙ: {level_count} уровня в одной зоне")
                    
                    logger.info(f"  ✅ {symbol} - Потенциал: {potential_analysis['target_pct']}% до {potential_analysis['target_level']}")
                else:
                    logger.info(f"  ⚠️ {symbol} - ПОТЕНЦИАЛ НЕ НАЙДЕН")
                        
            except Exception as e:
                logger.error(f"❌ Ошибка в расчете потенциала для {symbol}: {e}")
        
        # ===== АНАЛИЗ КАСАНИЙ EMA =====
        ema_touch = self.analyze_ema_touch(df, last)
        for signal in ema_touch['signals']:
            reasons.append(signal)
            confidence += 5  # небольшой бонус
        
        # ===== АНАЛИЗ КАСАНИЙ EMA НА ВСЕХ ТАЙМФРЕЙМАХ =====
        ema_touch_analysis = self.analyze_ema_touch_multi_timeframe(dataframes, last['close'])
        if ema_touch_analysis['signals']:
            for signal in ema_touch_analysis['signals']:
                reasons.append(signal)
            confidence += ema_touch_analysis['strength'] / 10
            logger.info(f"  ✅ {symbol} - Найдено {len(ema_touch_analysis['signals'])} касаний EMA")        
        
        # ===== АНАЛИЗ ОБЪЕМОВ =====
        if VOLUME_ANALYSIS_SETTINGS['enabled']:
            logger.info(f"  🔍 {symbol} - Анализ объемов")
            
            # 1. Детектор аномальных свечей
            volume_spike = self.calculate_volume_spike(df)
            if volume_spike['spike']:
                reasons.append(f"🔥 Аномальный объем x{volume_spike['ratio']:.1f}")
                confidence += VOLUME_ANALYSIS_SETTINGS['spike_detector']['weight']
                logger.info(f"  🔥 {symbol} - Объемный всплеск x{volume_spike['ratio']:.1f}")
            
            # 2. Дисперсия объема
            vol_dispersion = self.calculate_volume_dispersion(df, hours=2)
            if vol_dispersion['dispersion'] != 1.0:
                reasons.append(vol_dispersion['interpretation'])
                if vol_dispersion['dispersion'] > VOLUME_ANALYSIS_SETTINGS['volume_dispersion']['high_threshold']:
                    confidence += VOLUME_ANALYSIS_SETTINGS['volume_dispersion']['weight']
            
            # 3. Имбаланс buy/sell (если включен)
            if VOLUME_ANALYSIS_SETTINGS['imbalance']['enabled'] and self.imbalance:
                imbalance_result = self.imbalance.analyze(dataframes)
                if imbalance_result['has_imbalance']:
                    for signal in imbalance_result['signals']:
                        reasons.append(signal)
                    confidence += VOLUME_ANALYSIS_SETTINGS['imbalance']['weight']
                    logger.info(f"  📊 {symbol} - Имбаланс: {len(imbalance_result['signals'])} сигналов")

        # ===== RSI ДИВЕРГЕНЦИЯ НА СТАРШИХ ТФ =====
        rsi_divergence = False
        if self.divergence:
            for tf_name in senior_tfs:
                if tf_name in dataframes and dataframes[tf_name] is not None:
                    tf_df = dataframes[tf_name]
                    if 'rsi' in tf_df.columns:
                        try:
                            divergence = self.divergence.detect_rsi_divergence(tf_df, tf_name)
                            if divergence.get('bullish') or divergence.get('bearish'):
                                rsi_divergence = True
                                reasons.append(f"🔄 RSI дивергенция на {tf_short.get(tf_name, tf_name)}")
                                confidence += 20
                                break
                        except Exception as e:
                            logger.debug(f"Ошибка RSI дивергенции для {tf_name}: {e}")

        # ===== АНАЛИЗ ДИСПЕРСИИ =====
        if DISPERSION_ANALYSIS_SETTINGS['enabled']:
            logger.info(f"  🔍 {symbol} - Анализ дисперсии")
            
            # Анализируем за разные периоды
            dispersion_zones = []
            for hours, name in [(1, 'час'), (2, 'часа'), (4, 'часа')]:
                dispersion = self.calculate_price_dispersion(df, hours=hours)
                if dispersion['dispersion'] > 0:
                    reasons.append(f"📊 Дисперсия за {name}: {dispersion['interpretation']}")
                    dispersion_zones = dispersion.get('zones', [])
                    
                    # Влияние на уверенность
                    if dispersion['dispersion'] > DISPERSION_ANALYSIS_SETTINGS['thresholds']['high']:
                        confidence += DISPERSION_ANALYSIS_SETTINGS['weights']['high']
                    elif dispersion['dispersion'] < DISPERSION_ANALYSIS_SETTINGS['thresholds']['low']:
                        confidence += DISPERSION_ANALYSIS_SETTINGS['weights']['low']
                    else:
                        confidence += DISPERSION_ANALYSIS_SETTINGS['weights']['medium']
                    break  # берем только один период для простоты
        
        # ===== ФАНДИНГ =====
        funding = metadata.get('funding_rate')
        if funding is not None and funding != 0:
            funding_pct = funding * 100
            if funding > 0.001:
                reasons.append(f"Позитивный фандинг ({funding_pct:.4f}%)")
            elif funding < -0.001:
                reasons.append(f"Негативный фандинг ({funding_pct:.4f}%)")
        
        # Подсчет медвежьих/бычьих сигналов для DUMP/PUMP
        bearish_score = 0
        bullish_score = 0
        
        if 'Медвежье пересечение MACD' in str(reasons):
            bearish_score += 30
        if 'Бычье пересечение MACD' in str(reasons):
            bullish_score += 30
            
        if 'Цена ниже VWAP' in str(reasons):
            bearish_score += 20
        if 'Цена выше VWAP' in str(reasons):
            bullish_score += 20
            
        if 'НЕДЕЛЬНЫЙ ТРЕНД НИСХОДЯЩИЙ' in str(reasons):
            bearish_score += 40
        if 'НЕДЕЛЬНЫЙ ТРЕНД ВОСХОДЯЩИЙ' in str(reasons):
            bullish_score += 40
        
        logger.info(f"  📊 {symbol} - Бычий счет: {bullish_score}, Медвежий счет: {bearish_score}")

        # ===== ОПРЕДЕЛЕНИЕ БАЗОВОГО НАПРАВЛЕНИЯ =====
        bullish_keywords = ['перепродан', 'Бычье', 'восходящий', 'негативный фандинг', 'выше VWAP', 'пробой']
        bearish_keywords = ['перекуплен', 'Медвежье', 'нисходящий', 'позитивный фандинг', 'ниже VWAP']
        
        bullish = sum(1 for r in reasons if any(k in r for k in bullish_keywords))
        bearish = sum(1 for r in reasons if any(k in r for k in bearish_keywords))
        
        # Базовое направление от индикаторов
        base_direction = 'NEUTRAL'
        if accumulation_analysis and accumulation_analysis.get('direction') and accumulation_analysis.get('has_accumulation'):
            base_direction = accumulation_analysis['direction']
        if trendline_breakout:
            if alignment['weekly_trend'] == 'НИСХОДЯЩИЙ' and last['rsi'] > 70:
                # Пробой вверх при медвежьем тренде и перекупленности = ЛОВУШКА!
                base_direction = 'SHORT 📉 (ловушка быков)'
                reasons.append("⚠️ Пробой вверх при медвежьем тренде - возможен ложный пробой")
                confidence -= 20
            else:
                base_direction = 'LONG'
        elif bullish > bearish and confidence >= MIN_CONFIDENCE:
            if alignment['weekly_trend'] == 'ВОСХОДЯЩИЙ':
                base_direction = 'Разворот LONG'
            else:
                base_direction = 'LONG'
        elif bearish > bullish and confidence >= MIN_CONFIDENCE:
            if alignment['weekly_trend'] == 'НИСХОДЯЩИЙ':
                base_direction = 'Разворот SHORT'
            else:
                base_direction = 'SHORT'
        
        direction = base_direction
        logger.info(f"  🎯 [1] Направление после базового определения: {direction}")
        
        # ===== ПРИОРИТЕТ СТАРШИХ ТАЙМФРЕЙМОВ =====
        if senior_tf_analysis.get('has_senior_level', False):
            # Проверяем наличие бычьих сигналов на 1д/1н
            bullish_on_senior = False
            for signal in senior_tf_analysis.get('signals', []):
                if any(word in signal for word in ['FVG', 'быч', 'поддержка', 'EMA']):
                    if '1д' in signal or '1н' in signal:
                        bullish_on_senior = True
                        break
            
            # Медвежий сигнал на старших ТФ → SHORT
            bearish_on_senior = False
            for signal in senior_tf_analysis.get('signals', []):
                if any(word in signal for word in ['FVG', 'медвеж', 'сопротивление', 'EMA']):
                    if '1д' in signal or '1н' in signal:
                        bearish_on_senior = True
                        break
            
            if bearish_on_senior and direction == 'LONG':
                old_direction = direction
                direction = 'SHORT'
                reasons.append(f"🔄 СМЕНА НАПРАВЛЕНИЯ: {old_direction} → SHORT (медвежий сигнал на 1д/1н)")
                confidence += 20
                logger.info(f"  🔄 Приоритет старших ТФ: LONG → SHORT")
            
            # ✅ БЫЧИЙ СИГНАЛ НА СТАРШИХ ТФ → LONG
            if bullish_on_senior and direction == 'SHORT':
                old_direction = direction
                direction = 'LONG'
                reasons.append(f"🔄 СМЕНА НАПРАВЛЕНИЯ: {old_direction} → LONG (бычий сигнал на 1д/1н)")
                confidence += 20
                logger.info(f"  🔄 Приоритет старших ТФ: SHORT → LONG")

        # ===== ДЕТЕКТОР ВЫБИВА СТОПОВ =====
        stop_hunt = None
        if FEATURES['advanced']['patterns'] and STOP_HUNT_SETTINGS.get('enabled', True):
            try:
                logger.info(f"  🔍 {symbol} - Анализ выбива стопов")
                
                # Проверяем на разных ТФ
                tf_priority = ['current', '30m', 'hourly', 'four_hourly']
                for tf_name in tf_priority:
                    if tf_name not in dataframes or dataframes[tf_name] is None:
                        continue
                    
                    df_tf = dataframes[tf_name]
                    stop_hunt = self.stop_hunt_detector.detect_stop_hunt(
                        symbol, tf_name, df_tf, last['close']
                    )
                    
                    if stop_hunt:
                        logger.info(f"  ✅ {symbol} - Обнаружен стоп-хант на {tf_name}!")
                        reasons.append(stop_hunt['message'])
                        confidence += STOP_HUNT_SETTINGS['strength_bonus']
                        break
                        
            except Exception as e:
                logger.error(f"❌ Ошибка в детекторе стоп-хантов для {symbol}: {e}")

        # ===== ВХОД ПОСЛЕ ВЫБИВА СТОПОВ =====
        stop_hunt_signal = False  # ← флаг, что сигнал уже сформирован
        if stop_hunt and POST_STOP_HUNT_SETTINGS.get('enabled', True):
            # Стоп-хант уже обнаружен, формируем сигнал на разворот
            logger.info(f"  🎯 {symbol} - Формирую сигнал на вход после стоп-ханта")
            
            # Направление разворота уже определено в stop_hunt['direction']
            direction = stop_hunt['direction']
            signal_type = POST_STOP_HUNT_SETTINGS.get('signal_type', 'stop_hunt_reversal')
            
            # Добавляем специальную причину
            reasons.append(f"🚀 ВХОД ПОСЛЕ ВЫБИВА СТОПОВ: {stop_hunt['message']}")
            
            # Увеличиваем уверенность
            confidence += POST_STOP_HUNT_SETTINGS.get('min_confidence_bonus', 15)
            
            # Проверяем подтверждение от младших ТФ (1м, 3м, 5м)
            if POST_STOP_HUNT_SETTINGS.get('require_confirmation', True):
                minor_confirmation = 0
                for tf_name in ['1m', '3m', '5m']:
                    if tf_name in dataframes and dataframes[tf_name] is not None:
                        df_minor = dataframes[tf_name]
                        last_minor = df_minor.iloc[-1]
                        
                        if direction == 'LONG' and last_minor['ema_9'] > last_minor['ema_21']:
                            minor_confirmation += 1
                        elif direction == 'SHORT' and last_minor['ema_9'] < last_minor['ema_21']:
                            minor_confirmation += 1
                
                if minor_confirmation >= 2:
                    reasons.append(f"✅ Подтверждение от младших ТФ ({minor_confirmation}/3)")
                    confidence += 10
                else:
                    reasons.append(f"⚠️ Слабое подтверждение от младших ТФ ({minor_confirmation}/3)")
                    confidence -= 5                   
                   
        # ПОТОМ ВЫВОДИМ ЛОГ (уже с нормализованной уверенностью)
        logger.info(f"  📊 {symbol} - Направление: {direction}, Уверенность: {confidence}")
        
        if direction == 'NEUTRAL':
            logger.info(f"⏭️ NEUTRAL сигнал для {symbol}")
            return None
        
        logger.info(f"  🎯 [5] Направление перед проверкой NEUTRAL: {direction}")
               
        logger.info(f"  🎯 НАПРАВЛЕНИЕ ПЕРЕД РАСЧЕТОМ ЦЕЛЕЙ: {direction}")        

        # ===== ФИЛЬТР КАЧЕСТВА СИГНАЛА =====
        signal_quality = 0
        quality_reasons = []
        
        # Проверяем наличие сильных уровней на старших ТФ
        if senior_tf_analysis.get('has_senior_level', False):
            signal_quality += 2
            quality_reasons.append("✅ Есть сильный уровень на старших ТФ")
        
        # Проверяем наличие FVG
        if fvg_analysis.get('has_fvg', False):
            signal_quality += 2
            quality_reasons.append("✅ Есть FVG")
        
        # Проверяем наличие конфлюенции (2+ уровней)
        if potential_analysis and potential_analysis.get('level_count', 0) >= 2:
            signal_quality += 2
            quality_reasons.append(f"✅ Есть конфлюенция ({potential_analysis['level_count']} уровней)")
        
        # Проверяем наличие подтвержденного пробоя
        if breakout_confirmed:
            signal_quality += 3
            quality_reasons.append("✅ Есть подтвержденный пробой")
        
        # Проверяем наличие пробоя наклонного уровня на 5м/15м
        if trendline_breakout_5m or trendline_breakout_15m:
            signal_quality += 2
            quality_reasons.append("✅ Есть пробой наклонного уровня")
        
        # Проверяем наличие RSI дивергенции
        if rsi_divergence:
            signal_quality += 2
            quality_reasons.append("✅ Есть RSI дивергенция")
        
        # Проверяем наличие накопления
        if accumulation_analysis and accumulation_analysis.get('has_accumulation', False):
            signal_quality += 2
            quality_reasons.append("✅ Есть накопление")
        
        # Логируем качество
        logger.info(f"  📊 Качество сигнала: {signal_quality}/10")
        for qr in quality_reasons[:3]:
            logger.info(f"     {qr}")
        
        # Если качество ниже порога — отменяем сигнал
        MIN_QUALITY = 3  # минимум 3 балла
        if signal_quality < MIN_QUALITY:
            reasons.append(f"⚠️ Низкое качество сигнала ({signal_quality}/10) — требуется минимум {MIN_QUALITY}")
            direction = 'NEUTRAL'
            logger.info(f"  ⏭️ {symbol} - Сигнал отменен: низкое качество ({signal_quality}/10)")
        
        # ===== РАСЧЕТ ЦЕЛЕЙ ПО ATR С ДИНАМИЧЕСКИМИ НАСТРОЙКАМИ =====
        from config import DYNAMIC_TARGET_SETTINGS
        
        atr = last['atr'] if pd.notna(last['atr']) else (last['high'] - last['low']) * 0.3
        current_price = last['close']
        
        # Проверяем, идеальный ли сетап (все ТФ согласованы)
        is_perfect_setup = False
        if tf_alignment.get('percentage', 0) >= 100:
            is_perfect_setup = True
            reasons.append(f"🚀 ИДЕАЛЬНЫЙ СЕТАП: все доступные ТФ согласованы")
        
        # Проверяем, сильный ли тренд (недельный тренд + EMA 200)
        is_strong_trend = False
        if DYNAMIC_TARGET_SETTINGS.get('strong_trend', {}).get('enabled', True):
            weekly_trend = alignment.get('weekly_trend')
            if weekly_trend:
                # Проверяем, есть ли сигнал о EMA 200
                has_ema_200 = any('EMA 200' in s for s in alignment.get('signals', []))
                if has_ema_200:
                    is_strong_trend = True
                    reasons.append(f"📈 Сильный недельный тренд (выше/ниже EMA 200)")
        
        # Выбираем множители
        # from config import ACCUMULATION_SIGNAL_SETTINGS
        
        # Для накопления — специальные настройки
        if signal_type == 'accumulation':
            target_1_mult = ACCUMULATION_SIGNAL_SETTINGS['target_1_multiplier']
            target_2_mult = ACCUMULATION_SIGNAL_SETTINGS['target_2_multiplier']
            stop_mult = ACCUMULATION_SIGNAL_SETTINGS['stop_multiplier']
            reasons.append(f"📦 Накопление: увеличенные цели (x{target_1_mult:.1f}, x{target_2_mult:.1f} ATR) и стоп (x{stop_mult:.1f} ATR)")

        # Для памп-дамп не используем увеличенные цели
        if signal_type in ['PUMP', 'DUMP', 'pump']:
            target_1_mult = DYNAMIC_TARGET_SETTINGS['default']['target_1_mult']
            target_2_mult = DYNAMIC_TARGET_SETTINGS['default']['target_2_mult']
            stop_mult = DYNAMIC_TARGET_SETTINGS['default']['stop_mult']
            if is_strong_trend:
                stop_mult = DYNAMIC_TARGET_SETTINGS['strong_trend']['stop_mult']
            reasons.append(f"📊 Цели для памп-сигнала (стандартные)")
        elif is_perfect_setup:
            target_1_mult = DYNAMIC_TARGET_SETTINGS['perfect_setup']['target_1_mult']
            target_2_mult = DYNAMIC_TARGET_SETTINGS['perfect_setup']['target_2_mult']
            stop_mult = DYNAMIC_TARGET_SETTINGS['perfect_setup']['stop_mult']
            reasons.append(f"🎯 Увеличенные цели для идеального сетапа (x{target_1_mult:.1f}, x{target_2_mult:.1f} ATR)")
        elif is_strong_trend:
            target_1_mult = DYNAMIC_TARGET_SETTINGS['strong_trend']['target_1_mult']
            target_2_mult = DYNAMIC_TARGET_SETTINGS['strong_trend']['target_2_mult']
            stop_mult = DYNAMIC_TARGET_SETTINGS['strong_trend']['stop_mult']
            reasons.append(f"📈 Широкий стоп для сильного тренда (x{stop_mult:.1f} ATR)")
        else:
            target_1_mult = DYNAMIC_TARGET_SETTINGS['default']['target_1_mult']
            target_2_mult = DYNAMIC_TARGET_SETTINGS['default']['target_2_mult']
            stop_mult = DYNAMIC_TARGET_SETTINGS['default']['stop_mult']
        
        # Используем ATR старшего ТФ если нужно
        if DYNAMIC_TARGET_SETTINGS.get('use_higher_tf_atr', True):
            higher_tf = DYNAMIC_TARGET_SETTINGS.get('higher_tf', 'hourly')
            if higher_tf in dataframes and dataframes[higher_tf] is not None:
                df_higher = dataframes[higher_tf]
                if 'atr' in df_higher.columns and pd.notna(df_higher['atr'].iloc[-1]):
                    atr = df_higher['atr'].iloc[-1]
                    reasons.append(f"📊 Стоп рассчитан по ATR {higher_tf} ТФ")
        
        logger.info(f"  🔍 DIRECTION ПЕРЕД РАСЧЕТОМ ЦЕЛЕЙ: '{direction}'")
        logger.info(f"  🔍 'LONG' in direction: {'LONG' in direction}")
        logger.info(f"  🔍 'SHORT' in direction: {'SHORT' in direction}")

        # Расчет целей (более надежное определение направления)
        is_long = 'LONG' in direction and 'SHORT' not in direction
        if is_long:
            target_1 = current_price + atr * target_1_mult
            target_2 = current_price + atr * target_2_mult
            stop_loss = current_price - atr * stop_mult
        else:
            target_1 = current_price - atr * target_1_mult
            target_2 = current_price - atr * target_2_mult
            stop_loss = current_price + atr * stop_mult
        
        targets = {
            'target_1': target_1,
            'target_2': target_2,
            'stop_loss': stop_loss,
        }
        
        logger.info(f"  🎯 НАПРАВЛЕНИЕ ПОСЛЕ РАСЧЕТА ЦЕЛЕЙ: {direction}")

        # Округление целей
        for key in ['target_1', 'target_2', 'stop_loss']:
            if current_price < 0.0001:
                targets[key] = round(targets[key], 8)
            elif current_price < 0.001:
                targets[key] = round(targets[key], 6)
            elif current_price < 0.01:
                targets[key] = round(targets[key], 5)
            elif current_price < 0.1:
                targets[key] = round(targets[key], 4)
            elif current_price < 1:
                targets[key] = round(targets[key], 3)
            else:
                targets[key] = round(targets[key], 2)
        
        logger.info(f"  📈 {symbol} - ATR: {atr}, Цели: {targets}")        

        # ===== РАСЧЕТ ЗОН ДОП.ВХОДА =====
        from config import ENTRY_ZONES_SETTINGS
        
        entry_zones = []
        zone_descriptions = []
        
        # Выбираем настройки в зависимости от типа сигнала
        if signal_type == 'accumulation':
            zone_settings = ENTRY_ZONES_SETTINGS['accumulation']
        elif signal_type in ['PUMP', 'DUMP', 'pump']:
            zone_settings = ENTRY_ZONES_SETTINGS['pump']
        else:
            zone_settings = ENTRY_ZONES_SETTINGS['regular']
        
        lookback = ENTRY_ZONES_SETTINGS['lookback']
        
        # Зона 1
        tf1 = zone_settings.get('zone_1_tf')
        if tf1 and tf1 in dataframes and dataframes[tf1] is not None:
            df1 = dataframes[tf1]
            if 'LONG' in direction:
                zone1 = df1['low'].tail(lookback['zone_1']).min()
                desc1 = f"минимум {tf1}"
            else:
                zone1 = df1['high'].tail(lookback['zone_1']).max()
                desc1 = f"максимум {tf1}"
            entry_zones.append(zone1)
            zone_descriptions.append(desc1)
        
        # Зона 2
        tf2 = zone_settings.get('zone_2_tf')
        if tf2 and tf2 in dataframes and dataframes[tf2] is not None:
            df2 = dataframes[tf2]
            if 'LONG' in direction:
                zone2 = df2['low'].tail(lookback['zone_2']).min()
                desc2 = f"минимум {tf2}"
            else:
                zone2 = df2['high'].tail(lookback['zone_2']).max()
                desc2 = f"максимум {tf2}"
            entry_zones.append(zone2)
            zone_descriptions.append(desc2)
        
        # Зона 3 (опционально)
        tf3 = zone_settings.get('zone_3_tf')
        if tf3 and tf3 in dataframes and dataframes[tf3] is not None:
            df3 = dataframes[tf3]
            if 'LONG' in direction:
                zone3 = df3['low'].tail(lookback['zone_3']).min()
                desc3 = f"минимум {tf3}"
            else:
                zone3 = df3['high'].tail(lookback['zone_3']).max()
                desc3 = f"максимум {tf3}"
            entry_zones.append(zone3)
            zone_descriptions.append(desc3)
        
        # Форматируем зоны для отображения с описанием
        formatted_zones = []
        for i, zone in enumerate(entry_zones):
            if current_price < 0.0001:
                zone_str = f"{zone:.8f}".rstrip('0').rstrip('.')
            elif current_price < 0.001:
                zone_str = f"{zone:.6f}".rstrip('0').rstrip('.')
            elif current_price < 0.01:
                zone_str = f"{zone:.5f}".rstrip('0').rstrip('.')
            elif current_price < 0.1:
                zone_str = f"{zone:.4f}".rstrip('0').rstrip('.')
            elif current_price < 1:
                zone_str = f"{zone:.3f}".rstrip('0').rstrip('.')
            else:
                zone_str = f"{zone:.2f}".rstrip('0').rstrip('.')
            
            if i < len(zone_descriptions):
                formatted_zones.append(f"{zone_str} ({zone_descriptions[i]})")
            else:
                formatted_zones.append(zone_str)

        # ===== АНАЛИЗ ПАТТЕРНОВ =====
        pattern_analysis = None
        if PATTERN_SETTINGS.get('enabled', True):
            try:
                logger.info(f"  🔍 {symbol} - Анализ паттернов")
                logger.info(f"  🔍 ПАТТЕРНЫ: вызываю analyze_multi_timeframe для {symbol}")
                pattern_analysis = self.pattern_analyzer.analyze_multi_timeframe(dataframes)
                logger.info(f"  🔍 ПАТТЕРНЫ: результат has_pattern={pattern_analysis.get('has_pattern')}")
                
                if pattern_analysis['has_pattern']:
                    logger.info(f"  📊 Найдено паттернов: {len(pattern_analysis['patterns'])}")
                    for pattern in pattern_analysis['patterns']:
                        logger.info(f"     → {pattern.get('description', 'НЕТ ОПИСАНИЯ')}")
                        reasons.insert(0, pattern['description'])
                        logger.info(f"  📝 Добавлена причина от паттерна: {pattern['description']}")
                    confidence += pattern_analysis['strength'] / 5
                    
                    # Если паттерн даёт направление — используем его
                    if pattern_analysis['direction']:
                        old_dir = direction
                        direction = pattern_analysis['direction']
                        logger.info(f"  🎯 Направление от паттерна: {old_dir} → {direction}")
                        
            except Exception as e:
                logger.error(f"❌ Ошибка в анализе паттернов для {symbol}: {e}")

        # ✅ ЛОГИРОВАНИЕ ПОСЛЕ ПАТТЕРНОВ
        logger.info(f"  📊 Все причины ПОСЛЕ паттернов: {reasons}")
        
        # ✅ НОРМАЛИЗАЦИЯ УВЕРЕННОСТИ
        if confidence > 100:
            confidence = min(100, confidence)
            logger.info(f"  📊 Нормализована уверенность: {confidence:.1f}%")

        # ===== ФОРМИРОВАНИЕ РЕЗУЛЬТАТА =====
        result = {
            'symbol': symbol,
            'exchange': exchange,
            'price': current_price,
            'direction': direction,
            'atr': atr,
            'signal_type': signal_type,
            'signal_power': self._get_power_text(confidence),
            'confidence': round(confidence, 1),
            'signal_strength': round((confidence + alignment['trend_alignment']) / 2, 1),
            'reasons': reasons[:8],
            'funding_rate': metadata.get('funding_rate', 0),
            'volume_24h': metadata.get('volume_24h', 0),
            'price_change_24h': metadata.get('price_change_24h', 0),
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'alignment': alignment,
            'bearish_score': bearish_score,  # ← ДОБАВИТЬ
            'bullish_score': bullish_score,  # ← ДОБАВИТЬ (опционально)
            'entry_zones': formatted_zones,
            'tf_alignment_percentage': tf_alignment['percentage'],
            'tf_aligned_count': tf_alignment['aligned_count'],
            'tf_total_count': tf_alignment['total_count'],
            **targets
        }
        
        # Добавляем зоны FVG для графика
        if fvg_analysis['has_fvg'] and 'zones' in fvg_analysis:
            result['fvg_zones'] = fvg_analysis['zones']
            logger.info(f"  🎨 Добавлено {len(fvg_analysis['zones'])} FVG зон для графика")
        
        # Добавляем зоны дисперсии для графика
        if DISPERSION_ANALYSIS_SETTINGS['enabled'] and 'dispersion_zones' in locals() and dispersion_zones:
            result['dispersion_zones'] = dispersion_zones
            logger.info(f"  🎨 Добавлено {len(dispersion_zones)} зон дисперсии для графика")

        if fib_analysis:
            result['fibonacci'] = fib_analysis
        if accumulation_analysis:
            result['accumulation'] = accumulation_analysis
        
        logger.info(f"✅ generate_signal успешно завершен для {symbol}")
        return result
    
    def _get_power_text(self, confidence: float) -> str:
        """Определение текста силы сигнала по уверенности"""
        if confidence >= 85:
            return "🔥🔥🔥 ОЧЕНЬ СИЛЬНЫЙ"
        elif confidence >= 70:
            return "🔥🔥 СИЛЬНЫЙ"
        elif confidence >= 55:
            return "🔥 СРЕДНИЙ"
        elif confidence >= 40:
            return "📊 СЛАБЫЙ"
        else:
            return "👀 НАБЛЮДЕНИЕ"

# ============== БЫСТРЫЙ ПАМП-СКАНЕР ==============

class FastPumpScanner:
    def __init__(self, fetcher: BaseExchangeFetcher, settings: Dict = None, analyzer=None, telegram_bot=None, chart_generator=None):
        self.fetcher = fetcher
        self.settings = settings or PUMP_SCAN_SETTINGS
        self.analyzer = analyzer
        self.telegram_bot = telegram_bot  # ✅ Добавляем telegram_bot
        self.chart_generator = chart_generator  # ✅ Добавляем chart_generator
        self.threshold = self.settings.get('threshold', 3.0)
        self.instant_threshold = self.settings.get('instant_threshold', 1.0)  # Снижено до 1%
        self.shitcoin_instant_threshold = self.settings.get('shitcoin_instant_threshold', 0.8)  # Для щиткоинов 0.8%
        self.shitcoin_volume_threshold = self.settings.get('shitcoin_volume_threshold', 1_000_000)  # 1M$ = порог щиткоина
        self.timeframes = self.settings.get('timeframes', ['1m', '3m', '5m', '15m', '30m'])
        self.max_pairs = self.settings.get('max_pairs_to_scan', 600)
        self.websocket_top_pairs = self.settings.get('websocket_top_pairs', 100)
        self.last_pump_signals = {}
        self.cache = CacheManager(ttl=30)
        self.ws_signals_sent = set()  # отслеживаем отправленные через WebSocket сигналы
        # self.batch_size = PUMP_SCAN_SETTINGS.get('batch_size', 100)
        # self.delay_between_batches = PUMP_SCAN_SETTINGS.get('delay_between_batches', 0.1)
        
        # WebSocket менеджер
        try:
            from websocket_manager import BingXWebSocketManager
            self.ws_manager = BingXWebSocketManager(
                os.getenv('BINGX_API_KEY'),
                os.getenv('BINGX_SECRET_KEY')
            )
            self.websocket_available = True
            logger.info("✅ WebSocket менеджер инициализирован")
        except ImportError as e:
            logger.warning(f"⚠️ WebSocketManager не инициализирован: {e}")
            self.ws_manager = None
            self.websocket_available = False
        
        self.batch_size = PERFORMANCE_SETTINGS.get('pump_batch_size', 50)
        self.delay_between_batches = PERFORMANCE_SETTINGS.get('delay_between_batches', 0.5)
        self.websocket_reconnect_delay = self.settings.get('websocket_reconnect_delay', 5)
        
        # Очередь для быстрых сигналов
        self.instant_signals_queue = asyncio.Queue()
        
        logger.info(f"✅ FastPumpScanner инициализирован (WebSocket: {self.websocket_available})")
        logger.info(f"   Пороги: мейджоры {self.instant_threshold}%, щиткоины {self.shitcoin_instant_threshold}%")
    
    async def start_websocket_monitoring(self, symbols: List[str]):
        """
        Запуск WebSocket мониторинга с приоритетом на щиткоины
        """
        if not self.websocket_available or not self.ws_manager:
            logger.info("WebSocket мониторинг недоступен, используется REST API")
            return
        
        # Получаем щиткоины с малым объемом
        shitcoins = await self._get_volatile_shitcoins(symbols)
        logger.info(f"🎯 Найдено щиткоинов: {len(shitcoins)}")

        # Берем топ-5 мейджоров для контроля
        majors = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT']
        
        # Объединяем: сначала все щиткоины, потом мейджоры
        all_priority = shitcoins + [m for m in majors if m not in shitcoins]
        
        # Ограничиваем до 100 пар
        priority_symbols = all_priority[:self.websocket_top_pairs]
        logger.info(f"🎯 Итоговых монет для WebSocket: {len(priority_symbols)}")
        
        shitcoin_count = sum(1 for s in priority_symbols if s in shitcoins)
        major_count = len(priority_symbols) - shitcoin_count
        
        logger.info(f"🎯 WebSocket мониторинг: {len(priority_symbols)} пар")
        logger.info(f"   - Щиткоины: {shitcoin_count} (объем < {self.shitcoin_volume_threshold/1_000_000:.1f}M$)")
        logger.info(f"   - Мейджоры: {major_count}")

        # Запускаем WebSocket с callback
        await self.ws_manager.connect_ticker_stream(
            priority_symbols,
            self.handle_instant_signal
        )
        
        # Запускаем обработчик очереди
        asyncio.create_task(self.process_instant_signals())
        logger.info(f"📡 WebSocket мониторинг запущен")
    
    async def _get_volatile_shitcoins(self, all_symbols: List[str]) -> List[str]:
        """
        Определение самых волатильных щиткоинов на основе объема
        """
        shitcoins = []
        volumes = []
        
        logger.info("🔍 Сканирую щиткоины...")
        
        # Черный список мейджоров
        blacklist = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'DOT', 'LINK', 'MATIC', 'AVAX', 'UNI', 'SHIB']
        
        # Проверяем объемы (берем первые 300 пар для скорости)
        for symbol in all_symbols[:300]:
            try:
                coin = symbol.split('/')[0].upper()
                
                # Пропускаем мейджоры
                if coin in blacklist:
                    continue
                
                ticker = await self.fetcher.fetch_ticker(symbol)
                volume = ticker.get('volume_24h', 0)
                
                # Если объем меньше порога - это щиткоин
                if volume < self.shitcoin_volume_threshold:
                    shitcoins.append(symbol)
                    volumes.append((symbol, volume))
            except Exception as e:
                continue
        
        # Сортируем по объему (от самых маленьких - самых волатильных)
        volumes.sort(key=lambda x: x[1])
        top_shitcoins = [s for s, v in volumes[:300]]  # берем 150 самых маленьких
        
        logger.info(f"🎯 Найдено {len(top_shitcoins)} щиткоинов с объемом < {self.shitcoin_volume_threshold/1_000_000:.1f}M$")
        return top_shitcoins
    
    async def handle_instant_signal(self, signal_type: str, symbol: str, price: float, movement: Dict):
        """
        Обработка мгновенного сигнала от WebSocket с гибкими настройками
        """
        try:
            from config import PUMP_MECHANISM_SETTINGS
            
            mode = PUMP_MECHANISM_SETTINGS.get('mode', 'new_only')
            
            # Определяем, щиткоин или нет
            is_shitcoin = False
            try:
                ticker = await self.fetcher.fetch_ticker(symbol)
                volume = ticker.get('volume_24h', 1_000_000)
                if volume < self.shitcoin_volume_threshold:
                    is_shitcoin = True
            except:
                pass
            
            # ===== НОВЫЙ МЕХАНИЗМ (уже проверил в _check_instant_movement) =====
            new_passed = True  # новый механизм уже проверил движение
            
            # ===== СТАРЫЙ МЕХАНИЗМ (проверяем порог) =====
            old_passed = False
            if mode in ['old_only', 'both']:
                if is_shitcoin:
                    threshold = self.settings.get('shitcoin_instant_threshold', 1.5)
                else:
                    threshold = self.settings.get('instant_threshold', 2.0)
                
                if abs(movement['change_percent']) >= threshold:
                    old_passed = True
            
            # ===== ЛОГИКА ОТПРАВКИ =====
            should_send = False
            
            if mode == 'new_only':
                should_send = new_passed
            elif mode == 'old_only':
                should_send = old_passed
            elif mode == 'both':
                both_settings = PUMP_MECHANISM_SETTINGS.get('both_settings', {})
                if both_settings.get('require_both', False):
                    should_send = new_passed and old_passed  # нужны оба
                else:
                    should_send = new_passed or old_passed  # достаточно одного
            
            if should_send:
                logger.info(f"⚡ {symbol}: {movement['change_percent']:+.1f}% за {movement['time_window']:.1f} сек")
                
                await self.instant_signals_queue.put({
                    'symbol': symbol,
                    'price': price,
                    'movement': movement,
                    'is_shitcoin': is_shitcoin,
                    'time': datetime.now()
                })
            else:
                logger.debug(f"⏭️ {symbol}: движение {movement['change_percent']:.1f}% не прошло фильтр")
                
        except Exception as e:
            logger.error(f"Ошибка обработки мгновенного сигнала: {e}")
    
    async def process_instant_signals(self):
        """
        Обработка очереди мгновенных сигналов
        """
        while True:
            try:
                signal_data = await self.instant_signals_queue.get()
                
                # Отправляем быстрый предварительный сигнал
                await self.send_flash_signal(signal_data)
                
                # Запускаем полный анализ в фоне
                asyncio.create_task(self.confirm_signal(signal_data))
                
            except Exception as e:
                logger.error(f"Ошибка обработки очереди: {e}")
                await asyncio.sleep(1)
    
    async def send_flash_signal(self, signal_data: Dict):
        """
        Отправка быстрого предварительного сигнала (0-2 секунды)
        """
        symbol = signal_data['symbol']

        # ✅ Запоминаем, что сигнал отправлен
        if not hasattr(self, 'ws_signals_sent'):
            self.ws_signals_sent = set()
        
        self.ws_signals_sent.add(symbol)
        # Через 60 секунд удаляем из памяти
        asyncio.create_task(self._remove_from_ws_cache(symbol, 60))

        movement = signal_data['movement']
        coin = symbol.split('/')[0].replace('USDT', '')
        is_shitcoin = signal_data.get('is_shitcoin', False)
        
        # Эмодзи для щиткоинов - ⚡, для мейджоров - 🚀
        if is_shitcoin:
            direction_emoji = "⚡"
            coin_type = " [ЩИТКОИН]"
        else:
            direction_emoji = "🚀" if movement['change_percent'] > 0 else "📉"
            coin_type = ""
        
        msg = (
            f"{direction_emoji} <code>{coin}</code>{coin_type} {movement['change_percent']:+.1f}% за {movement['time_window']:.1f} сек\n"
            f"⏳ Полный анализ через 3-5 секунд...\n"
            f"💰 Цена: {signal_data['price']:.6f}"
        )
        
        # Создаем простую клавиатуру
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton(f"📋 Копировать {coin}", callback_data=f"copy_{coin}")
        ]])
        
        # Отправляем в памп-группу
        try:
            await self.fetcher.telegram_bot.send_message(
                chat_id=PUMP_CHAT_ID,
                text=msg,
                parse_mode='HTML',
                reply_markup=keyboard
            )
            logger.info(f"⚡ Отправлен мгновенный сигнал для {symbol} (щиткоин: {is_shitcoin})")
        except Exception as e:
            logger.error(f"Ошибка отправки мгновенного сигнала: {e}")
    
    async def _remove_from_ws_cache(self, symbol: str, delay: int):
        """Удаление символа из кэша WebSocket сигналов"""
        await asyncio.sleep(delay)
        if hasattr(self, 'ws_signals_sent') and symbol in self.ws_signals_sent:
            self.ws_signals_sent.remove(symbol)
            logger.info(f"🗑️ {symbol} удален из кэша WebSocket сигналов")

    async def confirm_signal(self, signal_data: Dict):
        """
        Подтверждение сигнала полным анализом
        """
        symbol = signal_data['symbol']
        
        # Ждем немного для накопления данных
        await asyncio.sleep(3)
        
        try:
            # Загружаем данные для полного анализа
            dataframes = {}
            for tf_name, tf_value in TIMEFRAMES.items():
                limit_tf = 200 if tf_name == 'current' else 100
                df_tf = await self.fetcher.fetch_ohlcv(symbol, tf_value, limit_tf)
                if df_tf is not None and not df_tf.empty:
                    df_tf = self.analyzer.calculate_indicators(df_tf)
                    dataframes[tf_name] = df_tf
            
            if not dataframes:
                logger.warning(f"⚠️ Нет данных для подтверждения {symbol}")
                return
            
            # Получаем метаданные
            funding = await self.fetcher.fetch_funding_rate(symbol)
            ticker = await self.fetcher.fetch_ticker(symbol)
            
            metadata = {
                'funding_rate': funding,
                'volume_24h': ticker.get('volume_24h'),
                'price_change_24h': ticker.get('percentage')
            }
            
            # Генерируем полный сигнал
            signal = self.analyzer.generate_signal(dataframes, metadata, symbol, self.fetcher.name)
            
            if signal and 'NEUTRAL' not in signal['direction']:
                # Добавляем информацию о быстром движении
                signal['pump_dump'] = [{
                    'change_percent': signal_data['movement']['change_percent'],
                    'time_window': signal_data['movement']['time_window'],
                    'start_price': signal_data['movement']['start_price'],
                    'end_price': signal_data['movement']['end_price']
                }]
                
                if signal_data['movement']['change_percent'] > 0:
                    signal['signal_type'] = "PUMP"
                else:
                    signal['signal_type'] = "DUMP"
                
                # Убедимся, что funding_rate не потерялся
                signal['funding_rate'] = funding
                
                # Отправляем подтвержденный сигнал
                contract_info = await self.fetcher.fetch_contract_info(symbol)
                msg, keyboard = self.format_pump_message(signal, contract_info)
                
                # Отправляем подтвержденный сигнал
                try:
                    await self.fetcher.telegram_bot.send_message(
                        chat_id=PUMP_CHAT_ID,
                        text=f"✅ ПОДТВЕРЖДЕНО\n\n{msg}",
                        parse_mode='HTML',
                        reply_markup=keyboard
                    )
                    logger.info(f"✅ Подтвержден сигнал для {symbol}")
                except Exception as e:
                    logger.error(f"Ошибка отправки подтвержденного сигнала: {e}")
                
        except Exception as e:
            logger.error(f"❌ Ошибка подтверждения сигнала {symbol}: {e}")
    
    async def scan_pair(self, pair: str) -> Optional[Dict]:
        """
        Сканирование одной пары (для параллельного вызова REST API)
        """
        try:
            # Проверяем кэш
            cache_key = f"{pair}_pump"
            cached = self.cache.get(cache_key)
            if cached:
                return cached
            
            for tf in self.timeframes:
                limit = 20
                df = await self.fetcher.fetch_ohlcv(pair, tf, limit=limit)
                
                if df is None or len(df) < 10:
                    continue
                
                bars_ago = 1
                minutes = self._timeframe_to_minutes(tf)
                
                start_price = df['close'].iloc[-bars_ago-1]
                current_price = df['close'].iloc[-1]
                change_percent = (current_price - start_price) / start_price * 100
                
                if abs(change_percent) >= self.threshold:
                    signal_key = f"{pair}_{tf}"
                    last_time = self.last_pump_signals.get(signal_key)
                    
                    if last_time and (datetime.now() - last_time).total_seconds() < (self.settings.get('cooldown_minutes', 10) * 60):
                        continue
                    
                    if self.analyzer:
                        dataframes = {}
                        for tf_name, tf_value in TIMEFRAMES.items():
                            limit_tf = 200 if tf_name == 'current' else 100
                            df_tf = await self.fetcher.fetch_ohlcv(pair, tf_value, limit_tf)
                            if df_tf is not None and not df_tf.empty:
                                df_tf = self.analyzer.calculate_indicators(df_tf)
                                dataframes[tf_name] = df_tf
                        
                        if dataframes:
                            funding = await self.fetcher.fetch_funding_rate(pair)
                            ticker = await self.fetcher.fetch_ticker(pair)
                            
                            metadata = {
                                'funding_rate': funding,
                                'volume_24h': ticker.get('volume_24h'),
                                'price_change_24h': ticker.get('percentage')
                            }
                            
                            signal = self.analyzer.generate_signal(dataframes, metadata, pair, self.fetcher.name)
                            
                            if signal and 'NEUTRAL' not in signal['direction']:
                                signal['pump_dump'] = [{
                                    'change_percent': change_percent,
                                    'time_window': minutes,
                                    'start_price': start_price,
                                    'end_price': current_price
                                }]
                                
                                if change_percent > 0:
                                    signal['signal_type'] = "PUMP"
                                else:
                                    signal['signal_type'] = "DUMP"
                                
                                signal['funding_rate'] = funding
                                
                                # ✅ ОТПРАВЛЯЕМ СИГНАЛ С ГРАФИКОМ!
                                try:
                                    contract_info = await self.fetcher.fetch_contract_info(pair)
                                    msg, keyboard = self.format_pump_message(signal, contract_info)
                                    
                                    # Загружаем данные для графика
                                    df = await self.fetcher.fetch_ohlcv(pair, TIMEFRAMES.get('current', '15m'), limit=200)
                                    
                                    coin = pair.split('/')[0].replace('USDT', '')
                                    
                                    if df is not None and not df.empty:
                                        df = self.analyzer.calculate_indicators(df)
                                        chart_buf = self.chart_generator.create_chart(df, signal, coin, TIMEFRAMES.get('current', '15m'))
                                        
                                        await self.telegram_bot.send_photo(
                                            chat_id=PUMP_CHAT_ID,
                                            photo=chart_buf,
                                            caption=msg,
                                            parse_mode='HTML',
                                            reply_markup=keyboard
                                        )
                                        logger.info(f"✅ Отправлен памп-сигнал с графиком: {pair}")
                                    else:
                                        await self.telegram_bot.send_message(
                                            chat_id=PUMP_CHAT_ID,
                                            text=msg,
                                            parse_mode='HTML',
                                            reply_markup=keyboard
                                        )
                                        logger.info(f"✅ Отправлен памп-сигнал (без графика): {pair}")
                                        
                                except Exception as e:
                                    logger.error(f"❌ Ошибка отправки сигнала {pair}: {e}")
                                
                                self.cache.set(cache_key, signal)
                                self.last_pump_signals[signal_key] = datetime.now()
                                
                                return signal
            return None
        except Exception as e:
            logger.error(f"Ошибка сканирования {pair}: {e}")
            return None
    
    async def scan_all_pairs(self) -> List[Dict]:
        """
        Оптимизированное сканирование всех пар с гибкими настройками режима
        """
        logger.info("🚀 ЗАПУСК БЫСТРОГО ПАМП-СКАНЕРА (ГИБРИДНЫЙ)")
        
        try:
            all_pairs = await self.fetcher.fetch_all_pairs()
            if not all_pairs:
                return []
            
            # ✅ Загружаем настройки
            from config import SMART_REPEAT_SETTINGS, SCAN_MODE
            
            smart_repeat = SMART_REPEAT_SETTINGS
            
            # ✅ Словарь для отслеживания последних сигналов по монетам
            last_signals = {}  # coin: {'time': datetime, 'change': float, 'direction': str}
            
            # ===== ВЫБОР РЕЖИМА СКАНИРОВАНИЯ =====
            scan_pairs = []
            mode = SCAN_MODE.get('mode', 'all')
            
            # РЕЖИМ 1: ТОП ПО ОБЪЕМУ
            if mode == 'top_volume':
                logger.info("📊 Режим: топ-объем")
                pairs_with_volume = []
                
                # Берем первые 300 пар для скорости
                for pair in all_pairs[:300]:
                    try:
                        ticker = await self.fetcher.fetch_ticker(pair)
                        volume = ticker.get('volume_24h', 0)
                        if volume >= SCAN_MODE['top_volume']['min_volume']:
                            pairs_with_volume.append((pair, volume))
                    except:
                        continue
                
                pairs_with_volume.sort(key=lambda x: x[1], reverse=True)
                scan_pairs = [p[0] for p in pairs_with_volume[:SCAN_MODE['top_volume']['count']]]
                logger.info(f"  📊 Отобрано {len(scan_pairs)} пар по объему")
            
            # РЕЖИМ 2: ЩИТКОИНЫ
            elif mode == 'shitcoin':
                logger.info("📊 Режим: щиткоины")
                shitcoins = await self._get_volatile_shitcoins(all_pairs)
                scan_pairs = shitcoins[:SCAN_MODE['shitcoin']['count']]
                
                # Добавляем мейджоры если нужно
                if SCAN_MODE['shitcoin']['include_majors']:
                    majors = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 
                            'SOL/USDT:USDT', 'XRP/USDT:USDT']
                    scan_pairs = majors[:SCAN_MODE['shitcoin']['majors_count']] + scan_pairs
                
                logger.info(f"  📊 Отобрано {len(scan_pairs)} щиткоинов")
            
            # РЕЖИМ 3: ГИБРИДНЫЙ
            elif mode == 'hybrid':
                logger.info("📊 Режим: гибридный")
                
                # Топ по объему
                pairs_with_volume = []
                for pair in all_pairs[:300]:
                    try:
                        ticker = await self.fetcher.fetch_ticker(pair)
                        volume = ticker.get('volume_24h', 0)
                        pairs_with_volume.append((pair, volume))
                    except:
                        continue
                pairs_with_volume.sort(key=lambda x: x[1], reverse=True)
                top_volume = [p[0] for p in pairs_with_volume[:SCAN_MODE['hybrid']['top_volume_count']]]
                
                # Щиткоины
                shitcoins = await self._get_volatile_shitcoins(all_pairs)
                shitcoins = shitcoins[:SCAN_MODE['hybrid']['shitcoin_count']]
                
                # Объединяем
                scan_pairs = list(set(top_volume + shitcoins))
                logger.info(f"  📊 Отобрано {len(scan_pairs)} пар (топ-объем: {len(top_volume)}, щиткоины: {len(shitcoins)})")
            
            # РЕЖИМ 4: ВСЕ ПАРЫ
            else:
                logger.info("📊 Режим: все пары")
                scan_pairs = all_pairs[:self.max_pairs]
            
            # Перемешиваем если нужно
            if SCAN_MODE.get('randomize', True):
                random.shuffle(scan_pairs)
            
            # Запускаем WebSocket мониторинг для быстрых сигналов
            if self.websocket_available:
                await self.start_websocket_monitoring(all_pairs)  # отправляем все пары для WebSocket
            
            logger.info(f"📊 Памп-сканер: анализирую {len(scan_pairs)} пар (WebSocket: {self.websocket_available})")
            
            pump_signals = []
            
            # Разбиваем на батчи для параллельной обработки
            batches = [scan_pairs[i:i+self.batch_size] for i in range(0, len(scan_pairs), self.batch_size)]
            
            for batch_num, batch in enumerate(batches):
                logger.info(f"🔄 Обработка батча {batch_num + 1}/{len(batches)} ({len(batch)} пар)")
                
                # Параллельная обработка батча
                tasks = [self.scan_pair(pair) for pair in batch]
                batch_results = await asyncio.gather(*tasks)
                
                # Собираем результаты с умной фильтрацией
                for signal in batch_results:
                    if not signal:
                        continue
                    
                    coin = signal['symbol'].split('/')[0]
                    current_change = abs(signal['pump_dump'][0]['change_percent'])
                    current_direction = 'LONG' if signal['pump_dump'][0]['change_percent'] > 0 else 'SHORT'
                    
                    # ===== УМНАЯ ЛОГИКА ПОВТОРОВ =====
                    if smart_repeat['enabled'] and coin in last_signals:
                        last = last_signals[coin]
                        time_diff = (datetime.now() - last['time']).total_seconds() / 60  # в минутах
                        
                        # Базовая проверка cooldown
                        if time_diff < smart_repeat['cooldown_minutes']:
                            # Проверяем, разрешены ли повторы при усилении
                            if smart_repeat['allow_stronger_moves']:
                                # Вычисляем порог усиления
                                required_strength = last['change'] * smart_repeat['strength_multiplier']
                                
                                # Проверяем, усилилось ли движение
                                if current_change > required_strength:
                                    # Проверяем минимальное время до повтора
                                    if time_diff >= smart_repeat['min_time_for_repeat']:
                                        logger.info(f"⚡ УСИЛЕНИЕ {coin}: {last['change']:.1f}% → {current_change:.1f}% (разрешен повтор)")
                                    else:
                                        logger.info(f"⏳ {coin} усилился, но слишком рано ({time_diff:.0f} мин < {smart_repeat['min_time_for_repeat']} мин)")
                                        continue
                                else:
                                    logger.info(f"⏭️ {coin} повтор: нужно > {required_strength:.1f}%, есть {current_change:.1f}%")
                                    continue
                            else:
                                logger.info(f"⏭️ {coin} повтор: cooldown {time_diff:.0f} мин")
                                continue
                        else:
                            logger.info(f"📌 {coin} повтор после {time_diff:.0f} мин (cooldown истек)")
                    
                    # ===== ФИЛЬТР ПО ТИПУ СИГНАЛА (PUMP/DUMP) =====
                    from config import PUMP_DUMP_FILTER

                    if PUMP_DUMP_FILTER.get('enabled', False):
                        filter_type = PUMP_DUMP_FILTER.get('type', 'both')
                        change = signal['pump_dump'][0]['change_percent']
                        
                        if filter_type == 'pump_only' and change < 0:
                            logger.info(f"⏭️ {signal['symbol']} пропущен (только пампы, а это дамп)")
                            continue
                        if filter_type == 'dump_only' and change > 0:
                            logger.info(f"⏭️ {signal['symbol']} пропущен (только дампы, а это памп)")
                            continue

                    # ✅ Сохраняем сигнал в историю
                    last_signals[coin] = {
                        'time': datetime.now(),
                        'change': current_change,
                        'direction': current_direction,
                        'symbol': signal['symbol']
                    }
                    
                    pump_signals.append(signal)
                    logger.info(f"✅ Памп-сигнал (REST): {signal['symbol']} {signal['pump_dump'][0]['change_percent']:+.1f}%")
                
                # Пауза между батчами
                if batch_num < len(batches) - 1:
                    await asyncio.sleep(self.delay_between_batches)
            
            # Сортируем по силе движения
            pump_signals.sort(key=lambda x: abs(x['pump_dump'][0]['change_percent']), reverse=True)
            logger.info(f"🎯 Памп-сканер: найдено {len(pump_signals)} сигналов (WebSocket активен)")
            return pump_signals
            
        except Exception as e:
            logger.error(f"❌ Ошибка памп-сканера: {e}")
            return []
    
    def _get_power_text(self, strength: float) -> str:
        """Определение текста силы сигнала"""
        if strength >= 90:
            return "🔥🔥🔥 ОЧЕНЬ СИЛЬНЫЙ"
        elif strength >= 75:
            return "🔥🔥 СИЛЬНЫЙ"
        elif strength >= 60:
            return "🔥 СРЕДНИЙ"
        else:
            return "⚡ СЛАБЫЙ"
    
    def _timeframe_to_minutes(self, tf: str) -> int:
        """Конвертация таймфрейма в минуты"""
        return {'1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30, '1h': 60}.get(tf, 15)
    
    def _format_compact(self, num: float) -> str:
        """Форматирование больших чисел"""
        if num is None:
            return "N/A"
        if num > 1_000_000_000:
            return f"{num/1_000_000_000:.1f}B"
        elif num > 1_000_000:
            return f"{num/1_000_000:.1f}M"
        elif num > 1_000:
            return f"{num/1_000:.1f}K"
        else:
            return f"{num:.0f}"
    
    def format_tf_name(self, tf: str) -> str:   # ← 4 пробела отступ
        """Преобразование названия таймфрейма в читаемый формат"""
        tf_map = SIGNAL_FORMAT_SETTINGS.get('tf_names', {
            'current': '15м',
            'hourly': '1ч',
            'four_hourly': '4ч',
            'daily': '1д',
            'weekly': '1н',
            'monthly': '1м'
        })
        return tf_map.get(tf, tf)
    
    def format_pump_message(self, signal: Dict, contract_info: Dict = None) -> Tuple[str, InlineKeyboardMarkup]:
        """
        Форматирование памп-сигнала для отправки с ПРАВИЛЬНЫМ направлением
        """
        logger.info(f"  📊 Все причины перед фильтром: {signal.get('reasons', [])}")

        logger.info(f"  📊 format_pump_message START: направление до изменений = {signal.get('direction')}")

        coin = signal['symbol'].split('/')[0].replace('USDT', '')
        
        # Получаем данные о пампа
        pump_data = signal.get('pump_dump', [{}])[0]
        pump_change = pump_data.get('change_percent', 0)
        pump_time = pump_data.get('time_window', 0)
        
        # ===== ПРАВИЛЬНАЯ ЛОГИКА НАПРАВЛЕНИЯ =====
        # PUMP + пробой - LONG 📈 (пробой после пампа)
        # PUMP без пробоя - SHORT 📉 (коррекция)
        # DUMP + пробой - SHORT 📉 (пробой после дампа)
        # DUMP без пробоя - LONG 📈 (отскок)

        # Проверяем, есть ли пробой уровня
        has_breakout = False
        if 'reasons' in signal:
            for reason in signal['reasons']:
                if 'Пробой' in reason:
                    has_breakout = True
                    break

        logger.info(f"  📊 format_pump_message: has_breakout = {has_breakout}, pump_change = {pump_change}")
        
        # ===== ПРАВИЛЬНАЯ ЛОГИКА НАПРАВЛЕНИЯ =====
        
        # Логируем исходное направление
        logger.info(f"  📊 format_pump_message START: исходное направление = {signal.get('direction')}")
        logger.info(f"  📊 format_pump_message: has_breakout = {has_breakout}, pump_change = {pump_change}")

        if pump_change > 0:  # PUMP
            if has_breakout:
                old_dir = signal['direction']
                signal_emoji = "🚀"
                signal_text = f"PUMP +{pump_change:.1f}%"
                signal['direction'] = 'LONG 📈 (пробой после пампа)'
                signal['signal_type'] = 'PUMP_BREAKOUT'
                logger.info(f"  📊 format_pump_message: СМЕНА! {old_dir} → {signal['direction']} (PUMP + пробой)")
                # Добавляем причину (если еще нет)
                if 'reasons' in signal:
                    has_pump_reason = any('Пробой уровня' in r or 'Коррекция' in r for r in signal['reasons'])
                    if not has_pump_reason:
                        signal['reasons'].insert(0, f"Пробой уровня после пампа +{pump_change:.1f}%")
            else:
                old_dir = signal['direction']
                signal_emoji = "🚨" if pump_change > 3.0 else "🚀"
                signal_text = f"PUMP +{pump_change:.1f}%"
                signal['direction'] = 'SHORT 📉 (коррекция)'
                signal['signal_type'] = 'PUMP'
                
                # ===== ДОБАВИТЬ ПЕРЕСЧЕТ ЦЕЛЕЙ ДЛЯ SHORT =====
                current_price = signal['price']
                atr = signal.get('atr', 0)
                
                if atr > 0:
                    from config import ATR_SETTINGS
                    signal['target_1'] = current_price - atr * ATR_SETTINGS['short_target_1_mult']
                    signal['target_2'] = current_price - atr * ATR_SETTINGS['short_target_2_mult']
                    signal['stop_loss'] = current_price + atr * ATR_SETTINGS['short_stop_loss_mult']
                    
                    # Округление
                    if current_price < 0.0001:
                        signal['target_1'] = round(signal['target_1'], 8)
                        signal['target_2'] = round(signal['target_2'], 8)
                        signal['stop_loss'] = round(signal['stop_loss'], 8)
                    elif current_price < 0.001:
                        signal['target_1'] = round(signal['target_1'], 6)
                        signal['target_2'] = round(signal['target_2'], 6)
                        signal['stop_loss'] = round(signal['stop_loss'], 6)
                    elif current_price < 0.01:
                        signal['target_1'] = round(signal['target_1'], 5)
                        signal['target_2'] = round(signal['target_2'], 5)
                        signal['stop_loss'] = round(signal['stop_loss'], 5)
                    elif current_price < 0.1:
                        signal['target_1'] = round(signal['target_1'], 4)
                        signal['target_2'] = round(signal['target_2'], 4)
                        signal['stop_loss'] = round(signal['stop_loss'], 4)
                    else:
                        signal['target_1'] = round(signal['target_1'], 2)
                        signal['target_2'] = round(signal['target_2'], 2)
                        signal['stop_loss'] = round(signal['stop_loss'], 2)
                    
                    logger.info(f"  📊 Пересчитаны цели для SHORT: t1={signal['target_1']:.6f}, t2={signal['target_2']:.6f}, sl={signal['stop_loss']:.6f}")
                
                logger.info(f"  📊 format_pump_message: СМЕНА! {old_dir} → {signal['direction']} (PUMP без пробоя)")
                if 'reasons' in signal:
                    has_pump_reason = any('Пробой уровня' in r or 'Коррекция' in r for r in signal['reasons'])
                    if not has_pump_reason:
                        signal['reasons'].insert(0, f"Коррекция после пампа +{pump_change:.1f}%")

        else:  # DUMP
            # Получаем bearish_score из сигнала (нужно передать из generate_signal)
            bearish_score = signal.get('bearish_score', 0)
            logger.info(f"  📊 format_pump_message: bearish_score = {bearish_score}")
            
            if has_breakout:
                old_dir = signal['direction']
                signal_emoji = "📉"
                signal_text = f"DUMP {pump_change:.1f}%"
                signal['direction'] = 'SHORT 📉 (пробой после дампа)'
                signal['signal_type'] = 'DUMP_BREAKOUT'
                
                logger.info(f"  🔍 bearish_score={bearish_score}, atr={signal.get('atr', 0)}")

                # ===== ПЕРЕСЧИТЫВАЕМ ЦЕЛИ ДЛЯ SHORT =====
                current_price = signal['price']
                atr = signal.get('atr', 0)
                
                if atr > 0:
                    from config import ATR_SETTINGS
                    signal['target_1'] = current_price - atr * ATR_SETTINGS['short_target_1_mult']
                    signal['target_2'] = current_price - atr * ATR_SETTINGS['short_target_2_mult']
                    signal['stop_loss'] = current_price + atr * ATR_SETTINGS['short_stop_loss_mult']
                    logger.info(f"  📊 Пересчитаны цели для SHORT: t1={signal['target_1']:.6f}, t2={signal['target_2']:.6f}, sl={signal['stop_loss']:.6f}")
                
                logger.info(f"  📊 format_pump_message: СМЕНА! {old_dir} → {signal['direction']} (DUMP + пробой)")
                
                if 'reasons' in signal and not any('Отскок' in r for r in signal['reasons']):
                    reason_text = f"Пробой уровня после дампа {pump_change:.1f}%"
                    if reason_text not in signal['reasons']:
                        signal['reasons'].insert(0, reason_text)
            else:
                # Без пробоя - проверяем силу медвежьих сигналов
                if bearish_score >= 50:
                    old_dir = signal['direction']
                    signal_emoji = "📉"
                    signal_text = f"DUMP {pump_change:.1f}%"
                    signal['direction'] = 'SHORT 📉 (продолжение)'
                    signal['signal_type'] = 'DUMP'
                    logger.info(f"  📊 format_pump_message: СМЕНА! {old_dir} → {signal['direction']} (DUMP, bearish_score={bearish_score})")
                    if 'reasons' in signal and not any('Отскок' in r for r in signal['reasons']):
                        reason_text = f"Продолжение дампа {pump_change:.1f}%"
                        if reason_text not in signal['reasons']:
                            signal['reasons'].insert(0, reason_text)
                else:
                    old_dir = signal['direction']
                    signal_emoji = "📊" if pump_change < -1.5 else "📉"
                    signal_text = f"DUMP {pump_change:.1f}%"
                    signal['direction'] = 'LONG 📈 (отскок)'
                    signal['signal_type'] = 'DUMP'
                    
                    # ===== ДОБАВИТЬ ПЕРЕСЧЕТ ЦЕЛЕЙ ДЛЯ LONG =====
                    current_price = signal['price']
                    atr = signal.get('atr', 0)
                    
                    if atr > 0:
                        from config import ATR_SETTINGS
                        signal['target_1'] = current_price + atr * ATR_SETTINGS['long_target_1_mult']
                        signal['target_2'] = current_price + atr * ATR_SETTINGS['long_target_2_mult']
                        signal['stop_loss'] = current_price - atr * ATR_SETTINGS['long_stop_loss_mult']
                        
                        # Округление (как в generate_signal)
                        if current_price < 0.0001:
                            signal['target_1'] = round(signal['target_1'], 8)
                            signal['target_2'] = round(signal['target_2'], 8)
                            signal['stop_loss'] = round(signal['stop_loss'], 8)
                        elif current_price < 0.001:
                            signal['target_1'] = round(signal['target_1'], 6)
                            signal['target_2'] = round(signal['target_2'], 6)
                            signal['stop_loss'] = round(signal['stop_loss'], 6)
                        elif current_price < 0.01:
                            signal['target_1'] = round(signal['target_1'], 5)
                            signal['target_2'] = round(signal['target_2'], 5)
                            signal['stop_loss'] = round(signal['stop_loss'], 5)
                        elif current_price < 0.1:
                            signal['target_1'] = round(signal['target_1'], 4)
                            signal['target_2'] = round(signal['target_2'], 4)
                            signal['stop_loss'] = round(signal['stop_loss'], 4)
                        else:
                            signal['target_1'] = round(signal['target_1'], 2)
                            signal['target_2'] = round(signal['target_2'], 2)
                            signal['stop_loss'] = round(signal['stop_loss'], 2)
                        
                        logger.info(f"  📊 Пересчитаны цели для LONG: t1={signal['target_1']:.6f}, t2={signal['target_2']:.6f}, sl={signal['stop_loss']:.6f}")
                    
                    logger.info(f"  📊 format_pump_message: СМЕНА! {old_dir} → {signal['direction']} (DUMP, bearish_score={bearish_score})")
                    if 'reasons' in signal and not any('Отскок' in r for r in signal['reasons']):
                        signal['reasons'].insert(0, f"Отскок после дампа {pump_change:.1f}%")
        
        logger.info(f"  📊 format_pump_message END: финальное направление = {signal['direction']}")
        
        # Определяем силу сигнала по модулю движения
        signal_power = self._get_power_text(abs(pump_change))
        signal['signal_power'] = signal_power
        
        # Проверяем, есть ли ожидание пробоя
        if any("ждем ПРОБОЯ" in r for r in signal.get('reasons', [])):
            signal_emoji = "⏳" + signal_emoji
            signal_text = f"{signal_text} (ждет пробоя)"

        line1 = f"{signal_emoji} <code>{coin}</code> {signal_text} {signal_power}"
        
        # Параметры контракта
        if contract_info:
            max_lev = contract_info.get('max_leverage')
            if max_lev is None or max_lev > 200:
                max_lev = 100
            
            # ✅ ДОБАВИТЬ: для накопления снижаем плечо
            if signal.get('signal_type') == 'accumulation' and ACCUMULATION_SIGNAL_SETTINGS.get('enabled', True):
                max_lev = ACCUMULATION_SIGNAL_SETTINGS.get('max_leverage', 20)
                if ACCUMULATION_SIGNAL_SETTINGS.get('show_leverage_warning', True):
                    line1 = f"{line1}\n⚠️ *СНИЖЕННОЕ ПЛЕЧО {max_lev}x* (накопление)"

            min_amt = contract_info.get('min_amount')
            if min_amt is None or min_amt > 1000:
                min_amt = 5.0
            
            max_amt = contract_info.get('max_amount')
            if max_amt is None or max_amt > 10_000_000:
                max_amt = 2_000_000
            
            line2 = f"📌 {max_lev}x / {min_amt:.0f}$ / {self._format_compact(max_amt)}"
            
            # Объем 24ч
            if signal.get('volume_24h') is not None and signal['volume_24h'] > 0:
                volume = signal['volume_24h']
                if volume > 1_000_000:
                    line2 += f" / {volume/1_000_000:.1f}M"
                elif volume > 1_000:
                    line2 += f" / {volume/1_000:.1f}K"
                else:
                    line2 += f" / {volume:.0f}"
            
            # Фандинг
            funding_rate = signal.get('funding_rate')
            if funding_rate is not None:
                funding = funding_rate * 100
                funding_emoji = "🟢" if funding > 0 else "🔴" if funding < 0 else "⚪"
                line2 += f" / {funding_emoji} {funding:.3f}%"
        else:
            line2 = f"📌 100x / 5$ / 2.0M"
            
            if signal.get('volume_24h') is not None and signal['volume_24h'] > 0:
                volume = signal['volume_24h']
                if volume > 1_000_000:
                    line2 += f" / {volume/1_000_000:.1f}M"
                elif volume > 1_000:
                    line2 += f" / {volume/1_000:.1f}K"
                else:
                    line2 += f" / {volume:.0f}"
            
            funding_rate = signal.get('funding_rate')
            if funding_rate is not None:
                funding = funding_rate * 100
                funding_emoji = "🟢" if funding > 0 else "🔴" if funding < 0 else "⚪"
                line2 += f" / {funding_emoji} {funding:.3f}%"
        
        # exchange_link = REF_LINKS.get(signal['exchange'], '#')
        # line3 = f"💲 Trade: <a href='{exchange_link}'>{signal['exchange']}</a>"
        # Строка 3: биржи (3 штуки)
        bingx_link = REF_LINKS.get('BingX', '#')
        bybit_link = REF_LINKS.get('Bybit', '#')
        mexc_link = REF_LINKS.get('MEXC', '#')
        line3 = f"💲 Trade: <a href='{bingx_link}'>BingX</a> | <a href='{bybit_link}'>Bybit</a> | <a href='{mexc_link}'>MEXC</a>"
        
        line4 = ""
        line5 = f"📊 Направление: {signal['direction']}"
        line6 = f"🕓 Таймфрейм: {TIMEFRAMES.get('current', '15m')}"
        
        # Форматирование цены
        if signal['price'] < 0.00001:
            price_formatted = f"{signal['price']:.8f}"
        elif signal['price'] < 0.0001:
            price_formatted = f"{signal['price']:.7f}"
        elif signal['price'] < 0.001:
            price_formatted = f"{signal['price']:.6f}"
        elif signal['price'] < 0.01:
            price_formatted = f"{signal['price']:.5f}"
        elif signal['price'] < 0.1:
            price_formatted = f"{signal['price']:.4f}"
        elif signal['price'] < 1:
            price_formatted = f"{signal['price']:.3f}"
        else:
            price_formatted = f"{signal['price']:.2f}"
        
        price_formatted = price_formatted.rstrip('0').rstrip('.') if '.' in price_formatted else price_formatted
        line7 = f"💰 Цена текущая: <code>{price_formatted}</code>"

        line8 = ""
        if pump_data:
            start_price = pump_data.get('start_price', signal['price'] / (1 + pump_change/100))
            if start_price < 0.001:
                start_formatted = f"{start_price:.8f}".rstrip('0').rstrip('.')
            else:
                start_formatted = f"{start_price:.4f}"
            
            # ✅ ПРАВИЛЬНО: проверяем направление движения
            if pump_change > 0:
                line8 = f"📈 Рост: <code>{start_formatted}</code> → <code>{price_formatted}</code> за {pump_time:.0f}с"
            else:
                line8 = f"📉 Падение: <code>{start_formatted}</code> → <code>{price_formatted}</code> за {pump_time:.0f}с"
        
        # Собираем строки сообщения
        lines = [line1, line2, line3, line4, line5, line6, line7]
        
        if line8:
            lines.append(line8)

        # Зоны доп.входа (добавить сюда)
        entry_zones = signal.get('entry_zones', [])
        logger.info(f"  🔍 Зоны доп.входа в памп-сигнале: {entry_zones}")
        if entry_zones:
            lines.append("🟣 Зоны доп.входа:")
            for zone in entry_zones:
                lines.append(f"     ▪️ <code>{zone}</code>")

        # Форматирование целей
        if signal.get('target_1') and signal.get('target_2') and signal.get('stop_loss'):
            # ✅ ЗАЩИТА ОТ НЕПРАВИЛЬНЫХ ЦЕЛЕЙ
            current_price = signal['price']
            direction = signal['direction']
            
            # Для SHORT цели должны быть ниже цены, стоп - выше
            if 'SHORT' in direction:
                if signal['target_1'] > current_price or signal['target_2'] > current_price:
                    logger.warning(f"  ⚠️ Неправильные цели для SHORT: цели выше цены. Меняем местами со стопом")
                    # Меняем местами цели и стоп
                    temp_t1 = signal['target_1']
                    temp_t2 = signal['target_2']
                    temp_sl = signal['stop_loss']
                    signal['target_1'] = temp_sl
                    signal['target_2'] = temp_sl - (temp_t1 - temp_sl)  # примерная вторая цель
                    signal['stop_loss'] = temp_t1
            
            # Для LONG цели должны быть выше цены, стоп - ниже
            if 'LONG' in direction:
                if signal['target_1'] < current_price or signal['target_2'] < current_price:
                    logger.warning(f"  ⚠️ Неправильные цели для LONG: цели ниже цены. Меняем местами со стопом")
                    temp_t1 = signal['target_1']
                    temp_t2 = signal['target_2']
                    temp_sl = signal['stop_loss']
                    signal['target_1'] = temp_sl
                    signal['target_2'] = temp_sl + (temp_sl - temp_t1)
                    signal['stop_loss'] = temp_t1
            def format_target(price):
                if price < 0.00001:
                    return f"{price:.8f}".rstrip('0').rstrip('.')
                elif price < 0.0001:
                    return f"{price:.7f}".rstrip('0').rstrip('.')
                elif price < 0.001:
                    return f"{price:.6f}".rstrip('0').rstrip('.')
                elif price < 0.01:
                    return f"{price:.5f}".rstrip('0').rstrip('.')
                elif price < 0.1:
                    return f"{price:.4f}".rstrip('0').rstrip('.')
                elif price < 1:
                    return f"{price:.3f}".rstrip('0').rstrip('.')
                else:
                    return f"{price:.2f}"
            
            t1 = format_target(signal['target_1'])
            t2 = format_target(signal['target_2'])
            sl = format_target(signal['stop_loss'])
            line9 = f"🎯 Цели: <code>{t1}</code> | <code>{t2}</code> | SL <code>{sl}</code>"
        else:
            line9 = "🎯 Цели: N/A | N/A | SL N/A"
        
        line10 = ""
        line11 = "💡 Причины:"
        
        # Очистка причин от эмодзи
        clean_reasons = []
        for reason in signal['reasons'][:15]:
            clean_reason = reason
            clean_reason = clean_reason.replace("📊 ", "")
            clean_reason = clean_reason.replace("✅ ", "")
            clean_reason = clean_reason.replace("🔄 ", "")
            clean_reason = clean_reason.replace("💰 ", "")
            clean_reason = clean_reason.replace("📈 ", "")
            clean_reason = clean_reason.replace("📉 ", "")
            clean_reason = clean_reason.replace("⚡️ ", "")
            clean_reason = clean_reason.replace("🔥 ", "")
            clean_reason = clean_reason.replace("🟢 ", "")
            clean_reason = clean_reason.replace("🔴 ", "")
            clean_reason = clean_reason.replace("⚪️ ", "")
            clean_reason = clean_reason.replace("⚪ ", "")
            clean_reason = clean_reason.replace("📦 ", "")
            clean_reason = clean_reason.replace("📐 ", "")
            clean_reason = clean_reason.replace("⚠️ ", "")
            clean_reason = clean_reason.strip()
            clean_reasons.append(clean_reason)        
               
        # ✅ Фильтрация причин (только важные)
        priority_keywords = [
            # SMC
            'EQH', 'EQL', 'Premium', 'Discount', 'FVG', 'Order Block', 'CHoCH', 'BOS',
            # Памп-дамп
            'PUMP', 'DUMP', 'пампа', 'дампа', 'Коррекция',
            # Технический анализ
            'RSI', 'MACD', 'EMA', 'VWAP', 'Согласованность', 'тренд',
            # Паттерны и уровни
            'Пробой', 'Отскок', 'Двойная', 'ФЛАГ', 'КЛИН', 'Голова',
            # Накопление
            'Накопление', 'аккумуляция', 'объем', 'сжатие',
            # Старшие ТФ
            'Старших ТФ', 'Конфлюенция', 'КОНФЛЮЕНЦИЯ'
        ]
        filtered_reasons = []
        for r in clean_reasons:
            if any(k in r for k in priority_keywords):
                filtered_reasons.append(r)
        
        # Если после фильтрации ничего не осталось — показываем первые 5
        if not filtered_reasons:
            filtered_reasons = clean_reasons[:5]
        
        reasons_lines = [f"     {r}" for r in filtered_reasons[:8]]

        # Собираем сообщение
        lines = [line1, line2, line3, line4, line5, line6, line7]
        if line8:
            lines.append(line8)
        lines.extend([line9, line10, line11])
        lines.extend(reasons_lines)
        
        message = "\n".join(lines)
        
        # Кнопки
        keyboard = []
        row1 = []
        if DISPLAY_SETTINGS['buttons']['copy']:
            row1.append(InlineKeyboardButton(f"📋 Копировать {coin}", callback_data=f"copy_{coin}"))
        if DISPLAY_SETTINGS['buttons']['trade']:
            row1.append(InlineKeyboardButton(f"🚀 Торговать на {signal['exchange']}", url=REF_LINKS.get(signal['exchange'], '#')))
        if row1:
            keyboard.append(row1)
        
        row2 = []
        if DISPLAY_SETTINGS['buttons']['refresh']:
            row2.append(InlineKeyboardButton("🔄 Обновить", callback_data=f"refresh_{coin}"))
        if DISPLAY_SETTINGS['buttons']['details']:
            row2.append(InlineKeyboardButton("📊 Детали", callback_data=f"details_{coin}"))
        if row2:
            keyboard.append(row2)
        
        return message, InlineKeyboardMarkup(keyboard) if keyboard else None

    def _get_power_text(self, strength: float) -> str:
        """Определение текста силы сигнала для ПАМП-ДВИЖЕНИЙ (0-100%+)"""
        if strength >= 20.0:
            return "🔥🔥🔥🔥 ЭКСТРЕМАЛЬНЫЙ"
        elif strength >= 12.0:
            return "🔥🔥🔥 ОЧЕНЬ СИЛЬНЫЙ"
        elif strength >= 8.0:
            return "🔥🔥 СИЛЬНЫЙ"
        elif strength >= 5.0:
            return "🔥 СРЕДНИЙ"
        elif strength >= 3.0:
            return "📊 СЛАБЫЙ"
        elif strength >= 1.5:
            return "⚡ ОЧЕНЬ СЛАБЫЙ"
        else:
            return "👀 НАБЛЮДЕНИЕ"

# ============== ОСНОВНОЙ КЛАСС БОТА ==============

class MultiExchangeScannerBot:
    def __init__(self):
        self.fetchers = {}
        self.analyzer = MultiTimeframeAnalyzer()
        self.chart_generator = ChartGenerator()
        self.telegram_bot = Bot(token=TELEGRAM_TOKEN)
        self.last_signals = {}
        self.accumulation_alerts_sent = set()
        self.breakout_tracker = BreakoutTracker()
        self.fakeout_detector = FakeoutDetector()
        
        self.divergence = DivergenceAnalyzer() if FEATURES['advanced']['divergence'] else None
        self.imbalance = ImbalanceAnalyzer(IMBALANCE_SETTINGS) if FEATURES['advanced']['imbalance'] else None
        self.liquidity = LiquidityAnalyzer(LIQUIDITY_SETTINGS) if FEATURES['advanced']['liquidity'] else None
        self.last_signal_time = {}  # {coin: datetime}
        self.last_signal_direction = {}  # {coin: direction}

        # Инициализация дополнительных анализаторов
        if FEATURES['advanced']['fibonacci']:
            from config import FIBONACCI_SETTINGS
            self.fibonacci = FibonacciAnalyzer(FIBONACCI_SETTINGS)
            self.analyzer.set_fibonacci(self.fibonacci)
            logger.info("✅ Анализатор Фибоначчи инициализирован")
        
        # Volume Profile (импорт вынесен ДО условия)
        from config import VOLUME_PROFILE_SETTINGS
        if FEATURES['advanced']['volume_profile'] and VOLUME_PROFILE_SETTINGS.get('enabled', False):
            self.volume_profile = VolumeProfileAnalyzer(VOLUME_PROFILE_SETTINGS)
            self.analyzer.set_volume_profile(self.volume_profile)
            logger.info("✅ Volume Profile анализатор инициализирован")
        
        # Инициализация анализатора накопления
        if FEATURES['advanced']['accumulation']:
            from config import ACCUMULATION_SETTINGS
            self.accumulation = AccumulationAnalyzer(ACCUMULATION_SETTINGS)
            self.analyzer.set_accumulation(self.accumulation)
            logger.info("✅ Анализатор накопления инициализирован")
        
        # Инициализация бирж
        from config import EXCHANGES
        
        for exchange_id, config in EXCHANGES.items():
            if config.get('enabled', False):
                if exchange_id == 'bingx':
                    self.fetchers['BingX'] = BingxFetcher()
                else:
                    self.fetchers[exchange_id.capitalize()] = MultiExchangeFetcher(
                        exchange_id,
                        config.get('api_key'),
                        config.get('api_secret')
                    )
        
        # Инициализация статистики
        if STATS_SETTINGS['enabled'] and STATS_SETTINGS['stats_chat_id']:
            self.stats = SignalStatistics(self.telegram_bot, STATS_SETTINGS['stats_chat_id'])
            logger.info("✅ Система статистики инициализирована")
            
            # Запускаем фоновые задачи
            asyncio.create_task(self.stats_updater_loop())
            asyncio.create_task(self.daily_report_loop())
    
    def extract_coin(self, symbol: str) -> str:
        if '/USDT' in symbol:
            return symbol.split('/')[0]
        return symbol.replace('USDT', '')
    
    def format_compact(self, num: float) -> str:
        if num is None:
            return "N/A"
        if num > 1_000_000_000:
            return f"{num/1_000_000_000:.1f}B"
        elif num > 1_000_000:
            return f"{num/1_000_000:.1f}M"
        elif num > 1_000:
            return f"{num/1_000:.1f}K"
        else:
            return f"{num:.0f}"
    
    # ============== НОВЫЕ ФУНКЦИИ ДЛЯ ПОНЯТНЫХ ОПИСАНИЙ ==============
    
    def get_volume_description(self, volume_ratio: float) -> str:
        """
        Преобразует числовое значение объема в понятное текстовое описание
        """
        if volume_ratio >= 5.0:
            return f"🔥 АНОМАЛЬНЫЙ объем (x{volume_ratio:.1f})"
        elif volume_ratio >= 3.0:
            return f"⚡ ОЧЕНЬ СИЛЬНЫЙ объем (x{volume_ratio:.1f})"
        elif volume_ratio >= 2.0:
            return f"✅ СИЛЬНЫЙ объем (x{volume_ratio:.1f})"
        elif volume_ratio >= 1.5:
            return f"📊 ПОВЫШЕННЫЙ объем (x{volume_ratio:.1f})"
        else:
            return f"📉 обычный объем (x{volume_ratio:.1f})"
    
    def get_vwap_description(self, price: float, vwap: float) -> str:
        """
        Преобразует положение цены относительно VWAP в понятное описание
        """
        if vwap is None or vwap == 0:
            return ""
        
        diff_percent = (price - vwap) / vwap * 100
        
        if price > vwap:
            if diff_percent > 5:
                return f"🔥 Цена значительно ВЫШЕ справедливой (VWAP +{diff_percent:.1f}%)"
            elif diff_percent > 2:
                return f"⚡ Цена ВЫШЕ справедливой (VWAP +{diff_percent:.1f}%)"
            else:
                return f"✅ Цена чуть ВЫШЕ справедливой (VWAP +{diff_percent:.1f}%)"
        else:
            if diff_percent < -5:
                return f"🔥 Цена значительно НИЖЕ справедливой (VWAP {diff_percent:.1f}%)"
            elif diff_percent < -2:
                return f"⚡ Цена НИЖЕ справедливой (VWAP {diff_percent:.1f}%)"
            else:
                return f"📉 Цена чуть НИЖЕ справедливой (VWAP {diff_percent:.1f}%)"
    
    def get_rsi_description(self, rsi: float) -> str:
        """
        Описание состояния RSI
        """
        if rsi >= 80:
            return f"🔥 RSI перекуплен ({rsi:.1f}) - сильный сигнал на продажу"
        elif rsi >= 70:
            return f"⚡ RSI перекуплен ({rsi:.1f}) - возможна коррекция"
        elif rsi <= 20:
            return f"🔥 RSI перепродан ({rsi:.1f}) - сильный сигнал на покупку"
        elif rsi <= 30:
            return f"⚡ RSI перепродан ({rsi:.1f}) - возможен отскок"
        elif 40 <= rsi <= 60:
            return f"📊 RSI нейтральный ({rsi:.1f})"
        else:
            return f"📉 RSI {rsi:.1f}"
    
    def get_funding_description(self, funding_rate: float) -> str:
        """
        Описание ставки фондирования
        """
        if funding_rate is None:
            return ""
        
        funding_pct = funding_rate * 100
        
        if funding_pct > 0.05:
            return f"🔥 Очень высокий позитивный фандинг ({funding_pct:.3f}%) - шортисты переплачивают"
        elif funding_pct > 0.01:
            return f"⚡ Высокий позитивный фандинг ({funding_pct:.3f}%) - рынок перегрет"
        elif funding_pct > 0.001:
            return f"✅ Позитивный фандинг ({funding_pct:.3f}%)"
        elif funding_pct < -0.05:
            return f"🔥 Очень высокий негативный фандинг ({funding_pct:.3f}%) - лонгисты переплачивают"
        elif funding_pct < -0.01:
            return f"⚡ Высокий негативный фандинг ({funding_pct:.3f}%) - рынок перегрет"
        elif funding_pct < -0.001:
            return f"📉 Негативный фандинг ({funding_pct:.3f}%)"
        else:
            return f"⚪ Фандинг нейтральный ({funding_pct:.3f}%)"
    
    # ============== ОСНОВНОЙ МЕТОД ФОРМАТИРОВАНИЯ ==============
    
    def format_message(self, signal: Dict, contract_info: Dict = None, pump_percent: float = None, df: pd.DataFrame = None) -> Tuple[str, InlineKeyboardMarkup]:
        """Форматирование сигнала с новым форматом"""
        
        # Определяем эмодзи
        if signal.get('signal_type') in ['PUMP', 'DUMP'] or pump_percent:
            main_emoji = '🚀' if signal.get('signal_type') == 'PUMP' else '📉'
            coin = self.extract_coin(signal['symbol'])
            if signal.get('pump_dump'):
                pump_text = f" {signal['pump_dump'][0]['change_percent']:+.1f}%"
            else:
                pump_text = f" {pump_percent:+.1f}%" if pump_percent else ""
        elif signal.get('signal_type') == 'accumulation':
            main_emoji = '📦'
            coin = self.extract_coin(signal['symbol'])
            pump_text = " НАКОПЛЕНИЕ"
        else:
            if 'LONG' in signal['direction']:
                main_emoji = '🟢'
            elif 'SHORT' in signal['direction']:
                main_emoji = '🔴'
            else:
                main_emoji = '⚪'
            coin = self.extract_coin(signal['symbol'])
            pump_text = ""
        
        # Определяем направление для отображения (с учетом предупреждений)
        display_direction = signal['direction']
        
        # Если есть предупреждение о противоречии, добавляем в направление
        if '⚠️' in str(signal.get('reasons', [])):
            if 'LONG' in display_direction and 'коррекция' in str(signal.get('reasons', [])):
                display_direction = "LONG (возможна коррекция)"
            elif 'SHORT' in display_direction and 'коррекция' in str(signal.get('reasons', [])):
                display_direction = "SHORT (возможна коррекция)"

        # Строка 1: название и сила
        line1 = f"{main_emoji} <code>{coin}</code>{pump_text} {signal['signal_power']}"
        
        # Строка 2: параметры контракта
        if contract_info:
            max_lev = contract_info.get('max_leverage', 100)
            if max_lev > 200:
                max_lev = 100
            min_amt = contract_info.get('min_amount', 5.0)
            if min_amt > 1000:
                min_amt = 5.0
            max_amt = contract_info.get('max_amount', 2_000_000)
            if max_amt > 10_000_000:
                max_amt = 2_000_000
            
            line2 = f"📌 {max_lev}x / {min_amt:.0f}$ / {self.format_compact(max_amt)}"
            
            if signal.get('volume_24h') and signal['volume_24h'] > 0:
                volume = signal['volume_24h']
                if volume > 1_000_000:
                    line2 += f" / {volume/1_000_000:.1f}M"
                elif volume > 1_000:
                    line2 += f" / {volume/1_000:.1f}K"
                else:
                    line2 += f" / {volume:.0f}"
            
            funding_rate = signal.get('funding_rate')
            if funding_rate is not None:
                funding = funding_rate * 100
                funding_emoji = "🟢" if funding > 0 else "🔴" if funding < 0 else "⚪"
                line2 += f" / {funding_emoji} {funding:.3f}%"
        else:
            line2 = f"📌 100x / 5$ / 2.0M"
            if signal.get('volume_24h') and signal['volume_24h'] > 0:
                volume = signal['volume_24h']
                if volume > 1_000_000:
                    line2 += f" / {volume/1_000_000:.1f}M"
                elif volume > 1_000:
                    line2 += f" / {volume/1_000:.1f}K"
                else:
                    line2 += f" / {volume:.0f}"
            
            funding_rate = signal.get('funding_rate')
            if funding_rate is not None:
                funding = funding_rate * 100
                funding_emoji = "🟢" if funding > 0 else "🔴" if funding < 0 else "⚪"
                line2 += f" / {funding_emoji} {funding:.3f}%"
        
        # Строка 3: биржи (3 штуки)
        bingx_link = REF_LINKS.get('BingX', '#')
        bybit_link = REF_LINKS.get('Bybit', '#')
        mexc_link = REF_LINKS.get('MEXC', '#')
        line3 = f"💲 Trade: <a href='{bingx_link}'>BingX</a> | <a href='{bybit_link}'>Bybit</a> | <a href='{mexc_link}'>MEXC</a>"
        
        # Пустая строка после бирж
        lines = [line1, line2, line3, ""]
        
        # Направление
        lines.append(f"📊 Направление: {display_direction}")
        
        # Таймфрейм (переводим)
        current_tf = TIMEFRAMES.get('current', '15m')
        tf_display = '15м'
        lines.append(f"🕓 Таймфрейм: {tf_display}")
        
        # Цена
        price = signal['price']
        if price < 0.00001:
            price_formatted = f"{price:.8f}".rstrip('0').rstrip('.')
        elif price < 0.0001:
            price_formatted = f"{price:.7f}".rstrip('0').rstrip('.')
        elif price < 0.001:
            price_formatted = f"{price:.6f}".rstrip('0').rstrip('.')
        elif price < 0.01:
            price_formatted = f"{price:.5f}".rstrip('0').rstrip('.')
        elif price < 0.1:
            price_formatted = f"{price:.4f}".rstrip('0').rstrip('.')
        elif price < 1:
            price_formatted = f"{price:.3f}".rstrip('0').rstrip('.')
        else:
            price_formatted = f"{price:.2f}".rstrip('0').rstrip('.')
        
        lines.append(f"💰 Цена текущая: {price_formatted}")
        
        # Зоны доп.входа
        entry_zones = signal.get('entry_zones', [])
        if entry_zones:
            lines.append("🟣 Зоны доп.входа:")
            for zone in entry_zones:
                lines.append(f"     ▪️ <code>{zone}</code>")
        
        # Потенциал для накопления
        potential_line = ""
        if signal.get('signal_type') == 'accumulation' and signal.get('accumulation', {}).get('potential'):
            potential = signal['accumulation']['potential']
            if potential.get('has_potential'):
                direction_emoji = "📈" if potential['target_pct'] > 0 else "📉"
                if potential['target_price'] < 0.001:
                    target_str = f"{potential['target_price']:.6f}".rstrip('0').rstrip('.')
                else:
                    target_str = f"{potential['target_price']:.4f}".rstrip('0').rstrip('.')
                potential_line = f"{direction_emoji} Потенциал: {potential['target_pct']:+.2f}% до {target_str} ({potential['target_level']})"
                lines.append(potential_line)
        
        # Рост для пампов
        pump_line = ""
        if pump_percent and signal.get('pump_dump') and len(signal['pump_dump']) > 0:
            pump_data = signal['pump_dump'][0]
            start_price = pump_data.get('start_price', signal['price'] / (1 + pump_percent/100))
            if start_price < 0.001:
                start_formatted = f"{start_price:.8f}".rstrip('0').rstrip('.')
            else:
                start_formatted = f"{start_price:.4f}"
            pump_line = f"📈 Рост: {start_formatted} → {price_formatted} за {pump_data.get('time_window', 0):.0f}с"
            lines.append(pump_line)
        
       # Цели
        if signal.get('target_1') and signal.get('target_2') and signal.get('stop_loss'):
            # ✅ ЗАЩИТА ОТ НЕПРАВИЛЬНЫХ ЦЕЛЕЙ
            current_price = signal['price']
            direction = signal['direction']
            
            # Для SHORT цели должны быть ниже цены, стоп - выше
            if 'SHORT' in direction:
                if signal['target_1'] > current_price or signal['target_2'] > current_price:
                    logger.warning(f"  ⚠️ Неправильные цели для SHORT: цели выше цены. Меняем местами со стопом")
                    # Меняем местами цели и стоп
                    temp_t1 = signal['target_1']
                    temp_t2 = signal['target_2']
                    temp_sl = signal['stop_loss']
                    signal['target_1'] = temp_sl
                    signal['target_2'] = temp_sl - (temp_t1 - temp_sl)  # примерная вторая цель
                    signal['stop_loss'] = temp_t1
            
            # Для LONG цели должны быть выше цены, стоп - ниже
            if 'LONG' in direction:
                if signal['target_1'] < current_price or signal['target_2'] < current_price:
                    logger.warning(f"  ⚠️ Неправильные цели для LONG: цели ниже цены. Меняем местами со стопом")
                    temp_t1 = signal['target_1']
                    temp_t2 = signal['target_2']
                    temp_sl = signal['stop_loss']
                    signal['target_1'] = temp_sl
                    signal['target_2'] = temp_sl + (temp_sl - temp_t1)
                    signal['stop_loss'] = temp_t1
            def format_target(p):
                if p < 0.00001:
                    return f"{p:.8f}".rstrip('0').rstrip('.')
                elif p < 0.0001:
                    return f"{p:.7f}".rstrip('0').rstrip('.')
                elif p < 0.001:
                    return f"{p:.6f}".rstrip('0').rstrip('.')
                elif p < 0.01:
                    return f"{p:.5f}".rstrip('0').rstrip('.')
                elif p < 0.1:
                    return f"{p:.4f}".rstrip('0').rstrip('.')
                elif p < 1:
                    return f"{p:.3f}".rstrip('0').rstrip('.')
                else:
                    return f"{p:.2f}".rstrip('0').rstrip('.')
            
            t1 = format_target(signal['target_1'])
            t2 = format_target(signal['target_2'])
            sl = format_target(signal['stop_loss'])
            lines.append(f"🎯 Цели: {t1} | {t2} | SL {sl}")
        
        # Пустая строка перед причинами
        lines.append("")
        
        # Причины
        lines.append("💡 Причины:")
        
        # Добавляем процент согласованности в начало причин
        if signal.get('tf_alignment_percentage'):
            alignment_text = f"📊 Согласованность ТФ: {signal['tf_alignment_percentage']}% ({signal.get('tf_aligned_count', 0)}/{signal.get('tf_total_count', 6)})"
            lines.append(f"     {alignment_text}")

        # Очистка и группировка причин
        clean_reasons = []
        for reason in signal.get('reasons', [])[:10]:
            clean_reason = reason
            # Убираем эмодзи
            for emoji in ["📊 ", "✅ ", "🔄 ", "💰 ", "📈 ", "📉 ", "⚡️ ", "🔥 ", "🟢 ", "🔴 ", "⚪️ ", "⚪ ", "📦 ", "📐 ", "⚠️ ", "🎯 "]:
                clean_reason = clean_reason.replace(emoji, "")
            # Заменяем VWAP на понятное
            clean_reason = clean_reason.replace("Цена выше VWAP", "Цена выше справедливой стоимости (VWAP)")
            clean_reason = clean_reason.replace("Цена ниже VWAP", "Цена ниже справедливой стоимости (VWAP)")
            clean_reasons.append(f"     {clean_reason}")
        
        lines.extend(clean_reasons)
        
        message = "\n".join(lines)
        
        # Кнопки
        keyboard = []
        row1 = []
        if DISPLAY_SETTINGS['buttons']['copy']:
            row1.append(InlineKeyboardButton(f"📋 Копировать {coin}", callback_data=f"copy_{coin}"))
        if DISPLAY_SETTINGS['buttons']['trade']:
            row1.append(InlineKeyboardButton(f"🚀 Торговать на BingX", url=REF_LINKS.get('BingX', '#')))
        if row1:
            keyboard.append(row1)
        
        row2 = []
        if DISPLAY_SETTINGS['buttons']['refresh']:
            row2.append(InlineKeyboardButton("🔄 Обновить", callback_data=f"refresh_{coin}"))
        if DISPLAY_SETTINGS['buttons']['details']:
            row2.append(InlineKeyboardButton("📊 Детали", callback_data=f"details_{coin}"))
        if row2:
            keyboard.append(row2)
        
        return message, InlineKeyboardMarkup(keyboard) if keyboard else None
            
    # ... остальные методы (scan_exchange, scan_all, fast_pump_scan, send_signal и т.д.) остаются без изменений ...
    
    async def scan_exchange(self, name: str, fetcher: BaseExchangeFetcher) -> List[Dict]:
        logger.info(f"🔍 Сканирую {name}...")
        signals = []
        
        try:
            pairs = await fetcher.fetch_all_pairs()
            if not pairs:
                logger.warning(f"⚠️ {name}: нет пар для анализа")
                return []
            
            scan_count = min(PAIRS_TO_SCAN, len(pairs))
            logger.info(f"📊 {name}: анализирую {scan_count} пар из {len(pairs)}")
            
            for i, pair in enumerate(pairs[:PAIRS_TO_SCAN]):
                try:
                    logger.info(f"🔄 [{i+1}/{scan_count}] Анализирую {pair}")
                    
                    dataframes = {}
                    for tf_name, tf_value in TIMEFRAMES.items():
                        limit = 100 if tf_name == 'current' else 50
                        df = await fetcher.fetch_ohlcv(pair, tf_value, limit)
                        if df is not None and not df.empty:
                            df = self.analyzer.calculate_indicators(df)
                            dataframes[tf_name] = df
                            logger.info(f"  ✅ Загружены данные для {tf_name}: {len(df)} свечей")
                        else:
                            logger.warning(f"  ⚠️ Нет данных для {tf_name}")
                    
                    if not dataframes:
                        logger.warning(f"  ⚠️ Нет данных для {pair}, пропускаю")
                        continue
                    
                    funding = await fetcher.fetch_funding_rate(pair)
                    ticker = await fetcher.fetch_ticker(pair)
                    
                    metadata = {
                        'funding_rate': funding,
                        'volume_24h': ticker.get('volume_24h'),
                        'price_change_24h': ticker.get('percentage')
                    }
                    
                    logger.info(f"  📊 Метаданные: funding={funding}, volume={ticker.get('volume_24h')}")
                    
                    try:
                        signal = self.analyzer.generate_signal(dataframes, metadata, pair, name)
                    except Exception as e:
                        logger.error(f"❌ Исключение в generate_signal для {pair}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                    
                    if signal is None:
                        logger.info(f"  ❌ generate_signal вернул None для {pair}")
                        continue
                    
                    logger.info(f"  ✅ Сгенерирован сигнал: {signal['direction']} (уверенность {signal['confidence']}%)")
                    
                    if 'NEUTRAL' in signal['direction']:
                        logger.info(f"  ⏭️ NEUTRAL сигнал пропущен")
                        continue
                    
                    # ✅ Нормализуем уверенность (максимум 100)
                    if signal['confidence'] > 100:
                        signal['confidence'] = min(100, signal['confidence'])
                        logger.info(f"  📊 Нормализована уверенность: {signal['confidence']:.1f}%")

                    if signal['confidence'] < MIN_CONFIDENCE:
                        logger.info(f"  ⏭️ Низкая уверенность: {signal['confidence']}% < {MIN_CONFIDENCE}%")
                        continue
                    
                    signals.append(signal)
                    logger.info(f"  ✅ ДОБАВЛЕН сигнал: {pair} - {signal['direction']} ({signal['confidence']}%)")
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"📊 Прогресс {name}: {i + 1}/{scan_count}")
                    
                    await asyncio.sleep(0.2)
                    
                except Exception as e:
                    logger.error(f"❌ Ошибка анализа {pair}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Ошибка сканирования {name}: {e}")

        # ✅ ВСТАВИТЬ ЗДЕСЬ, ПОСЛЕ ЦИКЛА, ПЕРЕД return
        accumulation_count = sum(1 for s in signals if s.get('signal_type') == 'accumulation')
        logger.info(f"🎯 {name}: найдено {len(signals)} сигналов (накопление: {accumulation_count})")        
        return signals
    
    async def scan_all(self) -> List[Dict]:
        logger.info("="*50)
        logger.info("🚀 НАЧАЛО ОСНОВНОГО СКАНИРОВАНИЯ")
        logger.info("="*50)
        
        all_signals = []
        for name, fetcher in self.fetchers.items():
            signals = await self.scan_exchange(name, fetcher)
            all_signals.extend(signals)
        
        all_signals.sort(key=lambda x: x['signal_strength'], reverse=True)
        logger.info(f"🎯 ВСЕГО СИГНАЛОВ: {len(all_signals)}")
        
        return all_signals[:15]
    
    async def fast_pump_scan(self) -> List[Dict]:
        if not FEATURES['advanced']['pump_dump']:
            return []
        
        pump_signals = []
        for name, fetcher in self.fetchers.items():
            scanner = FastPumpScanner(
                fetcher, 
                PUMP_SCAN_SETTINGS, 
                self.analyzer,
                self.telegram_bot,
                self.chart_generator
            )
            signals = await scanner.scan_all_pairs()
            
            for signal in signals:
                contract_info = await fetcher.fetch_contract_info(signal['symbol'])
                if 'funding_rate' not in signal:
                    signal['funding_rate'] = await fetcher.fetch_funding_rate(signal['symbol'])
                msg, keyboard = scanner.format_pump_message(signal, contract_info)
                pump_signals.append({
                    'signal': signal,
                    'message': msg,
                    'keyboard': keyboard
                })
        
        pump_signals.sort(key=lambda x: abs(x['signal']['pump_dump'][0]['change_percent']), reverse=True)
        return pump_signals
    
    async def send_signal(self, signal: Dict, pump_only: bool = False):
        if pump_only and not signal.get('pump_dump'):
            return
        
        if signal['confidence'] < MIN_CONFIDENCE:
            return
        
        coin = self.extract_coin(signal['symbol'])
        current_time = datetime.now()
        
        # ЗАЩИТА ОТ ДУБЛИРОВАНИЯ
        # ✅ Проверяем, не было ли сигнала по этой монете за последние 5 минут
        if hasattr(self, 'last_signal_time'):
            if coin in self.last_signal_time:
                time_diff = (current_time - self.last_signal_time[coin]).total_seconds() / 60
                if time_diff < 20:  # 5 минут кд
                    last_dir = self.last_signal_direction.get(coin)
                    if last_dir == signal['direction']:
                        logger.info(f"⏭️ Пропускаю повторный сигнал {coin} ({signal['direction']}) через {time_diff:.1f} мин")
                        return
        else:
            # Инициализируем словари, если еще не созданы
            self.last_signal_time = {}
            self.last_signal_direction = {}
        
        self.last_signals[coin] = {
            'symbol': signal['symbol'],
            'signal': signal,
            'time': current_time
        }
        
        contract_info = None
        df = None
        for fetcher in self.fetchers.values():
            if fetcher.name == signal['exchange']:
                contract_info = await fetcher.fetch_contract_info(signal['symbol'])
                df = await fetcher.fetch_ohlcv(signal['symbol'], TIMEFRAMES.get('current', '15m'), limit=200)
                break
        
        pump_percent = None
        if signal.get('pump_dump') and len(signal['pump_dump']) > 0:
            pump_percent = signal['pump_dump'][0].get('change_percent')
        
        msg, keyboard = self.format_message(signal, contract_info, pump_percent)
        
        if signal.get('signal_type') == 'accumulation':
            chat_id = ACCUMULATION_CHAT_ID
            signal_type = 'accumulation'
        elif signal.get('pump_dump'):
            chat_id = PUMP_CHAT_ID
            signal_type = 'pump'
        else:
            chat_id = TELEGRAM_CHAT_ID
            signal_type = 'regular'
        
        try:
            if df is not None and not df.empty:
                df = self.analyzer.calculate_indicators(df)
                chart_buf = self.chart_generator.create_chart(df, signal, coin, TIMEFRAMES.get('current', '15m'))
                
                await self.telegram_bot.send_photo(
                    chat_id=chat_id,
                    photo=chart_buf,
                    caption=msg,
                    parse_mode='HTML',
                    reply_markup=keyboard
                )
                logger.info(f"✅ Отправлен {signal_type} сигнал с графиком: {signal['symbol']}")
            else:
                await self.telegram_bot.send_message(
                    chat_id=chat_id,
                    text=msg,
                    parse_mode='HTML',
                    reply_markup=keyboard
                )
                logger.info(f"✅ Отправлен {signal_type} сигнал: {signal['symbol']}")
            
            # ✅ Сохраняем время и направление после успешной отправки
            self.last_signal_time[coin] = current_time
            self.last_signal_direction[coin] = signal['direction']
            
            if hasattr(self, 'stats'):
                self.stats.add_signal(signal, signal_type)
                
        except Exception as e:
            logger.error(f"❌ Ошибка отправки: {e}")
    
    async def send_pump_signal(self, pump_data: Dict):
        signal = pump_data['signal']

        # ✅ ЛОГИРОВАНИЕ
        logger.info(f"  📊 Причины в памп-сигнале перед отправкой: {signal.get('reasons', [])[:10]}")

        coin = self.extract_coin(signal['symbol'])
        current_time = datetime.now()
        
        # ✅ ЗАЩИТА ОТ ДУБЛИРОВАНИЯ
        if hasattr(self, 'last_signal_time'):
            if coin in self.last_signal_time:
                time_diff = (current_time - self.last_signal_time[coin]).total_seconds() / 60
                last_dir = self.last_signal_direction.get(coin)
                logger.info(f"  🔍 Последний сигнал {coin}: {time_diff:.1f} мин назад, направление={last_dir}")
                if time_diff < 30 and last_dir == signal['direction']:
                    logger.info(f"⏭️ Пропускаю повторный памп-сигнал {coin} ({signal['direction']}) через {time_diff:.1f} мин")
                    return
            else:
                logger.info(f"  🔍 Первый сигнал для {coin}")
        else:
            self.last_signal_time = {}
            self.last_signal_direction = {}
        
        self.last_signals[coin] = {
            'symbol': signal['symbol'],
            'signal': signal,
            'time': current_time
        }
        
        df = None
        for fetcher in self.fetchers.values():
            if fetcher.name == signal['exchange']:
                df = await fetcher.fetch_ohlcv(signal['symbol'], TIMEFRAMES.get('current', '15m'), limit=200)
                break
        
        try:
            if df is not None and not df.empty:
                df = self.analyzer.calculate_indicators(df)
                chart_buf = self.chart_generator.create_chart(df, signal, coin, TIMEFRAMES.get('current', '15m'))
                
                await self.telegram_bot.send_photo(
                    chat_id=PUMP_CHAT_ID,
                    photo=chart_buf,
                    caption=pump_data['message'],
                    parse_mode='HTML',
                    reply_markup=pump_data['keyboard']
                )
                logger.info(f"✅ Отправлен памп-сигнал с графиком: {signal['symbol']}")
            else:
                await self.telegram_bot.send_message(
                    chat_id=PUMP_CHAT_ID,
                    text=pump_data['message'],
                    parse_mode='HTML',
                    reply_markup=pump_data['keyboard']
                )
                logger.info(f"✅ Отправлен памп-сигнал: {signal['symbol']}")
            
            # ✅ Сохраняем время и направление после успешной отправки
            self.last_signal_time[coin] = current_time
            self.last_signal_direction[coin] = signal['direction']
            
            if hasattr(self, 'stats'):
                self.stats.add_signal(signal, 'pump')
                
        except Exception as e:
            logger.error(f"❌ Ошибка отправки пампа: {e}")
    
    async def _send_accumulation_message(self, signal: Dict, coin: str):
        """Отправка сообщения о накоплении"""
        contract_info = None
        df = None
        for fetcher in self.fetchers.values():
            if fetcher.name == signal['exchange']:
                contract_info = await fetcher.fetch_contract_info(signal['symbol'])
                df = await fetcher.fetch_ohlcv(signal['symbol'], TIMEFRAMES.get('current', '15m'), limit=200)
                break
        
        msg, keyboard = self.format_message(signal, contract_info)
        
        try:
            if df is not None and not df.empty:
                df = self.analyzer.calculate_indicators(df)
                chart_buf = self.chart_generator.create_chart(df, signal, coin, TIMEFRAMES.get('current', '15m'))
                await self.telegram_bot.send_photo(
                    chat_id=ACCUMULATION_CHAT_ID,
                    photo=chart_buf,
                    caption=msg,
                    parse_mode='HTML',
                    reply_markup=keyboard
                )
            else:
                await self.telegram_bot.send_message(
                    chat_id=ACCUMULATION_CHAT_ID,
                    text=msg,
                    parse_mode='HTML',
                    reply_markup=keyboard
                )
            logger.info(f"✅ Отправлен сигнал накопления: {signal['symbol']}")
            
            if hasattr(self, 'stats'):
                self.stats.add_signal(signal, 'accumulation')
                
        except Exception as e:
            logger.error(f"❌ Ошибка отправки сигнала накопления: {e}")

    async def send_accumulation_signal(self, signal: Dict):
        coin = self.extract_coin(signal['symbol'])
        
        self.last_signals[coin] = {
            'symbol': signal['symbol'],
            'signal': signal,
            'time': datetime.now()
        }

        # Сигнал 1: "В накоплении" (один раз)
        if ACCUMULATION_SETTINGS.get('send_accumulation_alert_once', True):
            if coin not in self.accumulation_alerts_sent:
                self.accumulation_alerts_sent.add(coin)
                signal['signal_type'] = 'accumulation'
                signal['direction'] = f"{signal['direction']} (В НАКОПЛЕНИИ)"
                await self._send_accumulation_message(signal, coin)
                return
        
        # Сигнал 2: "Поджатие к уровню"
        if signal.get('signal_type') == 'compression':
            signal['direction'] = f"{signal['direction']} (ПОДЖАТИЕ К УРОВНЮ)"
            await self._send_accumulation_message(signal, coin)
            return
        
        # Сигнал 3: "Выход из накопления"
        if ACCUMULATION_SETTINGS.get('require_breakout_for_entry', True):
            if signal.get('breakout_confirmed'):
                signal['direction'] = f"{signal['direction']} (ВЫХОД ИЗ НАКОПЛЕНИЯ)"
                await self._send_accumulation_message(signal, coin)
                return
        
        # Если ни одно условие не сработало — не отправляем сигнал
        logger.info(f"⏭️ {coin} - Накопление не подтверждено, сигнал не отправлен")
    
    async def get_detailed_analysis(self, fetcher, symbol: str, coin: str, signal_time: str = None) -> Tuple[str, InlineKeyboardMarkup]:
        try:
            lines = []
            lines.append(f"📊 *ДЕТАЛЬНЫЙ АНАЛИЗ {coin}*")
            if signal_time:
                lines.append(f"⏱️ Время сигнала: `{signal_time}`")
            lines.append(f"⏱️ Текущее время: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n")
            
            contract_info = await fetcher.fetch_contract_info(symbol)
            lines.append("⚡️ *ПАРАМЕТРЫ КОНТРАКТА:*")
            lines.append(f"└ Макс. плечо: `{contract_info.get('max_leverage', 100)}x`")
            lines.append(f"└ Мин. вход: `{contract_info.get('min_amount', 5):.2f} USDT`")
            lines.append(f"└ Макс. вход: `{self.format_compact(contract_info.get('max_amount', 2_000_000))} USDT`")
            
            if coin in self.last_signals:
                signal = self.last_signals[coin]['signal']
                lines.append("\n📊 *ТЕХНИЧЕСКИЙ АНАЛИЗ:*")
                for reason in signal['reasons']:
                    clean_reason = reason.replace("📊 ", "").replace("✅ ", "").replace("🔄 ", "")
                    lines.append(f"└ {clean_reason}")
                
                if 'fibonacci' in signal:
                    lines.append("\n📐 *ФИБОНАЧЧИ:*")
                    for tf, levels in signal['fibonacci']['levels'].items():
                        lines.append(f"└ {tf.upper()}: {len(levels)} уровней")
                
                if 'volume_profile' in signal:
                    lines.append("\n📊 *VOLUME PROFILE:*")
                    for tf, vp in signal['volume_profile']['levels'].items():
                        lines.append(f"└ {tf.upper()}: POC={vp['poc']:.2f}")
                
                if 'accumulation' in signal:
                    lines.append("\n📦 *НАКОПЛЕНИЕ:*")
                    acc = signal['accumulation']
                    for sig in acc.get('signals', [])[:3]:
                        lines.append(f"└ {sig}")
                    if acc.get('potential', {}).get('has_potential'):
                        pot = acc['potential']
                        lines.append(f"└ Потенциал: {pot['target_pct']:+.2f}% до {pot['target_level']}")
            
            detailed = "\n".join(lines)
            
            keyboard = InlineKeyboardMarkup([[
                InlineKeyboardButton("🔝 Вернуться к сигналу", callback_data=f"back_{coin}")
            ]])
            
            return detailed, keyboard
            
        except Exception as e:
            logger.error(f"Ошибка детального анализа {symbol}: {e}")
            return f"❌ Ошибка анализа: {e}", None
    
    async def stats_updater_loop(self):
        while True:
            await asyncio.sleep(STATS_SETTINGS['update_interval'])
            
            if not hasattr(self, 'stats'):
                continue
            
            # ✅ Создаём копию списка ключей (чтобы избежать ошибки)
            for signal_id in list(self.stats.db['signals'].keys()):
                signal_data = self.stats.db['signals'].get(signal_id)
                if not signal_data or signal_data['status'] != 'pending':
                    continue
                
                for fetcher in self.fetchers.values():
                    ticker = await fetcher.fetch_ticker(signal_data['symbol'])
                    if ticker and ticker.get('last'):
                        self.stats.update_signal(signal_id, ticker['last'])
                        break
    
    async def daily_report_loop(self):
        while True:
            now = datetime.now()
            target_time = datetime.strptime(STATS_SETTINGS['daily_report_time'], '%H:%M').time()
            target = datetime.combine(now.date(), target_time)
            
            if now > target:
                target += timedelta(days=1)
            
            wait_seconds = (target - now).total_seconds()
            await asyncio.sleep(wait_seconds)
            
            if hasattr(self, 'stats'):
                await self.stats.send_daily_report()
    
    async def run(self):
        logger.info("🤖 Мульти-биржевой бот запущен")
        logger.info(f"📊 Основной анализ: каждые {UPDATE_INTERVAL//60} мин")
        logger.info(f"🚀 Памп-сканер: каждые {PUMP_SCAN_INTERVAL} сек")
        
        last_full_scan = 0
        
        try:
            while True:
                current_time = time.time()
                
                pump_signals = await self.fast_pump_scan()
                if pump_signals:
                    for pump in pump_signals:
                        await self.send_pump_signal(pump)
                        await asyncio.sleep(3)
                
                if current_time - last_full_scan >= UPDATE_INTERVAL:
                    signals = await self.scan_all()
                    if signals:
                        for signal in signals:
                            if signal.get('signal_type') == 'accumulation':
                                await self.send_accumulation_signal(signal)
                            else:
                                await self.send_signal(signal)
                            await asyncio.sleep(3)
                    last_full_scan = current_time
                
                await asyncio.sleep(PUMP_SCAN_INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("🛑 Бот остановлен")
        finally:
            for fetcher in self.fetchers.values():
                await fetcher.close()

# ============== TELEGRAM HANDLER ==============

class TelegramHandler:
    def __init__(self, bot: MultiExchangeScannerBot):
        self.bot = bot
        self.app = Application.builder().token(TELEGRAM_TOKEN).build()
        self.register()        
        self.breakout_tracker = BreakoutTracker() # Трекер пробоев
    def register(self):
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("scan", self.scan))
        self.app.add_handler(CommandHandler("status", self.status))
        self.app.add_handler(CommandHandler("help", self.help))
        self.app.add_handler(CommandHandler("stats", self.stats_command))
        self.app.add_handler(CommandHandler("groups", self.groups_command))
        self.app.add_handler(CallbackQueryHandler(self.button))
        self.app.add_handler(CallbackQueryHandler(self.stats_button_handler, pattern="^stats_"))
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🤖 *Мульти-биржевой сканер*\n\n"
            "📊 *Доступные группы:*\n"
            "• Основная группа - обычные сигналы (LONG/SHORT)\n"
            "• Памп-группа - PUMP/DUMP сигналы\n"
            "• Накопление - ранние сигналы до импульса\n"
            "• Статистика - отчеты и метрики\n\n"
            "📋 *Команды:*\n"
            "/scan - Ручное сканирование\n"
            "/status - Статус бота\n"
            "/stats - Статистика сигналов\n"
            "/groups - Информация о группах\n"
            "/help - Помощь",
            parse_mode='Markdown'
        )
    
    async def scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = await update.message.reply_text("🔍 Сканирую...")
        signals = await self.bot.scan_all()
        if signals:
            await msg.edit_text(f"✅ Найдено {len(signals)} сигналов")
            for signal in signals:
                if signal.get('signal_type') == 'accumulation':
                    await self.bot.send_accumulation_signal(signal)
                else:
                    await self.bot.send_signal(signal)
                await asyncio.sleep(3)
        else:
            await msg.edit_text("❌ Сигналов не найдено")
    
    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = "*📡 Статус:*\n\n"
        text += f"✅ BingX Futures: активен\n"
        text += f"📊 Групп: 4 (осн., памп, накопление, статистика)\n"
        text += f"📈 Последних сигналов: {len(self.bot.last_signals)}"
        await update.message.reply_text(text, parse_mode='Markdown')
    
    async def groups_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = "*📊 ГРУППЫ СИГНАЛОВ*\n\n"
        text += "🔹 *Основная группа* - обычные LONG/SHORT сигналы\n"
        text += "   Технический анализ, тренды, уровни\n\n"
        text += "🔹 *Памп-группа* - PUMP/DUMP сигналы\n"
        text += "   Движения >3%, импульсы и развороты\n\n"
        text += "🔹 *Накопление* - ранние сигналы\n"
        text += "   Дивергенции, аномальный объем, накопление\n\n"
        text += "🔹 *Статистика* - отчеты и метрики\n"
        text += "   Ежедневные отчеты, статистика по команде /stats"
        
        await update.message.reply_text(text, parse_mode='Markdown')
    
    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "*Помощь*\n\n"
            "📊 *Анализ:* RSI, MACD, EMA, VWAP\n"
            "🔥 *Дополнительно:* Дивергенции, имбалансы, фракталы\n"
            "📐 *Фибоначчи:* Коррекции и расширения\n"
            "📦 *Накопление:* Ранние сигналы до импульса\n"
            "🚀 *Памп-сканер:* каждые 30 сек\n\n"
            "📋 *Команды:*\n"
            "/scan - ручное сканирование\n"
            "/status - состояние бота\n"
            "/stats - статистика сигналов\n"
            "/groups - информация о группах",
            parse_mode='Markdown'
        )
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if str(update.effective_chat.id) != STATS_SETTINGS['stats_chat_id']:
            await update.message.reply_text("❌ Эта команда доступна только в группе статистики")
            return
        
        if not hasattr(self.bot, 'stats'):
            await update.message.reply_text("❌ Статистика не инициализирована")
            return
        
        text = update.message.text
        parts = text.split()
        
        days = 7
        signal_type = 'all'
        coin = None
        
        if len(parts) > 1:
            for part in parts[1:]:
                if part.isdigit():
                    days = int(part)
                elif part.lower() in ['pump', 'pumps']:
                    signal_type = 'pump'
                elif part.lower() in ['regular', 'обычные']:
                    signal_type = 'regular'
                elif part.lower() in ['accumulation', 'накопление']:
                    signal_type = 'accumulation'
                elif part.upper() in [p.split('/')[0] for p in PAIRS_TO_SCAN]:
                    coin = part.upper()
        
        stats = self.bot.stats.get_statistics(
            days=days, 
            signal_type=signal_type,
            coin=coin
        )
        
        msg = self.bot.stats.format_stats_message(stats, days, signal_type, coin)
        
        keyboard = [
            [InlineKeyboardButton("📊 Общая", callback_data="stats_7"),
             InlineKeyboardButton("🚀 Пампы", callback_data="stats_pump_7")],
            [InlineKeyboardButton("📦 Накопление", callback_data="stats_accum_7"),
             InlineKeyboardButton("📈 По монетам", callback_data="stats_coins")],
            [InlineKeyboardButton("❓ Помощь", callback_data="stats_help")]
        ]
        
        await update.message.reply_text(
            msg, 
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    async def stats_button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        
        data = query.data
        
        if data == "stats_7":
            stats = self.bot.stats.get_statistics(days=7)
            msg = self.bot.stats.format_stats_message(stats, 7)
        elif data == "stats_pump_7":
            stats = self.bot.stats.get_statistics(days=7, signal_type='pump')
            msg = self.bot.stats.format_stats_message(stats, 7, signal_type='pump')
        elif data == "stats_regular_7":
            stats = self.bot.stats.get_statistics(days=7, signal_type='regular')
            msg = self.bot.stats.format_stats_message(stats, 7, signal_type='regular')
        elif data == "stats_accum_7":
            stats = self.bot.stats.get_statistics(days=7, signal_type='accumulation')
            msg = self.bot.stats.format_stats_message(stats, 7, signal_type='accumulation')
        elif data == "stats_coins":
            coins = set()
            for signal in self.bot.stats.db['signals'].values():
                coins.add(signal['coin'])
            
            msg = "📈 *Выберите монету:*\n\n"
            keyboard = []
            row = []
            
            for i, coin in enumerate(sorted(coins)[:12]):
                row.append(InlineKeyboardButton(coin, callback_data=f"stats_coin_{coin}"))
                if len(row) == 3:
                    keyboard.append(row)
                    row = []
            if row:
                keyboard.append(row)
            
            keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="stats_back")])
            
            await query.edit_message_text(
                msg,
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return
        elif data.startswith("stats_coin_"):
            coin = data.replace("stats_coin_", "")
            stats = self.bot.stats.get_statistics(days=7, coin=coin)
            msg = self.bot.stats.format_stats_message(stats, 7, coin=coin)
        elif data == "stats_help":
            msg = """
📚 *ПОМОЩЬ ПО СТАТИСТИКЕ*

Вы можете нажимать кнопки или вводить команды:

🔹 *Простые команды:*
/stats - статистика за 7 дней
/stats 30 - статистика за 30 дней
/stats 1 - статистика за сегодня

🔹 *По типу сигналов:*
/stats pump - только пампы
/stats regular - только обычные
/stats accumulation - только накопление

🔹 *По монетам:*
/stats BTC - по Bitcoin
/stats ETH 14 - по Ethereum за 14 дней

🔹 *Примеры:*
/stats 7 pump - пампы за неделю
/stats 30 accumulation - накопление за месяц
/stats 14 BTC - по BTC за 14 дней

📌 *Совет:* Просто нажимайте кнопки! 👆
"""
        elif data == "stats_back":
            stats = self.bot.stats.get_statistics(days=7)
            msg = self.bot.stats.format_stats_message(stats, 7)
        else:
            return
        
        keyboard = [
            [InlineKeyboardButton("📊 Общая", callback_data="stats_7"),
             InlineKeyboardButton("🚀 Пампы", callback_data="stats_pump_7")],
            [InlineKeyboardButton("📦 Накопление", callback_data="stats_accum_7"),
             InlineKeyboardButton("📈 По монетам", callback_data="stats_coins")],
            [InlineKeyboardButton("❓ Помощь", callback_data="stats_help")]
        ]
        
        await query.edit_message_text(
            msg,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    async def button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        data = query.data
        logger.info(f"🖱️ Нажата кнопка: {data}")
        
        if data.startswith("copy_"):
            coin = data.replace("copy_", "")
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"<code>{coin}</code>",
                parse_mode='HTML'
            )
            await query.answer(f"✅ {coin} скопирован")
            return
        
        elif data.startswith("refresh_"):
            coin = data.replace("refresh_", "")
            await query.edit_message_text(f"🔄 Обновляю сигнал по {coin}...")
            
            if coin in self.bot.last_signals:
                signal_data = self.bot.last_signals[coin]
                signal = signal_data['signal']
                
                contract_info, df = None, None
                for fetcher in self.bot.fetchers.values():
                    if fetcher.name == signal['exchange']:
                        contract_info = await fetcher.fetch_contract_info(signal['symbol'])
                        df = await fetcher.fetch_ohlcv(signal['symbol'], TIMEFRAMES.get('current', '15m'), limit=200)
                        break
                
                pump_percent = None
                if signal.get('pump_dump') and len(signal['pump_dump']) > 0:
                    pump_percent = signal['pump_dump'][0].get('change_percent')
                
                msg, keyboard = self.bot.format_message(signal, contract_info, pump_percent)
                
                if df is not None and not df.empty:
                    df = self.bot.analyzer.calculate_indicators(df)
                    chart_buf = self.bot.chart_generator.create_chart(df, signal, coin, TIMEFRAMES.get('current', '15m'))
                    await query.message.delete()
                    await context.bot.send_photo(
                        chat_id=update.effective_chat.id,
                        photo=chart_buf,
                        caption=msg,
                        parse_mode='HTML',
                        reply_markup=keyboard
                    )
                else:
                    await query.edit_message_text(text=msg, parse_mode='HTML', reply_markup=keyboard)
                await query.answer("🔄 Сигнал обновлен")
            else:
                await query.edit_message_text(f"❌ Нет данных для {coin}")
            return
        
        elif data.startswith("details_"):
            coin = data.replace("details_", "")
            if coin in self.bot.last_signals:
                signal_data = self.bot.last_signals[coin]
                signal = signal_data['signal']
                signal_time = signal_data['time'].strftime('%Y-%m-%d %H:%M:%S')
                
                for fetcher in self.bot.fetchers.values():
                    if fetcher.name == signal['exchange']:
                        detailed, keyboard = await self.bot.get_detailed_analysis(
                            fetcher, signal['symbol'], coin, signal_time
                        )
                        await context.bot.send_message(
                            chat_id=update.effective_chat.id,
                            text=detailed,
                            parse_mode='Markdown',
                            reply_markup=keyboard
                        )
                        await query.answer("📊 Детали загружены")
                        return
            await query.answer(f"❌ Нет данных для {coin}")
            return
        
        elif data.startswith("back_"):
            coin = data.replace("back_", "")
            if coin in self.bot.last_signals:
                signal = self.bot.last_signals[coin]['signal']
                contract_info = None
                for fetcher in self.bot.fetchers.values():
                    if fetcher.name == signal['exchange']:
                        contract_info = await fetcher.fetch_contract_info(signal['symbol'])
                        break
                pump_percent = None
                if signal.get('pump_dump') and len(signal['pump_dump']) > 0:
                    pump_percent = signal['pump_dump'][0].get('change_percent')
                msg, keyboard = self.bot.format_message(signal, contract_info, pump_percent)
                await query.edit_message_text(
                    text=msg,
                    parse_mode='HTML',
                    reply_markup=keyboard
                )
                await query.answer("↩️ Возврат к сигналу")
            return
    
    def run(self):
        self.app.run_polling()

# ============== MAIN ==============

async def main():
    bot = MultiExchangeScannerBot()
    handler = TelegramHandler(bot)
    polling = asyncio.create_task(asyncio.to_thread(handler.run))
    
    try:
        await bot.run()
    finally:
        polling.cancel()

if __name__ == "__main__":
    asyncio.run(main())
