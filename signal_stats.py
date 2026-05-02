#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import asyncio
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

STATS_SETTINGS = {
    'enabled': True,
    'stats_chat_id': os.getenv('STATS_CHAT_ID', ''),
    'daily_report_time': '20:00',
    'update_interval': 300,
    'history_days': 90,
    # На Railway /tmp обнуляется при каждом redeploy. Если у вас подключён
    # persistent volume — задайте STATS_DB_PATH=/data/signals_database.json
    # в переменных окружения.
    'db_file': os.getenv('STATS_DB_PATH', '/tmp/signals_database.json'),
}

class SignalStatistics:
    def __init__(self, bot, stats_chat_id: str):
        self.bot = bot
        self.stats_chat_id = stats_chat_id
        self.db_file = STATS_SETTINGS['db_file']
        self.last_error: Optional[str] = None
        self.load_database()
    
    def load_database(self):
        try:
            with open(self.db_file, 'r', encoding='utf-8') as f:
                self.db = json.load(f)
        except FileNotFoundError:
            self.db = {
                'signals': {},
                'statistics': {
                    'total_signals': 0,
                    'by_type': {'regular': 0, 'pump': 0, 'accumulation': 0, 'vip_pump': 0, 'discovery': 0},
                    'by_power': {},
                    'by_pair': {},
                    'daily_stats': {}
                }
            }
            self.save_database()
    
    def save_database(self):
        try:
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(self.db, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 База данных сохранена ({len(self.db['signals'])} сигналов)")
        except Exception as e:
            logger.error(f"Ошибка сохранения базы: {e}")
    
    def add_signal(self, signal: Dict, signal_type: str = 'regular') -> Optional[str]:
        try:
            symbol = signal.get('symbol')
            if not symbol:
                logger.error(f"add_signal вызван без symbol: {signal!r}")
                return None

            entry_price = signal.get('entry_price') or signal.get('price')
            if entry_price is None:
                logger.error(f"add_signal: нет entry_price/price для {symbol}: {signal!r}")
                return None

            logger.info(f"🔥🔥🔥 add_signal ВЫЗВАН для {symbol} тип={signal_type}")
            logger.info(f"   STATS_CHAT_ID={self.stats_chat_id}")
            logger.info(f"   сигнал: {signal.get('direction')}, цена={entry_price}")

            signal_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"   signal_id={signal_id}")

            self.cleanup_old_signals()

            reasons = signal.get('reasons') or []
            if not isinstance(reasons, list):
                reasons = [str(reasons)]

            self.db['signals'][signal_id] = {
                'id': signal_id,
                'type': signal_type,
                'symbol': symbol,
                'coin': symbol.split('/')[0],
                'direction': signal.get('direction', ''),
                'entry_price': entry_price,
                'target_1': signal.get('target_1'),
                'target_2': signal.get('target_2'),
                'stop_loss': signal.get('stop_loss'),
                'signal_power': signal.get('signal_power', ''),
                'signal_strength': signal.get('signal_strength', 0),
                'reasons': list(reasons)[:3],
                'status': 'pending',
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'max_price': entry_price,
                'min_price': entry_price,
                'final_result': None,
                'profit_percent': 0,
            }

            self.db['statistics']['total_signals'] += 1
            self.db['statistics']['by_type'][signal_type] = self.db['statistics']['by_type'].get(signal_type, 0) + 1

            if signal_type == 'discovery':
                self.db['statistics']['by_type']['discovery'] = self.db['statistics']['by_type'].get('discovery', 0) + 1

            if signal_type == 'vip_pump':
                self.db['statistics']['by_type']['vip_pump'] = self.db['statistics']['by_type'].get('vip_pump', 0) + 1

            self.save_database()
            logger.info(f"✅ Сигнал {signal_id} сохранен в БД")
            return signal_id
        except Exception as e:
            self.last_error = f"add_signal: {e!r}"
            logger.error(f"add_signal упал для {signal!r}", exc_info=True)
            return None

    async def update_signal(self, signal_id: str, current_price: float):
        if signal_id not in self.db['signals']:
            return

        signal = self.db['signals'][signal_id]
        if signal['status'] != 'pending':
            return

        prev_status = signal['status']
        signal['max_price'] = max(signal['max_price'], current_price)
        signal['min_price'] = min(signal['min_price'], current_price)

        profit_pct = 0

        # Определяем направление (LONG или SHORT)
        is_long = 'LONG' in signal['direction'] and 'SHORT' not in signal['direction']
        
        if is_long:
            # Для LONG: цели выше, стоп ниже
            if signal['target_2'] and current_price >= signal['target_2']:
                signal['status'] = 'victory'
                signal['final_result'] = 'target_2_hit'
                profit_pct = ((signal['target_2'] - signal['entry_price']) / signal['entry_price']) * 100
            elif signal['target_1'] and current_price >= signal['target_1']:
                signal['status'] = 'profit'
                signal['final_result'] = 'target_1_hit'
                profit_pct = ((signal['target_1'] - signal['entry_price']) / signal['entry_price']) * 100
            elif signal['stop_loss'] and current_price <= signal['stop_loss']:
                signal['status'] = 'loss'
                signal['final_result'] = 'stop_loss'
                profit_pct = ((signal['stop_loss'] - signal['entry_price']) / signal['entry_price']) * 100
        else:
            # Для SHORT: цели ниже, стоп выше
            if signal['target_2'] and current_price <= signal['target_2']:
                signal['status'] = 'victory'
                signal['final_result'] = 'target_2_hit'
                profit_pct = ((signal['entry_price'] - signal['target_2']) / signal['entry_price']) * 100
            elif signal['target_1'] and current_price <= signal['target_1']:
                signal['status'] = 'profit'
                signal['final_result'] = 'target_1_hit'
                profit_pct = ((signal['entry_price'] - signal['target_1']) / signal['entry_price']) * 100
            elif signal['stop_loss'] and current_price >= signal['stop_loss']:
                signal['status'] = 'loss'
                signal['final_result'] = 'stop_loss'
                profit_pct = ((signal['entry_price'] - signal['stop_loss']) / signal['entry_price']) * 100
        
        signal['profit_percent'] = round(profit_pct, 2)
        signal['updated_at'] = datetime.now().isoformat()

        self.save_database()

        # Сигнал перешёл из pending в финальный статус — отправляем уведомление.
        if signal['status'] != prev_status and signal['status'] in ('victory', 'profit', 'loss'):
            try:
                await self.send_result_notification(signal_id)
            except Exception as e:
                self.last_error = f"send_result_notification: {e!r}"
                logger.error(f"send_result_notification упал для {signal_id}", exc_info=True)

    def cleanup_old_signals(self):
        cutoff = datetime.now() - timedelta(days=STATS_SETTINGS['history_days'])
        to_delete = []
        
        for signal_id, signal in self.db['signals'].items():
            created = datetime.fromisoformat(signal['created_at'])
            if created < cutoff:
                to_delete.append(signal_id)
        
        for signal_id in to_delete:
            del self.db['signals'][signal_id]
    
    def get_statistics(self, days: int = 7, signal_type: str = 'all', 
                      coin: str = None, power_filter: str = None) -> Dict:
        cutoff = datetime.now() - timedelta(days=days)
        
        stats = {
            'total': 0,
            'victory': 0,
            'profit': 0,
            'loss': 0,
            'pending': 0,
            'by_type': {'regular': 0, 'pump': 0, 'accumulation': 0, 'vip_pump': 0, 'discovery': 0},
            'by_power': defaultdict(lambda: {'total': 0, 'victory': 0, 'profit': 0, 'loss': 0}),
            'by_pair': defaultdict(lambda: {'total': 0, 'success': 0, 'profit_sum': 0}),
            'total_profit': 0,
            'avg_profit': 0
        }
        
        for signal_id, signal in self.db['signals'].items():
            created = datetime.fromisoformat(signal['created_at'])
            if created < cutoff:
                continue
            
            if signal_type != 'all' and signal['type'] != signal_type:
                continue
            if coin and signal['coin'] != coin:
                continue
            if power_filter and signal['signal_power'] != power_filter:
                continue
            
            stats['total'] += 1
            stats['by_type'][signal['type']] += 1
            
            if signal['status'] in ['victory', 'profit', 'loss']:
                stats[signal['status']] += 1
                stats['total_profit'] += signal.get('profit_percent', 0)
            elif signal['status'] == 'pending':
                stats['pending'] += 1
            
            power = signal['signal_power']
            stats['by_power'][power]['total'] += 1
            if signal['status'] in ['victory', 'profit']:
                stats['by_power'][power]['victory'] += 1
                if signal['status'] == 'victory':
                    stats['by_power'][power]['profit'] += 1
            elif signal['status'] == 'loss':
                stats['by_power'][power]['loss'] += 1
            
            pair = signal['coin']
            stats['by_pair'][pair]['total'] += 1
            if signal['status'] in ['victory', 'profit']:
                stats['by_pair'][pair]['success'] += 1
                stats['by_pair'][pair]['profit_sum'] += signal.get('profit_percent', 0)
        
        if stats['total'] > 0:
            successful = stats['victory'] + stats['profit']
            stats['win_rate'] = round((successful / stats['total']) * 100, 1)
            stats['avg_profit'] = round(stats['total_profit'] / stats['total'], 2)
        
        return stats
    
    def format_stats_message(self, stats: Dict, days: int, 
                            signal_type: str = 'all', coin: str = None) -> str:
        if coin:
            title = f"📊 СТАТИСТИКА ПО {coin} ЗА {days} ДНЕЙ"
        elif signal_type == 'regular':
            title = f"📊 ОБЫЧНЫЕ СИГНАЛЫ ЗА {days} ДНЕЙ"
        elif signal_type == 'pump':
            title = f"🚀 ПАМП-СИГНАЛЫ ЗА {days} ДНЕЙ"
        elif signal_type == 'accumulation':
            title = f"📦 НАКОПЛЕНИЕ ЗА {days} ДНЕЙ"
        else:
            title = f"📊 ОБЩАЯ СТАТИСТИКА ЗА {days} ДНЕЙ"
        
        if stats['total'] == 0:
            return f"{title}\n\n❌ Нет данных за этот период"
        
        msg = f"{title}\n\n"
        msg += f"📈 Всего сигналов: {stats['total']}\n"
        msg += f"🏆 Достигли цели 2: {stats['victory']}\n"
        msg += f"💰 Достигли цели 1: {stats['profit']}\n"
        msg += f"❌ Сработал стоп: {stats['loss']}\n"
        msg += f"🔄 В процессе: {stats['pending']}\n"
        msg += f"📊 Win Rate: {stats['win_rate']}%\n"
        msg += f"💵 Средняя прибыль: {stats['avg_profit']}%\n\n"
        
        if signal_type == 'all':
            msg += f"*По типам сигналов:*\n"
            msg += f"📊 Обычные: {stats['by_type']['regular']}\n"
            msg += f"🚀 Пампы: {stats['by_type']['pump']}\n"
            msg += f"👑 VIP Пампы: {stats['by_type'].get('vip_pump', 0)}\n"
            msg += f"📦 Накопление: {stats['by_type']['accumulation']}\n\n"
            msg += f"🔍 Дискавери: {stats['by_type'].get('discovery', 0)}\n\n"
        
        if stats['by_power']:
            # Сортируем по силе (сначала слабые, потом сильные)
            power_order = ['📊 СЛАБЫЙ', '🔥 СРЕДНИЙ', '🔥🔥 СИЛЬНЫЙ', '🔥🔥🔥 ОЧЕНЬ СИЛЬНЫЙ', '🔥🔥🔥🔥 ЭКСТРЕМАЛЬНЫЙ']
            msg += f"*По силе сигнала:*\n"
            for power in power_order:
                if power in stats['by_power']:
                    data = stats['by_power'][power]
                    if data['total'] > 0:
                        win = data['victory'] + data['profit']
                        rate = (win / data['total']) * 100
                        msg += f"{power}: {win}/{data['total']} ({rate:.1f}%)\n"
            msg += "\n"
        
        if not coin and stats['by_pair']:
            top_pairs = sorted(stats['by_pair'].items(), 
                              key=lambda x: x[1]['success']/x[1]['total'] if x[1]['total']>0 else 0, 
                              reverse=True)[:3]
            msg += f"*Лучшие монеты:*\n"
            for pair, data in top_pairs:
                if data['total'] > 0:
                    rate = (data['success'] / data['total']) * 100
                    msg += f"• {pair}: {data['success']}/{data['total']} ({rate:.1f}%)\n"
        
        return msg
    
    async def send_daily_report(self):
        stats = self.get_statistics(days=1)
        
        if stats['total'] == 0:
            return
        
        msg = self.format_stats_message(stats, 1)
        msg += f"\n\n📌 /stats - общая статистика"
        msg += f"\n📌 /stats pump - только пампы"
        msg += f"\n📌 /stats regular - только обычные"
        msg += f"\n📌 /stats accumulation - только накопление"
        
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        
        keyboard = [
            [InlineKeyboardButton("📊 Общая", callback_data="stats_7"),
             InlineKeyboardButton("🚀 Пампы", callback_data="stats_pump_7")],
            [InlineKeyboardButton("📦 Накопление", callback_data="stats_accum_7"),
             InlineKeyboardButton("📈 По монетам", callback_data="stats_coins")],
            [InlineKeyboardButton("❓ Помощь", callback_data="stats_help")]
        ]
        
        await self.bot.send_message(
            chat_id=self.stats_chat_id,
            text=msg,
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    async def send_result_notification(self, signal_id: str):
        logger.info(f"📤 ОТПРАВКА УВЕДОМЛЕНИЯ для {signal_id}")
        signal = self.db['signals'].get(signal_id)
        if not signal:
            return
        if not self.stats_chat_id:
            return

        emoji = {
            'victory': '🏆',
            'profit': '💰',
            'loss': '❌',
        }.get(signal['status'], '🔄')

        type_label = {
            'pump': '🚀 Памп',
            'vip_pump': '👑 VIP Памп',
            'accumulation': '📦 Накопление',
            'discovery': '🔍 Дискавери',
        }.get(signal['type'], '📊 Обычный')

        msg = f"{emoji} *Сигнал завершен*\n\n"
        msg += f"Монета: `{signal['coin']}`\n"
        msg += f"Тип: {type_label}\n"
        msg += f"Направление: {signal['direction']}\n"
        msg += f"Вход: {signal['entry_price']}\n"
        msg += f"Результат: {signal['final_result']}\n"
        msg += f"Прибыль: {signal['profit_percent']:+.2f}%\n"

        await self.bot.send_message(
            chat_id=self.stats_chat_id,
            text=msg,
            parse_mode='Markdown',
        )

    def get_health_summary(self) -> Dict:
        signals = self.db.get('signals', {})
        statuses: Dict[str, int] = defaultdict(int)
        last_created: Optional[str] = None
        for sig in signals.values():
            statuses[sig.get('status', 'unknown')] += 1
            created = sig.get('created_at')
            if created and (last_created is None or created > last_created):
                last_created = created

        try:
            db_size = os.path.getsize(self.db_file) if os.path.exists(self.db_file) else 0
        except OSError:
            db_size = -1

        return {
            'db_file': self.db_file,
            'db_size_bytes': db_size,
            'stats_chat_id': self.stats_chat_id,
            'total_signals': len(signals),
            'status_counts': dict(statuses),
            'last_signal_created_at': last_created,
            'last_error': self.last_error,
        }
