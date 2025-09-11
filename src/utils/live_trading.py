# src/utils/live_trading.py

import numpy as np
import pandas as pd
import torch
import asyncio
import websocket
import json
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path
from datetime import datetime, timedelta
import yfinance as yf
from collections import deque
import time
from src.utils.logger import FinFlowLogger
from src.core.env import PortfolioEnv
from src.agents.b_cell import BCell
from src.agents.t_cell import TCell
from src.agents.memory import MemoryCell
from src.agents.gating import GatingNetwork

class LiveTradingSystem:
    """
    실거래 시스템
    
    실시간 데이터 수신, 거래 신호 생성, 주문 실행
    Paper Trading과 Real Trading 모드 지원
    """
    
    def __init__(self, 
                 model_path: str,
                 config: Optional[Dict] = None,
                 mode: str = 'paper',
                 broker: str = 'alpaca'):
        """
        Args:
            model_path: 학습된 모델 경로
            config: 거래 설정
            mode: 'paper' or 'live'
            broker: 브로커 ('alpaca', 'interactive_brokers', 'binance')
        """
        self.model_path = model_path
        self.config = config or self._default_config()
        self.mode = mode
        self.broker = broker
        
        self.logger = FinFlowLogger(f"LiveTrading-{mode}")
        self.logger.info(f"실거래 시스템 초기화: {mode} 모드, {broker} 브로커")
        
        # 모델 로드
        self._load_models()
        
        # 브로커 연결
        self.broker_client = self._init_broker()
        
        # 실시간 데이터
        self.market_data = {}
        self.price_buffer = deque(maxlen=100)
        self.volume_buffer = deque(maxlen=100)
        
        # 포지션 관리
        self.current_positions = {}
        self.pending_orders = {}
        self.executed_orders = []
        
        # 위험 관리
        self.risk_manager = RiskManager(self.config['risk'])
        
        # 성능 추적
        self.performance_tracker = PerformanceTracker()
        
        # 거래 상태
        self.is_trading = False
        self.last_rebalance = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _default_config(self) -> Dict:
        """기본 설정"""
        return {
            'symbols': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 
                       'NVDA', 'TSLA', 'JPM', 'V', 'JNJ'],
            'rebalance_frequency': 'daily',  # 'minute', 'hourly', 'daily'
            'max_position_size': 0.2,
            'min_position_size': 0.01,
            'initial_capital': 100000,
            'risk': {
                'max_drawdown': 0.25,
                'position_limit': 0.2,
                'daily_loss_limit': 0.05,
                'var_limit': 0.02,
                'stop_loss': 0.1,
                'take_profit': 0.5
            },
            'execution': {
                'order_type': 'market',  # 'market', 'limit'
                'slippage_buffer': 0.001,
                'timeout': 30,
                'retry_count': 3
            },
            'data': {
                'lookback_period': 30,
                'update_frequency': 1,  # seconds
                'data_source': 'yfinance'  # 'yfinance', 'alpaca', 'polygon'
            }
        }
    
    def _load_models(self):
        """모델 로드"""
        self.logger.info(f"모델 로드: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 차원 정보
        state_dim = checkpoint.get('state_dim', 43)
        action_dim = checkpoint.get('action_dim', len(self.config['symbols']))
        
        # 에이전트 초기화
        self.b_cell = BCell(state_dim, action_dim, device=self.device)
        self.b_cell.load_state_dict(checkpoint['b_cell'])
        self.b_cell.eval()
        
        self.t_cell = TCell()
        if 't_cell' in checkpoint:
            self.t_cell.load_state(checkpoint['t_cell'])
        
        self.memory_cell = MemoryCell()
        if 'memory_cell' in checkpoint:
            self.memory_cell.memories = checkpoint['memory_cell'].get('memories', [])
        
        self.gating_network = GatingNetwork(state_dim).to(self.device)
        if 'gating_network' in checkpoint:
            self.gating_network.load_state_dict(checkpoint['gating_network'])
        self.gating_network.eval()
        
        self.logger.info("모델 로드 완료")
    
    def _init_broker(self):
        """브로커 클라이언트 초기화"""
        if self.broker == 'alpaca':
            return AlpacaBroker(self.config, self.mode)
        elif self.broker == 'interactive_brokers':
            return IBBroker(self.config, self.mode)
        elif self.broker == 'binance':
            return BinanceBroker(self.config, self.mode)
        else:
            raise ValueError(f"지원하지 않는 브로커: {self.broker}")
    
    def start_trading(self):
        """거래 시작"""
        self.logger.info("=" * 50)
        self.logger.info("실거래 시작")
        self.logger.info(f"모드: {self.mode}")
        self.logger.info(f"종목: {self.config['symbols']}")
        self.logger.info(f"초기 자본: ${self.config['initial_capital']:,.0f}")
        self.logger.info("=" * 50)
        
        self.is_trading = True
        
        # 실시간 데이터 스트림 시작
        self._start_data_stream()
        
        # 거래 루프 시작
        self._trading_loop()
    
    def stop_trading(self):
        """거래 중지"""
        self.logger.info("거래 중지 요청")
        self.is_trading = False
        
        # 모든 포지션 청산
        if self.config.get('close_all_on_stop', True):
            self._close_all_positions()
        
        # 성능 리포트
        self.performance_tracker.generate_report()
        
        self.logger.info("거래 중지 완료")
    
    def _start_data_stream(self):
        """실시간 데이터 스트림 시작"""
        self.logger.info("실시간 데이터 스트림 시작")
        
        # WebSocket 또는 REST API를 통한 데이터 수신
        if self.config['data']['data_source'] == 'yfinance':
            # yfinance는 실시간 스트림 미지원, 폴링 방식 사용
            self._start_polling_data()
        else:
            # 브로커별 실시간 데이터 스트림
            self.broker_client.start_data_stream(
                self.config['symbols'],
                self._on_price_update
            )
    
    def _start_polling_data(self):
        """폴링 방식 데이터 업데이트 (yfinance)"""
        def poll_data():
            while self.is_trading:
                try:
                    # 최신 데이터 가져오기
                    for symbol in self.config['symbols']:
                        ticker = yf.Ticker(symbol)
                        data = ticker.history(period='1d', interval='1m')
                        
                        if not data.empty:
                            latest = data.iloc[-1]
                            self.market_data[symbol] = {
                                'price': latest['Close'],
                                'volume': latest['Volume'],
                                'timestamp': datetime.now()
                            }
                    
                    # 업데이트 주기
                    time.sleep(self.config['data']['update_frequency'])
                    
                except Exception as e:
                    self.logger.error(f"데이터 폴링 오류: {e}")
                    time.sleep(5)
        
        # 별도 스레드에서 실행
        data_thread = threading.Thread(target=poll_data)
        data_thread.daemon = True
        data_thread.start()
    
    def _on_price_update(self, symbol: str, price: float, volume: float):
        """가격 업데이트 콜백"""
        self.market_data[symbol] = {
            'price': price,
            'volume': volume,
            'timestamp': datetime.now()
        }
        
        # 버퍼 업데이트
        self.price_buffer.append({symbol: price})
        self.volume_buffer.append({symbol: volume})
    
    def _trading_loop(self):
        """메인 거래 루프"""
        while self.is_trading:
            try:
                # 시장 시간 확인
                if not self._is_market_open():
                    time.sleep(60)
                    continue
                
                # 리밸런싱 시간 확인
                if self._should_rebalance():
                    self.logger.info("리밸런싱 시작")
                    
                    # 시장 상태 생성
                    state = self._get_market_state()
                    
                    # 거래 신호 생성
                    target_weights = self._generate_signals(state)
                    
                    # 위험 관리 체크
                    if self.risk_manager.check_risk(target_weights, self.current_positions):
                        # 주문 실행
                        self._execute_orders(target_weights)
                        self.last_rebalance = datetime.now()
                    else:
                        self.logger.warning("위험 한도 초과, 거래 건너뜀")
                    
                    # 성능 업데이트
                    self._update_performance()
                
                # 대기
                time.sleep(1)
                
            except KeyboardInterrupt:
                self.logger.info("사용자 중단")
                break
            except Exception as e:
                self.logger.error(f"거래 루프 오류: {e}")
                time.sleep(5)
    
    def _is_market_open(self) -> bool:
        """시장 개장 여부 확인"""
        now = datetime.now()
        
        # 주말 제외
        if now.weekday() >= 5:
            return False
        
        # 거래 시간 확인 (미국 시장 기준)
        market_open = now.replace(hour=9, minute=30, second=0)
        market_close = now.replace(hour=16, minute=0, second=0)
        
        return market_open <= now <= market_close
    
    def _should_rebalance(self) -> bool:
        """리밸런싱 필요 여부 확인"""
        if self.last_rebalance is None:
            return True
        
        freq = self.config['rebalance_frequency']
        now = datetime.now()
        
        if freq == 'minute':
            return (now - self.last_rebalance).seconds >= 60
        elif freq == 'hourly':
            return (now - self.last_rebalance).seconds >= 3600
        elif freq == 'daily':
            return now.date() > self.last_rebalance.date()
        
        return False
    
    def _get_market_state(self) -> np.ndarray:
        """현재 시장 상태 생성"""
        # 가격 데이터 수집
        prices = []
        volumes = []
        
        for symbol in self.config['symbols']:
            if symbol in self.market_data:
                prices.append(self.market_data[symbol]['price'])
                volumes.append(self.market_data[symbol]['volume'])
            else:
                prices.append(0)
                volumes.append(0)
        
        # 특징 추출 (간단한 버전)
        prices = np.array(prices)
        volumes = np.array(volumes)
        
        # 수익률 계산
        if len(self.price_buffer) > 1:
            prev_prices = np.array([
                self.price_buffer[-2].get(s, prices[i]) 
                for i, s in enumerate(self.config['symbols'])
            ])
            returns = (prices - prev_prices) / (prev_prices + 1e-8)
        else:
            returns = np.zeros_like(prices)
        
        # 현재 포지션
        positions = np.array([
            self.current_positions.get(s, 0) 
            for s in self.config['symbols']
        ])
        
        # 위기 레벨 (간단한 추정)
        volatility = np.std(returns) if len(returns) > 0 else 0.02
        crisis_level = min(volatility / 0.02, 1.0)
        
        # 상태 벡터 구성
        state = np.concatenate([
            returns,
            positions / (positions.sum() + 1e-8),
            [crisis_level]
        ])
        
        return state
    
    def _generate_signals(self, state: np.ndarray) -> np.ndarray:
        """거래 신호 생성"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # 위기 감지
            crisis_info = self.t_cell.detect_crisis(self.market_data)
            
            # 메모리 가이던스
            memory_guidance = self.memory_cell.get_memory_guidance(
                state, crisis_info['overall_crisis']
            )
            
            # 게이팅 결정
            gating_decision = self.gating_network(
                state_tensor, memory_guidance, crisis_info['overall_crisis']
            )
            
            # 액션 선택
            action = self.b_cell.select_action(
                state_tensor,
                bcell_type=gating_decision.selected_bcell,
                deterministic=True
            )
        
        # 가중치 정규화
        weights = action / (action.sum() + 1e-8)
        
        # 제약 조건 적용
        weights = np.clip(weights, 
                         self.config['min_position_size'],
                         self.config['max_position_size'])
        weights = weights / weights.sum()
        
        return weights
    
    def _execute_orders(self, target_weights: np.ndarray):
        """주문 실행"""
        self.logger.info("주문 실행 시작")
        
        # 현재 포트폴리오 가치
        portfolio_value = self._get_portfolio_value()
        
        for i, symbol in enumerate(self.config['symbols']):
            target_value = target_weights[i] * portfolio_value
            current_value = self.current_positions.get(symbol, 0) * self.market_data[symbol]['price']
            
            diff_value = target_value - current_value
            
            # 최소 거래 금액 체크
            if abs(diff_value) < 100:
                continue
            
            # 주문 생성
            if diff_value > 0:
                # 매수
                shares = int(diff_value / self.market_data[symbol]['price'])
                if shares > 0:
                    order = self._create_order(symbol, 'buy', shares)
                    self._submit_order(order)
            else:
                # 매도
                shares = int(-diff_value / self.market_data[symbol]['price'])
                if shares > 0:
                    order = self._create_order(symbol, 'sell', shares)
                    self._submit_order(order)
    
    def _create_order(self, symbol: str, side: str, quantity: int) -> Dict:
        """주문 생성"""
        order = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'type': self.config['execution']['order_type'],
            'time_in_force': 'day',
            'timestamp': datetime.now()
        }
        
        # Limit 주문인 경우 가격 설정
        if order['type'] == 'limit':
            current_price = self.market_data[symbol]['price']
            if side == 'buy':
                order['limit_price'] = current_price * (1 - self.config['execution']['slippage_buffer'])
            else:
                order['limit_price'] = current_price * (1 + self.config['execution']['slippage_buffer'])
        
        return order
    
    def _submit_order(self, order: Dict):
        """주문 제출"""
        try:
            if self.mode == 'paper':
                # Paper Trading
                self._simulate_order_execution(order)
            else:
                # Live Trading
                order_id = self.broker_client.submit_order(order)
                self.pending_orders[order_id] = order
                self.logger.info(f"주문 제출: {order}")
        except Exception as e:
            self.logger.error(f"주문 제출 실패: {e}")
    
    def _simulate_order_execution(self, order: Dict):
        """주문 실행 시뮬레이션 (Paper Trading)"""
        symbol = order['symbol']
        quantity = order['quantity']
        side = order['side']
        
        # 가상 체결
        execution_price = self.market_data[symbol]['price']
        
        # 슬리피지 적용
        slippage = np.random.uniform(0, self.config['execution']['slippage_buffer'])
        if side == 'buy':
            execution_price *= (1 + slippage)
            self.current_positions[symbol] = self.current_positions.get(symbol, 0) + quantity
        else:
            execution_price *= (1 - slippage)
            self.current_positions[symbol] = self.current_positions.get(symbol, 0) - quantity
        
        # 실행 기록
        self.executed_orders.append({
            'order': order,
            'execution_price': execution_price,
            'execution_time': datetime.now(),
            'slippage': slippage
        })
        
        self.logger.info(f"주문 체결 (시뮬레이션): {symbol} {side} {quantity}주 @ ${execution_price:.2f}")
    
    def _get_portfolio_value(self) -> float:
        """포트폴리오 가치 계산"""
        value = self.config['initial_capital']  # 현금
        
        for symbol, quantity in self.current_positions.items():
            if symbol in self.market_data:
                value += quantity * self.market_data[symbol]['price']
        
        return value
    
    def _update_performance(self):
        """성능 업데이트"""
        portfolio_value = self._get_portfolio_value()
        self.performance_tracker.update(
            portfolio_value,
            self.current_positions,
            self.executed_orders
        )
    
    def _close_all_positions(self):
        """모든 포지션 청산"""
        self.logger.info("모든 포지션 청산")
        
        for symbol, quantity in self.current_positions.items():
            if quantity > 0:
                order = self._create_order(symbol, 'sell', quantity)
                self._submit_order(order)
            elif quantity < 0:
                order = self._create_order(symbol, 'buy', -quantity)
                self._submit_order(order)


class RiskManager:
    """위험 관리 모듈"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = FinFlowLogger("RiskManager")
        self.daily_pnl = 0
        self.max_portfolio_value = 0
        
    def check_risk(self, target_weights: np.ndarray, current_positions: Dict) -> bool:
        """위험 체크"""
        # 포지션 한도 체크
        if np.max(target_weights) > self.config['position_limit']:
            self.logger.warning("포지션 한도 초과")
            return False
        
        # 일일 손실 한도 체크
        if self.daily_pnl < -self.config['daily_loss_limit']:
            self.logger.warning("일일 손실 한도 초과")
            return False
        
        # VAR 체크
        var_estimate = self._calculate_var(target_weights)
        if var_estimate > self.config['var_limit']:
            self.logger.warning(f"VaR 한도 초과: {var_estimate:.2%}")
            return False
        
        return True
    
    def _calculate_var(self, weights: np.ndarray, confidence: float = 0.95) -> float:
        """Value at Risk 계산"""
        # 간단한 파라메트릭 VaR
        portfolio_volatility = 0.02  # 추정치
        z_score = 1.645 if confidence == 0.95 else 2.326
        var = portfolio_volatility * z_score * np.sqrt(1/252)
        return var


class PerformanceTracker:
    """성능 추적 모듈"""
    
    def __init__(self):
        self.portfolio_values = []
        self.timestamps = []
        self.trades = []
        self.positions_history = []
        
    def update(self, portfolio_value: float, positions: Dict, trades: List):
        """성능 업데이트"""
        self.portfolio_values.append(portfolio_value)
        self.timestamps.append(datetime.now())
        self.positions_history.append(positions.copy())
        self.trades.extend(trades)
    
    def generate_report(self) -> Dict:
        """성능 리포트 생성"""
        if len(self.portfolio_values) < 2:
            return {}
        
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        
        report = {
            'total_return': (self.portfolio_values[-1] - self.portfolio_values[0]) / self.portfolio_values[0],
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(),
            'n_trades': len(self.trades),
            'win_rate': self._calculate_win_rate()
        }
        
        return report
    
    def _calculate_max_drawdown(self) -> float:
        """최대 낙폭 계산"""
        cumulative = np.array(self.portfolio_values)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def _calculate_win_rate(self) -> float:
        """승률 계산"""
        if not self.trades:
            return 0
        
        winning_trades = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
        return winning_trades / len(self.trades)


# 브로커 인터페이스 (예시)
class AlpacaBroker:
    """Alpaca 브로커 클라이언트"""
    
    def __init__(self, config: Dict, mode: str):
        self.config = config
        self.mode = mode
        # 실제 구현 시 Alpaca API 키 설정
        
    def submit_order(self, order: Dict) -> str:
        """주문 제출"""
        # Alpaca API 호출
        return f"order_{datetime.now().timestamp()}"
    
    def start_data_stream(self, symbols: List[str], callback: Callable):
        """데이터 스트림 시작"""
        # Alpaca WebSocket 연결
        pass


class IBBroker:
    """Interactive Brokers 클라이언트"""
    
    def __init__(self, config: Dict, mode: str):
        self.config = config
        self.mode = mode
        # IB Gateway 연결
        
    def submit_order(self, order: Dict) -> str:
        """주문 제출"""
        # IB API 호출
        return f"order_{datetime.now().timestamp()}"


class BinanceBroker:
    """Binance 클라이언트"""
    
    def __init__(self, config: Dict, mode: str):
        self.config = config
        self.mode = mode
        # Binance API 설정
        
    def submit_order(self, order: Dict) -> str:
        """주문 제출"""
        # Binance API 호출
        return f"order_{datetime.now().timestamp()}"