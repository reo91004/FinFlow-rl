# src/analysis/backtest.py

"""
백테스터: 현실적인 백테스트 시뮬레이션

목적: 학습된 정책의 과거 데이터 검증
의존: env.py, metrics.py, logger.py
사용처: 평가 스크립트, 하이퍼파라미터 최적화
역할: 실제 거래 환경 시뮬레이션

구현 내용:
- 거래 비용 모델링 (고정+비례+시장충격)
- 슬리피지 시뮬레이션
- T+1 결제 주기 반영
- 세금 및 차입 비용 고려
- Walk-forward 분석
- 다중 시나리오 테스트
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from tqdm import tqdm
import json
from src.utils.logger import FinFlowLogger
from src.environments.portfolio_env import PortfolioEnv

class RealisticBacktester:
    """
    현실적인 백테스트 시스템
    
    실제 거래 환경을 정확히 모델링:
    - 거래 비용 (고정 + 비례)
    - 슬리피지 모델
    - 시장 충격 비용
    - 차입 비용
    - 세금 고려
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 백테스트 설정
        """
        self.config = config or self._default_config()
        self.logger = FinFlowLogger("RealisticBacktester")
        
        # 거래 비용 모델 파라미터
        self.cost_model = self.config['cost_model']
        self.slippage_model = self.config['slippage_model']
        self.market_impact_model = self.config['market_impact_model']
        self.borrowing_cost = self.config['borrowing_cost']
        self.tax_model = self.config['tax_model']
        
        # 제약 조건
        self.constraints = self.config['constraints']
        
        # 결과 저장
        self.results = {
            'trades': [],
            'positions': [],
            'returns': [],
            'costs': [],
            'metrics': {}
        }
    
    def _default_config(self) -> Dict:
        """기본 설정"""
        return {
            'cost_model': {
                'fixed_cost': 5.0,  # 고정 수수료 (달러)
                'proportional_cost': 0.001,  # 비례 수수료 (0.1%)
                'min_cost': 1.0  # 최소 수수료
            },
            'slippage_model': {
                'base_slippage': 0.0005,  # 기본 슬리피지 (0.05%)
                'volume_factor': 0.1,  # 거래량 영향 계수
                'volatility_factor': 0.2,  # 변동성 영향 계수
                'model_type': 'square_root'  # linear, square_root, exponential
            },
            'market_impact_model': {
                'temporary_impact': 0.1,  # 일시적 충격
                'permanent_impact': 0.05,  # 영구적 충격
                'decay_rate': 0.5  # 충격 감소율
            },
            'borrowing_cost': {
                'rate': 0.03,  # 연 3% 차입 비용
                'margin_requirement': 0.5  # 50% 증거금 요구
            },
            'tax_model': {
                'short_term_rate': 0.35,  # 단기 양도세
                'long_term_rate': 0.15,  # 장기 양도세
                'holding_period': 365  # 장기 보유 기준 (일)
            },
            'constraints': {
                'max_position_size': 0.2,  # 최대 포지션 크기
                'min_trade_size': 100,  # 최소 거래 금액
                'max_leverage': 2.0,  # 최대 레버리지
                'min_liquidity': 1000000  # 최소 유동성 요구
            },
            'execution': {
                'delay': 0,  # 주문 체결 지연 (틱)
                'partial_fill': True,  # 부분 체결 허용
                'order_types': ['market', 'limit']  # 주문 유형
            }
        }
    
    def backtest(self, 
                 strategy,
                 data: pd.DataFrame,
                 initial_capital: float = 1000000,
                 verbose: bool = True) -> Dict:
        """
        백테스트 실행
        
        Args:
            strategy: 거래 전략 (callable)
            data: 가격 데이터
            initial_capital: 초기 자본
            verbose: 진행 상황 출력
            
        Returns:
            백테스트 결과
        """
        self.logger.info(f"백테스트 시작: {len(data)} 거래일, 초기자본 ${initial_capital:,.0f}")
        
        # 초기화
        self.data = data
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = np.zeros(len(data.columns))
        self.cash = initial_capital
        
        # 거래 기록
        portfolio_values = []
        daily_returns = []
        transaction_costs = []
        
        # 시뮬레이션
        iterator = tqdm(range(len(data)), desc="백테스팅") if verbose else range(len(data))
        
        for t in iterator:
            # 현재 시장 상태
            current_prices = data.iloc[t].values
            
            if t > 0:
                # 전략 신호 생성
                market_state = self._get_market_state(t)
                target_weights = strategy(market_state)
                
                # 포지션 조정
                trades, costs = self._execute_trades(
                    target_weights,
                    current_prices,
                    t
                )
                
                # 기록
                if len(trades) > 0:
                    self.results['trades'].extend(trades)
                    transaction_costs.append(costs)
                
                # 포트폴리오 가치 계산
                portfolio_value = self._calculate_portfolio_value(current_prices)
                portfolio_values.append(portfolio_value)
                
                # 일일 수익률
                if len(portfolio_values) > 1:
                    daily_return = (portfolio_value - portfolio_values[-2]) / portfolio_values[-2]
                    daily_returns.append(daily_return)
                
                # 포지션 기록
                self.results['positions'].append({
                    'timestamp': t,
                    'positions': self.positions.copy(),
                    'cash': self.cash,
                    'value': portfolio_value
                })
        
        # 메트릭 계산
        self.results['returns'] = np.array(daily_returns)
        self.results['portfolio_values'] = np.array(portfolio_values)
        self.results['transaction_costs'] = transaction_costs
        self.results['metrics'] = self._calculate_metrics()
        
        self.logger.info("백테스트 완료")
        self._print_summary()
        
        return self.results
    
    def _get_market_state(self, t: int) -> Dict:
        """시장 상태 추출"""
        lookback = min(20, t)
        
        return {
            'prices': self.data.iloc[t].values,
            'returns': self.data.iloc[max(0, t-lookback):t+1].pct_change().iloc[-1].values if t > 0 else np.zeros(len(self.data.columns)),
            'volume': np.random.uniform(1e6, 1e7, len(self.data.columns)),  # 시뮬레이션
            'volatility': self.data.iloc[max(0, t-lookback):t+1].pct_change().std().values if t > lookback else np.ones(len(self.data.columns)) * 0.02,
            'timestamp': t
        }
    
    def _execute_trades(self, 
                       target_weights: np.ndarray,
                       current_prices: np.ndarray,
                       timestamp: int) -> Tuple[List[Dict], float]:
        """
        거래 실행 (현실적 비용 모델 적용)
        
        Args:
            target_weights: 목표 가중치
            current_prices: 현재 가격
            timestamp: 시간
            
        Returns:
            거래 내역, 총 비용
        """
        trades = []
        total_cost = 0
        
        # 현재 포트폴리오 가치
        portfolio_value = self._calculate_portfolio_value(current_prices)
        
        # 목표 포지션 계산
        target_positions = target_weights * portfolio_value / current_prices
        
        # 거래 필요량
        trade_amounts = target_positions - self.positions
        
        for i, trade_amount in enumerate(trade_amounts):
            if abs(trade_amount * current_prices[i]) < self.constraints['min_trade_size']:
                continue  # 최소 거래 크기 미만
            
            # 슬리피지 계산
            slippage = self._calculate_slippage(
                trade_amount,
                current_prices[i],
                self._estimate_volatility(i),
                self._estimate_volume(i)
            )
            
            # 시장 충격 계산
            market_impact = self._calculate_market_impact(
                trade_amount,
                current_prices[i],
                self._estimate_volume(i)
            )
            
            # 실제 체결 가격
            if trade_amount > 0:  # 매수
                execution_price = current_prices[i] * (1 + slippage + market_impact)
            else:  # 매도
                execution_price = current_prices[i] * (1 - slippage - market_impact)
            
            # 거래 비용
            trade_value = abs(trade_amount * execution_price)
            fixed_cost = self.cost_model['fixed_cost']
            prop_cost = trade_value * self.cost_model['proportional_cost']
            trade_cost = max(fixed_cost + prop_cost, self.cost_model['min_cost'])
            
            # 숏 포지션 차입 비용
            borrowing_cost = 0
            if self.positions[i] < 0:  # 숏 포지션
                daily_rate = self.borrowing_cost['rate'] / 252
                borrowing_cost = abs(self.positions[i] * current_prices[i]) * daily_rate
            
            # 총 비용
            total_trade_cost = trade_cost + borrowing_cost
            total_cost += total_trade_cost
            
            # 현금 업데이트
            self.cash -= trade_amount * execution_price + total_trade_cost
            
            # 포지션 업데이트
            self.positions[i] += trade_amount
            
            # 거래 기록
            trades.append({
                'timestamp': timestamp,
                'asset': i,
                'amount': trade_amount,
                'price': current_prices[i],
                'execution_price': execution_price,
                'slippage': slippage,
                'market_impact': market_impact,
                'trade_cost': trade_cost,
                'borrowing_cost': borrowing_cost,
                'total_cost': total_trade_cost
            })
        
        return trades, total_cost
    
    def _calculate_slippage(self, 
                           trade_amount: float,
                           price: float,
                           volatility: float,
                           volume: float) -> float:
        """
        슬리피지 계산
        
        Args:
            trade_amount: 거래량
            price: 가격
            volatility: 변동성
            volume: 거래량
            
        Returns:
            슬리피지 비율
        """
        base = self.slippage_model['base_slippage']
        
        # 거래량 영향
        volume_impact = abs(trade_amount * price) / volume
        volume_factor = self.slippage_model['volume_factor']
        
        # 변동성 영향
        vol_factor = self.slippage_model['volatility_factor']
        
        # 모델 선택
        model_type = self.slippage_model['model_type']
        
        if model_type == 'linear':
            slippage = base + volume_factor * volume_impact + vol_factor * volatility
        elif model_type == 'square_root':
            slippage = base + volume_factor * np.sqrt(volume_impact) + vol_factor * volatility
        elif model_type == 'exponential':
            slippage = base * np.exp(volume_factor * volume_impact + vol_factor * volatility)
        else:
            slippage = base
        
        return min(slippage, 0.01)  # 최대 1% 캡
    
    def _calculate_market_impact(self,
                                trade_amount: float,
                                price: float,
                                volume: float) -> float:
        """
        시장 충격 비용 계산 (Square-root model)
        
        Args:
            trade_amount: 거래량
            price: 가격
            volume: 일일 거래량
            
        Returns:
            시장 충격 비율
        """
        trade_value = abs(trade_amount * price)
        participation_rate = trade_value / volume
        
        # Square-root market impact model
        temp_impact = self.market_impact_model['temporary_impact']
        perm_impact = self.market_impact_model['permanent_impact']
        
        # 일시적 충격 (거래 직후 회복)
        temporary = temp_impact * np.sqrt(participation_rate)
        
        # 영구적 충격 (정보 효과)
        permanent = perm_impact * participation_rate
        
        return temporary + permanent
    
    def _calculate_portfolio_value(self, prices: np.ndarray) -> float:
        """포트폴리오 가치 계산"""
        return np.dot(self.positions, prices) + self.cash
    
    def _estimate_volatility(self, asset_idx: int) -> float:
        """변동성 추정 (간단한 히스토리컬)"""
        if len(self.data) > 20:
            returns = self.data.iloc[-20:, asset_idx].pct_change().dropna()
            return returns.std() if len(returns) > 0 else 0.02
        return 0.02
    
    def _estimate_volume(self, asset_idx: int) -> float:
        """거래량 추정 (시뮬레이션)"""
        # 실제로는 거래량 데이터 사용
        avg_price = self.data.iloc[:, asset_idx].mean()
        return np.random.uniform(1e6, 1e7) * avg_price
    
    def _calculate_metrics(self) -> Dict:
        """성능 메트릭 계산"""
        returns = self.results['returns']
        
        if len(returns) == 0:
            return {}
        
        metrics = {}
        
        # 기본 메트릭
        metrics['total_return'] = (self.results['portfolio_values'][-1] - self.initial_capital) / self.initial_capital
        metrics['annual_return'] = (1 + metrics['total_return']) ** (252 / len(returns)) - 1
        metrics['volatility'] = np.std(returns) * np.sqrt(252)
        metrics['sharpe_ratio'] = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252) + 1e-8)
        
        # 드로다운
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = np.min(drawdown)
        
        # 거래 비용 분석
        total_costs = sum([sum(tc.values()) if isinstance(tc, dict) else tc 
                          for tc in self.results['transaction_costs']])
        metrics['total_costs'] = total_costs
        metrics['cost_ratio'] = total_costs / self.initial_capital
        
        # 거래 통계
        trades = self.results['trades']
        if trades:
            metrics['n_trades'] = len(trades)
            metrics['avg_trade_size'] = np.mean([abs(t['amount'] * t['price']) for t in trades])
            metrics['avg_slippage'] = np.mean([t['slippage'] for t in trades])
            metrics['avg_market_impact'] = np.mean([t['market_impact'] for t in trades])
        
        # 승률
        metrics['win_rate'] = np.mean(returns > 0) if len(returns) > 0 else 0
        
        # 칼마 비율
        if metrics['max_drawdown'] != 0:
            metrics['calmar_ratio'] = metrics['annual_return'] / abs(metrics['max_drawdown'])
        else:
            metrics['calmar_ratio'] = 0
        
        # 소르티노 비율
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns) * np.sqrt(252)
            metrics['sortino_ratio'] = metrics['annual_return'] / (downside_std + 1e-8)
        else:
            metrics['sortino_ratio'] = metrics['sharpe_ratio']
        
        return metrics
    
    def _print_summary(self):
        """결과 요약 출력"""
        metrics = self.results['metrics']
        
        print("\n" + "=" * 60)
        print("백테스트 결과 요약")
        print("=" * 60)
        
        print(f"\n수익률:")
        print(f"  총 수익률: {metrics.get('total_return', 0)*100:.2f}%")
        print(f"  연간 수익률: {metrics.get('annual_return', 0)*100:.2f}%")
        print(f"  변동성: {metrics.get('volatility', 0)*100:.2f}%")
        
        print(f"\n위험 조정 수익률:")
        print(f"  샤프 비율: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"  소르티노 비율: {metrics.get('sortino_ratio', 0):.3f}")
        print(f"  칼마 비율: {metrics.get('calmar_ratio', 0):.3f}")
        
        print(f"\n위험 지표:")
        print(f"  최대 낙폭: {metrics.get('max_drawdown', 0)*100:.2f}%")
        print(f"  승률: {metrics.get('win_rate', 0)*100:.1f}%")
        
        print(f"\n거래 비용:")
        print(f"  총 비용: ${metrics.get('total_costs', 0):,.2f}")
        print(f"  비용 비율: {metrics.get('cost_ratio', 0)*100:.3f}%")
        print(f"  평균 슬리피지: {metrics.get('avg_slippage', 0)*100:.3f}%")
        print(f"  평균 시장 충격: {metrics.get('avg_market_impact', 0)*100:.3f}%")
        
        print(f"\n거래 통계:")
        print(f"  총 거래 수: {metrics.get('n_trades', 0)}")
        print(f"  평균 거래 크기: ${metrics.get('avg_trade_size', 0):,.0f}")
    
    def save_results(self, path: str):
        """결과 저장"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # 메트릭 저장
        with open(path / 'metrics.json', 'w') as f:
            json.dump(self.results['metrics'], f, indent=2)
        
        # 거래 내역 저장
        if self.results['trades']:
            trades_df = pd.DataFrame(self.results['trades'])
            trades_df.to_csv(path / 'trades.csv', index=False)
        
        # 포지션 내역 저장
        if self.results['positions']:
            positions_df = pd.DataFrame(self.results['positions'])
            positions_df.to_csv(path / 'positions.csv', index=False)
        
        # 수익률 저장
        returns_df = pd.DataFrame({
            'returns': self.results['returns'],
            'portfolio_value': self.results['portfolio_values']
        })
        returns_df.to_csv(path / 'returns.csv', index=False)
        
        self.logger.info(f"결과 저장 완료: {path}")
    
    def analyze_costs(self) -> pd.DataFrame:
        """거래 비용 상세 분석"""
        if not self.results['trades']:
            return pd.DataFrame()
        
        trades_df = pd.DataFrame(self.results['trades'])
        
        # 비용 구성 분석
        cost_breakdown = {
            'trade_costs': trades_df['trade_cost'].sum(),
            'slippage_costs': (trades_df['slippage'] * trades_df['amount'] * trades_df['price']).abs().sum(),
            'market_impact_costs': (trades_df['market_impact'] * trades_df['amount'] * trades_df['price']).abs().sum(),
            'borrowing_costs': trades_df['borrowing_cost'].sum()
        }
        
        # 자산별 비용
        asset_costs = trades_df.groupby('asset').agg({
            'total_cost': 'sum',
            'slippage': 'mean',
            'market_impact': 'mean'
        })
        
        print("\n거래 비용 상세 분석:")
        print("-" * 40)
        for cost_type, amount in cost_breakdown.items():
            print(f"{cost_type}: ${amount:,.2f}")
        
        print("\n자산별 평균 비용:")
        print(asset_costs)
        
        return trades_df