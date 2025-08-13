# core/backtester.py

import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import matplotlib.pyplot as plt
import torch
import gc
from datetime import datetime, timedelta
from constant import *
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional
from .system import ImmunePortfolioSystem
from .reward import RewardCalculator
from .curriculum import CurriculumLearningManager
from xai import generate_dashboard, create_visualizations
from constant import *
from utils.checkpoint import CheckpointManager
from utils.validator import DataLeakageValidator, SystemValidator
from utils.rl_tracker import RLTracker

import warnings
import json

warnings.filterwarnings("ignore")


class ImmunePortfolioBacktester:
    def __init__(self, symbols, train_start, train_end, test_start, test_end):
        self.symbols = symbols
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end

        # 타임스탬프 기반 통합 출력 디렉토리 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(RESULTS_DIR, f"analysis_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)

        # RL Tracker 전용 디렉토리 생성
        self.rl_tracker_dir = os.path.join(self.output_dir, "rl_tracker")
        os.makedirs(self.rl_tracker_dir, exist_ok=True)


        # 체크포인트 관리자 초기화
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.checkpoint_manager = CheckpointManager(checkpoint_dir, save_interval=100)

        # 시스템 검증기 초기화
        self.validator = SystemValidator()

        # RLTracker 초기화
        self.rl_tracker = RLTracker(output_dir=self.rl_tracker_dir)

        # 고도화된 보상 계산기 초기화
        self.reward_calculator = RewardCalculator(
            lookback_window=20,
            transaction_cost_rate=0.001,
            target_volatility=0.15,
            target_max_drawdown=0.1,
        )

        # 데이터 로드
        data_filename = f"market_data_{'_'.join(symbols)}_{train_start}_{test_end}.pkl"
        self.data_path = os.path.join(DATA_DIR, data_filename)

        if os.path.exists(self.data_path):
            print(f"기존 데이터를 로드하고 있습니다: {data_filename}")
            with open(self.data_path, "rb") as f:
                self.data = pickle.load(f)
        else:
            print("포괄적 시장 데이터를 다운로드하고 있습니다...")
            raw_data = yf.download(
                symbols, start="2007-12-01", end="2025-01-01", progress=True
            )

            self.data = self._process_comprehensive_data(raw_data, symbols)

            print("데이터 전처리를 완료했습니다.")

            with open(self.data_path, "wb") as f:
                pickle.dump(self.data, f)
            print(f"포괄적 시장 데이터를 저장했습니다: {data_filename}")
            print(f"데이터 구조: {list(self.data.keys())}")

        # 데이터 분할
        self.train_data = self.data["prices"][train_start:train_end]
        self.test_data = self.data["prices"][test_start:test_end]
        self.train_features = self.data["features"][train_start:train_end]
        self.test_features = self.data["features"][test_start:test_end]

        self.train_data = self._clean_data(self.train_data)
        self.test_data = self._clean_data(self.test_data)

        # 데이터 리키지 검증
        DataLeakageValidator.validate_train_test_split(self.train_data, self.test_data)

        # 가중치 추적용 변수
        self.previous_weights = None
        self.current_weights = None

        # 커리큘럼 학습 관리자 초기화
        self.curriculum_manager = None


    def _process_comprehensive_data(self, raw_data, symbols):
        """포괄적인 시장 데이터 처리"""
        print("다중 지표 데이터를 처리하고 있습니다...")

        if len(symbols) == 1:
            if "Adj Close" in raw_data.columns:
                prices = raw_data["Adj Close"].to_frame(symbols[0])
            elif "Close" in raw_data.columns:
                prices = raw_data["Close"].to_frame(symbols[0])
            else:
                raise ValueError("가격 데이터를 찾을 수 없습니다.")
        else:
            try:
                prices = raw_data["Adj Close"]
            except KeyError:
                try:
                    prices = raw_data["Close"]
                    print("주의: 'Adj Close' 없음, 'Close' 사용")
                except KeyError:
                    price_data = {}
                    for symbol in symbols:
                        if ("Adj Close", symbol) in raw_data.columns:
                            price_data[symbol] = raw_data[("Adj Close", symbol)]
                        elif ("Close", symbol) in raw_data.columns:
                            price_data[symbol] = raw_data[("Close", symbol)]
                        else:
                            print(f"경고: {symbol} 가격 데이터를 찾을 수 없습니다.")
                            continue
                    if not price_data:
                        raise ValueError("사용 가능한 가격 데이터가 없습니다.")
                    prices = pd.DataFrame(price_data)

        features = self._calculate_technical_indicators(raw_data, symbols)

        prices = self._clean_data(prices)
        features = self._clean_data(features)

        return {"prices": prices, "features": features, "raw_data": raw_data}

    def _calculate_technical_indicators(self, raw_data, symbols):
        """기술적 지표 계산"""
        print("기술적 지표를 계산하고 있습니다...")

        features = {}

        for symbol in symbols:
            try:
                if len(symbols) == 1:
                    high = (
                        raw_data["High"]
                        if "High" in raw_data.columns
                        else raw_data["Close"]
                    )
                    low = (
                        raw_data["Low"]
                        if "Low" in raw_data.columns
                        else raw_data["Close"]
                    )
                    close = (
                        raw_data["Adj Close"]
                        if "Adj Close" in raw_data.columns
                        else raw_data["Close"]
                    )
                    volume = (
                        raw_data["Volume"]
                        if "Volume" in raw_data.columns
                        else pd.Series(1, index=raw_data.index)
                    )
                else:
                    high = (
                        raw_data["High"][symbol]
                        if "High" in raw_data.columns
                        else raw_data["Close"][symbol]
                    )
                    low = (
                        raw_data["Low"][symbol]
                        if "Low" in raw_data.columns
                        else raw_data["Close"][symbol]
                    )
                    close = (
                        raw_data["Adj Close"][symbol]
                        if "Adj Close" in raw_data.columns
                        else raw_data["Close"][symbol]
                    )
                    volume = (
                        raw_data["Volume"][symbol]
                        if "Volume" in raw_data.columns
                        else pd.Series(1, index=raw_data.index)
                    )

                symbol_features = pd.DataFrame(index=close.index)

                symbol_features[f"{symbol}_returns"] = close.pct_change()
                symbol_features[f"{symbol}_volatility"] = (
                    symbol_features[f"{symbol}_returns"].rolling(20).std()
                )
                symbol_features[f"{symbol}_sma_20"] = close.rolling(20).mean()
                symbol_features[f"{symbol}_sma_50"] = close.rolling(50).mean()
                symbol_features[f"{symbol}_price_sma20_ratio"] = (
                    close / symbol_features[f"{symbol}_sma_20"]
                )
                symbol_features[f"{symbol}_price_sma50_ratio"] = (
                    close / symbol_features[f"{symbol}_sma_50"]
                )

                symbol_features[f"{symbol}_rsi"] = self._calculate_rsi(close, 14)
                symbol_features[f"{symbol}_momentum"] = close / close.shift(10) - 1

                bb_upper, bb_lower = self._calculate_bollinger_bands(close, 20, 2)
                symbol_features[f"{symbol}_bb_position"] = (close - bb_lower) / (
                    bb_upper - bb_lower
                )

                symbol_features[f"{symbol}_volume_sma"] = volume.rolling(20).mean()
                symbol_features[f"{symbol}_volume_ratio"] = (
                    volume / symbol_features[f"{symbol}_volume_sma"]
                )

                symbol_features[f"{symbol}_high_low_ratio"] = (high - low) / close
                symbol_features[f"{symbol}_price_range"] = (high - low) / close.rolling(
                    20
                ).mean()

                features[symbol] = symbol_features

            except Exception as e:
                print(f"[경고] {symbol} 기술적 지표 계산 중 오류 발생: {e}")
                continue

        all_features = pd.concat(features.values(), axis=1)
        all_features = self._add_market_indicators(all_features, symbols)

        return all_features

    def _calculate_rsi(self, prices, period=14):
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """볼린저 밴드 계산"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band

    def _add_market_indicators(self, features, symbols):
        """시장 전체 지표 추가"""
        print("시장 전체 지표를 계산하고 있습니다...")

        try:
            return_cols = [col for col in features.columns if "_returns" in col]
            if return_cols:
                features["market_return"] = features[return_cols].mean(axis=1)
                features["market_volatility"] = features[return_cols].std(axis=1)
                corr_values = []
                for i in range(len(features)):
                    try:
                        window_data = features[return_cols].iloc[max(0, i - 19) : i + 1]
                        if len(window_data) >= 2:
                            corr_matrix = window_data.corr()
                            upper_tri = corr_matrix.where(
                                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                            )
                            corr_values.append(upper_tri.stack().mean())
                        else:
                            corr_values.append(0.0)
                    except:
                        corr_values.append(0.0)
                features["market_correlation"] = pd.Series(
                    corr_values, index=features.index
                )

            vol_cols = [col for col in features.columns if "_volatility" in col]
            if vol_cols:
                features["vix_proxy"] = (
                    features[vol_cols].mean(axis=1).rolling(10).std()
                )

            rsi_cols = [col for col in features.columns if "_rsi" in col]
            if rsi_cols:
                features["market_stress"] = features[rsi_cols].apply(
                    lambda x: (x < 30).sum() + (x > 70).sum(), axis=1
                )
            else:
                features["market_stress"] = 0

            market_cols = [
                "market_return",
                "market_volatility",
                "market_correlation",
                "vix_proxy",
                "market_stress",
            ]
            for col in market_cols:
                if col in features.columns:
                    features[col] = features[col].fillna(0)

        except Exception as e:
            print(f"[경고] 시장 전체 지표 계산 중 오류 발생: {e}")
            features["market_return"] = 0.0
            features["market_volatility"] = 0.1
            features["market_correlation"] = 0.5
            features["vix_proxy"] = 0.1
            features["market_stress"] = 0.0

        return features

    def _clean_data(self, data):
        """데이터 정리"""
        print("데이터를 전처리하고 있습니다...")

        if data.isnull().values.any():
            print("결측값을 발견했습니다. 전방향/후방향 채우기를 적용합니다.")
            data = data.fillna(method="ffill").fillna(method="bfill")

        if data.isnull().values.any():
            print("잔여 결측값을 0으로 채웁니다.")
            data = data.fillna(0)

        if np.isinf(data.values).any():
            print("무한대 값을 발견했습니다. 유한값으로 변환합니다.")
            data = data.replace([np.inf, -np.inf], 0)

        if data.isnull().values.any() or np.isinf(data.values).any():
            print("최종 데이터 정리를 진행합니다...")
            data = pd.DataFrame(
                np.nan_to_num(data.values, nan=0.0, posinf=0.0, neginf=0.0),
                index=data.index,
                columns=data.columns,
            )

        return data

    def calculate_metrics(self, returns, initial_capital=1e6):
        """성과 지표 계산"""
        cum_returns = (1 + returns).cumprod()
        final_value = initial_capital * cum_returns.iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital

        volatility = returns.std() * np.sqrt(252)
        max_drawdown = self.calculate_max_drawdown(returns)

        sharpe_ratio = (
            returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        )

        calmar_ratio = (
            returns.mean() * 252 / abs(max_drawdown) if max_drawdown != 0 else 0
        )

        return {
            "Total Return": total_return,
            "Volatility": volatility,
            "Max Drawdown": max_drawdown,
            "Sharpe Ratio": sharpe_ratio,
            "Calmar Ratio": calmar_ratio,
            "Final Value": final_value,
            "Initial Capital": initial_capital,
        }

    def calculate_max_drawdown(self, returns):
        """최대 낙폭 계산"""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()

    def backtest_single_run(
        self,
        seed=None,
        return_model=False,
        use_learning_bcells=True,
        use_hierarchical=True,
        use_curriculum=True,
        logging_level="full",
    ):
        """단일 백테스트 실행"""

        if seed is not None:
            np.random.seed(seed)
            if use_learning_bcells:
                torch.manual_seed(seed)

        immune_system = ImmunePortfolioSystem(
            n_assets=len(self.symbols),
            random_state=seed,
            use_learning_bcells=use_learning_bcells,
            use_hierarchical=use_hierarchical,
            logging_level=logging_level,
            output_dir=self.output_dir,
        )

        # 특성 데이터 전달
        immune_system.train_features = self.train_features
        immune_system.test_features = self.test_features

        # 커리큘럼 학습 초기화
        if use_curriculum and use_learning_bcells:
            self.curriculum_manager = CurriculumLearningManager(
                market_data=self.train_data, total_episodes=TOTAL_EPISODES, episode_length=EPISODE_LENGTH
            )
            print("커리큘럼 학습이 활성화되었습니다.")

        try:
            # 사전 훈련
            if use_learning_bcells:
                immune_system.pretrain_bcells(self.train_data, episodes=PRETRAIN_EPISODES)

            # 커리큘럼 기반 적응형 학습
            if use_curriculum and self.curriculum_manager:
                print("커리큘럼 기반 적응형 학습을 진행하고 있습니다...")
                self._curriculum_training(immune_system)
            else:
                print("기존 방식 적응형 학습을 진행하고 있습니다...")
                self._traditional_training(immune_system)

            # 강화학습 시각화 생성
            if use_learning_bcells:
                print("강화학습 추적 시각화를 생성하고 있습니다...")

                rl_plot_path = os.path.join(
                    self.rl_tracker_dir, "training_analysis.png"
                )
                self.rl_tracker.create_comprehensive_plot(save_path=rl_plot_path)

                data_path = self.rl_tracker.save_training_data(
                    os.path.join(self.rl_tracker_dir, "training_data.csv")
                )

                print(f"강화학습 분석 완료:")
                print(f"  시각화: {rl_plot_path}")
                print(f"  데이터: {data_path}")

            # 테스트 단계
            print("테스트 데이터 기반 성능 평가를 진행하고 있습니다...")
            test_portfolio_returns = self._run_test_phase(immune_system, logging_level)

            # 검증 요약 출력
            validation_summary = self.validator.get_validation_summary()
            print(
                f"검증 요약: 성공률 {validation_summary['success_rate']:.1%} ({validation_summary['successful']}/{validation_summary['total']})"
            )

        except Exception as e:
            print(f"백테스트 실행 중 오류 발생: {e}")
            self._save_emergency_checkpoint(immune_system, str(e))
            raise

        if return_model:
            return (
                pd.Series(
                    test_portfolio_returns,
                    index=self.test_data.pct_change().dropna().index,
                ),
                immune_system,
            )
        else:
            return pd.Series(
                test_portfolio_returns, index=self.test_data.pct_change().dropna().index
            )

    def _curriculum_training(self, immune_system):
        """커리큘럼 기반 훈련"""

        # 가중치 초기화
        base_weights = np.ones(len(self.symbols)) / len(self.symbols)
        self.previous_weights = base_weights.copy()
        self.current_weights = base_weights.copy()

        episode_rewards = []

        with tqdm(
            total=self.curriculum_manager.scheduler.total_episodes, desc="커리큘럼 학습"
        ) as pbar:
            while not self.curriculum_manager.is_curriculum_complete():
                # 커리큘럼에 맞는 에피소드 데이터 획득
                episode_data, episode_features, curriculum_config = (
                    self.curriculum_manager.get_next_training_episode()
                )

                # 데이터 검증
                validation_result = self.validator.validate_episode_data(
                    episode_data, episode_features
                )
                if (
                    not validation_result["data_valid"]
                    or not validation_result["features_valid"]
                ):
                    print(f"에피소드 데이터 검증 실패: {validation_result['issues']}")
                    continue

                # 에피소드 실행
                episode_reward, episode_return = self._run_training_episode(
                    immune_system, episode_data, curriculum_config
                )

                episode_rewards.append(episode_reward)

                # 성과 메트릭 계산
                episode_returns = episode_data.pct_change().dropna()
                if len(episode_returns) > 5:
                    sharpe_ratio = (
                        episode_returns.mean().mean()
                        / episode_returns.std().mean()
                        * np.sqrt(252)
                        if episode_returns.std().mean() > 0
                        else 0
                    )
                    max_drawdown = self.calculate_max_drawdown(
                        episode_returns.mean(axis=1)
                    )
                else:
                    sharpe_ratio = 0
                    max_drawdown = 0

                # 커리큘럼 결과 기록
                self.curriculum_manager.record_episode_result(
                    reward=episode_reward,
                    portfolio_return=episode_return,
                    sharpe_ratio=sharpe_ratio,
                    max_drawdown=max_drawdown,
                )

                # RLTracker에 데이터 기록
                current_episode = self.curriculum_manager.scheduler.current_episode
                current_level = self.curriculum_manager.scheduler.current_level

                # B-Cell 개별 성과 수집
                bcell_rewards = {}
                learning_rates = {}
                losses = {}
                epsilon = None

                if immune_system.use_learning_bcells:
                    for i, bcell in enumerate(immune_system.bcells):
                        if (
                            hasattr(bcell, "specialist_performance")
                            and bcell.specialist_performance
                        ):
                            bcell_rewards[f"bcell_{i}_{bcell.risk_type}"] = np.mean(
                                list(bcell.specialist_performance)[-3:]
                            )

                        if hasattr(bcell, "actor_optimizer"):
                            learning_rates[f"actor_{i}"] = (
                                bcell.actor_optimizer.param_groups[0]["lr"]
                            )
                            learning_rates[f"critic_{i}"] = (
                                bcell.critic_optimizer.param_groups[0]["lr"]
                            )

                        if hasattr(bcell, "last_actor_loss"):
                            losses[f"actor_{i}"] = bcell.last_actor_loss
                        if hasattr(bcell, "last_critic_loss"):
                            losses[f"critic_{i}"] = bcell.last_critic_loss

                        if hasattr(bcell, "epsilon"):
                            epsilon = bcell.epsilon

                # 메타 컨트롤러 성과
                meta_reward = None
                if (
                    hasattr(immune_system, "hierarchical_controller")
                    and immune_system.hierarchical_controller
                ):
                    if immune_system.hierarchical_controller.meta_level_rewards:
                        meta_reward = (
                            immune_system.hierarchical_controller.meta_level_rewards[-1]
                        )

                # RLTracker 로깅
                self.rl_tracker.log_episode(
                    episode=current_episode,
                    reward=episode_reward,
                    portfolio_return=episode_return,
                    learning_rates=learning_rates,
                    losses=losses,
                    epsilon=epsilon,
                    curriculum_level=current_level,
                    bcell_rewards=bcell_rewards,
                    meta_reward=meta_reward,
                )

                # 체크포인트 저장
                if self.checkpoint_manager.should_save_checkpoint():
                    episode_info = {
                        "reward": episode_reward,
                        "return": episode_return,
                        "config": curriculum_config,
                        "progress": self.curriculum_manager.get_curriculum_progress(),
                    }
                    self.checkpoint_manager.save_checkpoint(
                        immune_system, self.curriculum_manager, episode_info
                    )

                # 진행률 업데이트
                progress = self.curriculum_manager.get_curriculum_progress()
                pbar.set_postfix(
                    {
                        "Level": progress["current_level"],
                        "Episode": progress["current_episode"],
                        "Reward": f"{episode_reward:.3f}",
                        "Config": curriculum_config["name"],
                    }
                )
                pbar.update(1)

                # B-세포 학습
                if immune_system.use_learning_bcells:
                    for bcell in immune_system.bcells:
                        if hasattr(bcell, "end_episode"):
                            bcell.end_episode()

                # 계층적 제어기 학습
                if (
                    hasattr(immune_system, "hierarchical_controller")
                    and immune_system.hierarchical_controller
                ):
                    immune_system.update_hierarchical_learning(episode_reward)

        # 커리큘럼 요약 저장
        curriculum_summary = self.curriculum_manager.get_training_summary()
        summary_path = os.path.join(self.output_dir, "curriculum_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(curriculum_summary, f, ensure_ascii=False, indent=2, default=str)

        print(f"커리큘럼 학습이 완료되었습니다. 요약: {summary_path}")

    def _run_training_episode(self, immune_system, episode_data, curriculum_config):
        """올바른 State Transition을 보장하는 에피소드 실행 - 핵심 재구축"""
        
        episode_returns = episode_data.pct_change().dropna()
        if len(episode_returns) < 2:
            return 0.0, 0.0
            
        transitions = []
        portfolio_returns = []
        
        # State Transition 올바르게 구축
        for i in range(len(episode_returns) - 1):
            try:
                # 현재 상태
                current_data = episode_data.iloc[:i+1]
                current_state = immune_system.extract_market_features(current_data)
                if current_state is None or len(current_state) == 0:
                    current_state = np.zeros(12, dtype=np.float32)
                
                # 다음 상태  
                next_data = episode_data.iloc[:i+2]
                next_state = immune_system.extract_market_features(next_data)
                if next_state is None or len(next_state) == 0:
                    next_state = np.zeros(12, dtype=np.float32)
                
                # 행동 실행
                weights, response_type, bcell_decisions = immune_system.immune_response(
                    current_state, training=True
                )
                
                # 가중치 검증
                if weights is None or len(weights) != len(episode_returns.columns):
                    weights = np.ones(len(episode_returns.columns)) / len(episode_returns.columns)
                
                # 포트폴리오 수익률 계산
                portfolio_return = np.sum(weights * episode_returns.iloc[i])
                portfolio_return = np.clip(portfolio_return, -0.5, 0.5)
                portfolio_returns.append(portfolio_return)
                
                # 보상 계산 (단일 클리핑만)
                try:
                    reward_details = self.reward_calculator.calculate_comprehensive_reward(
                        current_return=portfolio_return,
                        previous_weights=getattr(self, 'previous_weights', None),
                        current_weights=weights,
                        market_features=current_state,
                        crisis_level=immune_system.crisis_level,
                    )
                    total_reward = reward_details["total_reward"]
                except Exception as e:
                    total_reward = 0.0
                
                # 단일 클리핑만 적용 (삼중 클리핑 제거)
                final_reward = np.clip(total_reward, *REWARD_CLIPPING_RANGE)
                
                # 에피소드 종료 플래그
                done = (i == len(episode_returns) - 2)
                
                # Transition 저장 (완전한 (s,a,r,s',done) 튜플)
                transitions.append({
                    'state': current_state.copy(),
                    'action': weights.copy(),
                    'reward': final_reward,
                    'next_state': next_state.copy(),
                    'done': done
                })
                
                # 가중치 업데이트
                self.previous_weights = getattr(self, 'current_weights', weights.copy())
                self.current_weights = weights.copy()
                
            except Exception as e:
                print(f"[경고] 스텝 {i} 실행 오류: {e}")
                continue
        
        # 모든 B-Cell에 올바른 경험 전달
        if immune_system.use_learning_bcells and transitions:
            for bcell in immune_system.bcells:
                try:
                    for transition in transitions:
                        bcell.add_experience(
                            transition['state'],
                            transition['action'], 
                            transition['reward'],
                            transition['next_state'],
                            transition['done']
                        )
                    
                    # 배치 학습 (충분한 경험이 쌓였을 때만)
                    if len(bcell.experience_buffer) >= DEFAULT_BATCH_SIZE:
                        loss = bcell.learn_from_batch()
                        if loss is not None:
                            self.rl_tracker.log_loss(bcell.risk_type, loss)
                            
                except Exception as e:
                    print(f"[경고] B-세포 {bcell.cell_id} 학습 오류: {e}")
        
        # 메모리 시스템 업데이트
        if immune_system.crisis_level > 0.15 and transitions:
            try:
                last_transition = transitions[-1]
                memory_reward = np.clip(last_transition['reward'], -1.0, 1.0)
                immune_system.update_memory(
                    last_transition['state'], 
                    last_transition['action'], 
                    memory_reward
                )
            except Exception as e:
                print(f"[경고] 메모리 업데이트 오류: {e}")
        
        # 결과 계산
        if transitions:
            avg_reward = np.mean([t['reward'] for t in transitions])
            avg_return = np.mean(portfolio_returns) if portfolio_returns else 0.0
        else:
            avg_reward = 0.0
            avg_return = 0.0
        
        # NaN/Inf 검증
        if np.isnan(avg_reward) or np.isinf(avg_reward):
            avg_reward = 0.0
        if np.isnan(avg_return) or np.isinf(avg_return):
            avg_return = 0.0
            
        return float(avg_reward), float(avg_return)

    def _traditional_training(self, immune_system):
        """기존 방식 훈련"""

        train_returns = self.train_data.pct_change().dropna()
        portfolio_values = [1.0]

        # 가중치 초기화
        base_weights = np.ones(len(self.symbols)) / len(self.symbols)
        self.previous_weights = base_weights.copy()
        self.current_weights = base_weights.copy()

        for i in tqdm(range(len(train_returns)), desc="적응형 학습"):
            current_data = self.train_data.iloc[: i + 1]
            market_features = immune_system.extract_market_features(current_data)

            # 마지막 특성 저장
            immune_system.last_market_features = market_features

            weights, response_type, bcell_decisions = immune_system.immune_response(
                market_features, training=True
            )

            # 면역 반응 검증
            validation_result = self.validator.validate_immune_response(
                weights, response_type, bcell_decisions
            )
            if not validation_result["weights_valid"]:
                print(f"면역 반응 검증 실패: {validation_result['issues']}")
                weights = np.ones(len(self.symbols)) / len(self.symbols)

            portfolio_return = np.sum(weights * train_returns.iloc[i])
            portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))

            if hasattr(immune_system, "analyzer") and immune_system.enable_logging:
                current_date = train_returns.index[i]
                immune_system.analyzer.log_decision(
                    date=current_date,
                    market_features=market_features,
                    tcell_analysis=getattr(
                        immune_system,
                        "detailed_tcell_analysis",
                        {"crisis_level": immune_system.crisis_level},
                    ),
                    bcell_decisions=bcell_decisions,
                    final_weights=weights,
                    portfolio_return=portfolio_return,
                    crisis_level=immune_system.crisis_level,
                )

            # 고도화된 보상 계산
            if len(portfolio_values) > 20:
                reward_details = self.reward_calculator.calculate_comprehensive_reward(
                    current_return=portfolio_return,
                    previous_weights=self.previous_weights,
                    current_weights=weights,
                    market_features=market_features,
                    crisis_level=immune_system.crisis_level,
                )

                total_reward = reward_details["total_reward"]

                # RLTracker 로깅
                if i % 10 == 0:  # 일정 간격으로만 로깅
                    learning_rates = {}
                    losses = {}
                    epsilon = None
                    bcell_rewards = {}

                    if immune_system.use_learning_bcells:
                        for j, bcell in enumerate(immune_system.bcells):
                            if hasattr(bcell, "actor_optimizer"):
                                learning_rates[f"actor_{j}"] = (
                                    bcell.actor_optimizer.param_groups[0]["lr"]
                                )
                                learning_rates[f"critic_{j}"] = (
                                    bcell.critic_optimizer.param_groups[0]["lr"]
                                )

                            if (
                                hasattr(bcell, "specialist_performance")
                                and bcell.specialist_performance
                            ):
                                bcell_rewards[f"bcell_{j}_{bcell.risk_type}"] = np.mean(
                                    list(bcell.specialist_performance)[-3:]
                                )

                            if hasattr(bcell, "epsilon"):
                                epsilon = bcell.epsilon

                    self.rl_tracker.log_episode(
                        episode=i,
                        reward=total_reward,
                        portfolio_return=portfolio_return,
                        learning_rates=learning_rates,
                        losses=losses,
                        epsilon=epsilon,
                        bcell_rewards=bcell_rewards,
                    )

                # B-세포 학습
                if immune_system.use_learning_bcells:
                    for bcell in immune_system.bcells:
                        if hasattr(bcell, "last_strategy"):
                            is_specialist_today = bcell.is_my_specialty_situation(
                                market_features, immune_system.crisis_level
                            )

                            if is_specialist_today:
                                specialist_reward = total_reward * 2.0
                            else:
                                specialist_reward = total_reward * 0.8

                            final_reward = np.clip(specialist_reward, -2, 2)

                            bcell.add_experience(
                                market_features,
                                immune_system.crisis_level,
                                bcell.last_strategy.numpy(),
                                final_reward,
                            )

                            if i % 20 == 0:
                                bcell.learn_from_specialized_experience()

                # 기억 세포 업데이트
                if immune_system.crisis_level > 0.15:
                    immune_system.update_memory(
                        market_features, weights, np.clip(total_reward, -1, 1)
                    )

                # 계층적 제어기 학습
                if (
                    hasattr(immune_system, "hierarchical_controller")
                    and immune_system.hierarchical_controller
                ):
                    immune_system.update_hierarchical_learning(total_reward)

            # 가중치 업데이트
            self.previous_weights = self.current_weights.copy()
            self.current_weights = weights.copy()

        # 에피소드 종료
        if immune_system.use_learning_bcells:
            for bcell in immune_system.bcells:
                if hasattr(bcell, "end_episode"):
                    bcell.end_episode()

    def _run_test_phase(self, immune_system, logging_level):
        """테스트 단계 실행"""

        test_returns = self.test_data.pct_change().dropna()
        test_portfolio_returns = []

        for i in tqdm(range(len(test_returns)), desc="성능 평가"):
            current_data = self.test_data.iloc[: i + 1]
            market_features = immune_system.extract_market_features(current_data)

            # 마지막 특성 저장
            immune_system.last_market_features = market_features

            weights, response_type, bcell_decisions = immune_system.immune_response(
                market_features, training=False
            )

            # 면역 반응 검증
            validation_result = self.validator.validate_immune_response(
                weights, response_type, bcell_decisions
            )
            if not validation_result["weights_valid"]:
                weights = np.ones(len(self.symbols)) / len(self.symbols)

            portfolio_return = np.sum(weights * test_returns.iloc[i])
            test_portfolio_returns.append(portfolio_return)

            should_log = False
            if hasattr(immune_system, "analyzer") and immune_system.enable_logging:
                if immune_system.logging_level == "full":
                    should_log = True
                elif immune_system.logging_level == "sample":
                    should_log = i % 10 == 0
                elif immune_system.logging_level == "minimal":
                    should_log = i % 50 == 0

                if should_log:
                    current_date = test_returns.index[i]
                    immune_system.analyzer.log_decision(
                        date=current_date,
                        market_features=market_features,
                        tcell_analysis=getattr(
                            immune_system,
                            "detailed_tcell_analysis",
                            {"crisis_level": immune_system.crisis_level},
                        ),
                        bcell_decisions=bcell_decisions,
                        final_weights=weights,
                        portfolio_return=portfolio_return,
                        crisis_level=immune_system.crisis_level,
                    )

            # 가중치 업데이트
            self.previous_weights = self.current_weights.copy()
            self.current_weights = weights.copy()

        return test_portfolio_returns

    def _save_emergency_checkpoint(self, immune_system, error_message):
        """응급 체크포인트 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            emergency_path = os.path.join(
                self.output_dir, f"emergency_checkpoint_{timestamp}.pkl"
            )

            emergency_data = {
                "timestamp": timestamp,
                "error_message": error_message,
                "immune_system_state": self.checkpoint_manager._extract_immune_system_state(
                    immune_system
                ),
                "validation_summary": self.validator.get_validation_summary(),
            }

            with open(emergency_path, "wb") as f:
                pickle.dump(emergency_data, f)

            print(f"응급 체크포인트 저장: {emergency_path}")

        except Exception as e:
            print(f"응급 체크포인트 저장 실패: {e}")

    def analyze_bcell_expertise(self):
        """B-세포 전문성 분석"""

        if (
            not hasattr(self, "immune_system")
            or not self.immune_system.use_learning_bcells
        ):
            return {"error": "Learning-based system is not available."}

        print("B-세포 전문화 시스템을 분석하고 있습니다...")

        total_specialist_exp = 0
        total_general_exp = 0
        bcell_analysis = []

        for bcell in self.immune_system.bcells:
            if hasattr(bcell, "get_expertise_metrics"):
                metrics = bcell.get_expertise_metrics()
                bcell_analysis.append(metrics)

                total_specialist_exp += metrics["specialist_experiences"]
                total_general_exp += metrics["general_experiences"]

        overall_specialization = total_specialist_exp / max(
            1, total_specialist_exp + total_general_exp
        )

        analysis_result = {
            "bcell_metrics": bcell_analysis,
            "overall_specialization_ratio": overall_specialization,
            "total_specialist_experiences": total_specialist_exp,
            "total_general_experiences": total_general_exp,
        }

        return analysis_result

    def save_comprehensive_analysis(
        self,
        start_date: str,
        end_date: str,
        filename: str = None,
        output_dir: str = None,
    ):
        """통합 분석 결과 저장"""

        if output_dir is None:
            output_dir = self.output_dir

        if filename is None:
            filename = f"bipd_comprehensive_{start_date}_{end_date}"

        decision_analysis = {}
        if hasattr(self, "immune_system") and hasattr(self.immune_system, "analyzer"):
            try:
                decision_analysis = (
                    self.immune_system.analyzer.generate_analysis_report(
                        start_date, end_date
                    )
                )
            except Exception as e:
                print(f"의사결정 분석 오류: {e}")
                decision_analysis = {"error": f"의사결정 분석 실패: {e}"}
        else:
            decision_analysis = {"error": "분석 시스템을 사용할 수 없습니다."}

        expertise_analysis = self.analyze_bcell_expertise()

        # 계층적 메트릭 추가
        hierarchical_metrics = {}
        if hasattr(self, "immune_system") and hasattr(
            self.immune_system, "get_hierarchical_metrics"
        ):
            hierarchical_metrics = self.immune_system.get_hierarchical_metrics()

        # 커리큘럼 메트릭 추가
        curriculum_metrics = {}
        if self.curriculum_manager:
            curriculum_metrics = self.curriculum_manager.get_training_summary()

        comprehensive_data = {
            "metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_period": {"start": start_date, "end": end_date},
                "system_type": (
                    "Advanced Learning-based"
                    if (
                        hasattr(self, "immune_system")
                        and self.immune_system.use_learning_bcells
                    )
                    else "Rule-based"
                ),
                "features_enabled": {
                    "hierarchical_control": bool(
                        hasattr(self, "immune_system")
                        and hasattr(self.immune_system, "hierarchical_controller")
                        and self.immune_system.hierarchical_controller
                    ),
                    "curriculum_learning": bool(self.curriculum_manager),
                    "attention_mechanism": True,
                    "memory_augmentation": True,
                    "advanced_rewards": True,
                },
            },
            "decision_analysis": decision_analysis,
            "expertise_analysis": expertise_analysis,
            "hierarchical_metrics": hierarchical_metrics,
            "curriculum_metrics": curriculum_metrics,
        }

        json_path = os.path.join(output_dir, f"{filename}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(comprehensive_data, f, ensure_ascii=False, indent=2, default=str)

        md_content = self._generate_comprehensive_markdown(comprehensive_data)
        md_path = os.path.join(output_dir, f"{filename}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"통합 분석 결과를 저장했습니다:")
        print(f"  디렉토리: {output_dir}")
        print(f"  JSON: {os.path.basename(json_path)}")
        print(f"  Markdown: {os.path.basename(md_path)}")

        return json_path, md_path

    def _generate_comprehensive_markdown(self, comprehensive_data: Dict) -> str:
        """통합 분석 마크다운 생성"""

        metadata = comprehensive_data["metadata"]
        decision_data = comprehensive_data["decision_analysis"]
        expertise_data = comprehensive_data["expertise_analysis"]
        hierarchical_data = comprehensive_data["hierarchical_metrics"]
        curriculum_data = comprehensive_data["curriculum_metrics"]

        md_content = f"""# BIPD 시스템 통합 분석 보고서

## 분석 메타데이터
- 분석 시간: {metadata['analysis_timestamp']}
- 시스템 유형: {metadata['system_type']}
- 분석 기간: {metadata['analysis_period']['start']} ~ {metadata['analysis_period']['end']}

### 활성화된 기능
- 계층적 제어: {metadata['features_enabled']['hierarchical_control']}
- 커리큘럼 학습: {metadata['features_enabled']['curriculum_learning']}
- 어텐션 메커니즘: {metadata['features_enabled']['attention_mechanism']}
- 기억 기반 학습: {metadata['features_enabled']['memory_augmentation']}
- 고도화된 보상: {metadata['features_enabled']['advanced_rewards']}

---

"""

        # 커리큘럼 학습 분석
        if curriculum_data and "error" not in curriculum_data:
            md_content += """## 커리큘럼 학습 분석

"""
            if "current_level" in curriculum_data:
                md_content += f"- 최종 레벨: {curriculum_data['current_level']}\n"
                md_content += f"- 총 에피소드: {curriculum_data['total_episodes']}\n"
                md_content += (
                    f"- 레벨 전환 횟수: {curriculum_data['level_transitions']}\n\n"
                )

        # 계층적 제어 분석
        if hierarchical_data and "hierarchical_system" not in hierarchical_data:
            md_content += """## 계층적 제어 시스템 분석

"""
            if "total_expert_selections" in hierarchical_data:
                md_content += f"- 총 전문가 선택: {hierarchical_data['total_expert_selections']}\n"
                md_content += f"- 선택 다양성: {hierarchical_data.get('selection_diversity', 0):.3f}\n"
                md_content += f"- 전환 엔트로피: {hierarchical_data.get('expert_transition_entropy', 0):.3f}\n\n"

        # 기존 분석들...
        if "error" in decision_data:
            md_content += (
                f"## 의사결정 분석\n\n**오류:** {decision_data['error']}\n\n---\n\n"
            )
        else:
            period = decision_data["period"]
            stats = decision_data["basic_stats"]
            risk_dist = decision_data["risk_distribution"]
            efficiency = decision_data["system_efficiency"]

            md_content += f"""## 의사결정 분석

### 분석 기간
- 시작일: {period['start']}
- 종료일: {period['end']}

### 기본 통계
- 총 거래일: {stats['total_days']}일
- 위기 감지일: {stats['crisis_days']}일 ({stats['crisis_ratio']:.1%})
- 기억 세포 활성화: {stats['memory_activations']}일 ({stats['memory_activation_ratio']:.1%})
- 평균 일수익률: {stats['avg_daily_return']:+.3%}

### 위험 유형별 분포
"""

            for risk, count in sorted(
                risk_dist.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = count / stats["total_days"] * 100
                md_content += f"- {risk}: {count}일 ({percentage:.1f}%)\n"

            md_content += f"""
### 시스템 효율성
- 위기 대응률: {efficiency['crisis_response_rate']:.1%}
- 학습 활성화율: {efficiency['learning_activation_rate']:.1%}
- 시스템 안정성: {efficiency['system_stability']}

---

"""

        if "error" in expertise_data:
            md_content += f"## 전문성 분석\n\n**오류:** {expertise_data['error']}\n\n"
        else:
            md_content += "## 전문성 분석\n\n"

            for bcell_metrics in expertise_data["bcell_metrics"]:
                md_content += f"### {bcell_metrics['risk_type'].upper()} 전문가\n"
                md_content += (
                    f"- 전문성 강도: {bcell_metrics['specialization_strength']:.3f}\n"
                )
                md_content += (
                    f"- 전문 경험: {bcell_metrics['specialist_experiences']}건\n"
                )
                md_content += f"- 일반 경험: {bcell_metrics['general_experiences']}건\n"
                md_content += (
                    f"- 전문화 비율: {bcell_metrics['specialization_ratio']:.1%}\n"
                )
                md_content += f"- 전문 분야 평균 보상: {bcell_metrics['specialist_avg_reward']:+.3f}\n"
                md_content += f"- 일반 분야 평균 보상: {bcell_metrics['general_avg_reward']:+.3f}\n"
                md_content += (
                    f"- 전문성 우위: {bcell_metrics['expertise_advantage']:+.3f}\n\n"
                )

            md_content += "### 전체 시스템 현황\n"
            md_content += f"- 전체 전문화 비율: {expertise_data['overall_specialization_ratio']:.1%}\n"
            md_content += (
                f"- 총 전문 경험: {expertise_data['total_specialist_experiences']}건\n"
            )
            md_content += (
                f"- 총 일반 경험: {expertise_data['total_general_experiences']}건\n"
            )

        return md_content

    def save_analysis_results(
        self, start_date: str, end_date: str, filename: str = None
    ):
        """분석 결과 저장"""

        if not hasattr(self, "immune_system") or not hasattr(
            self.immune_system, "analyzer"
        ):
            print("분석 시스템을 사용할 수 없습니다.")
            return None, None, None

        try:
            json_path, md_path = self.immune_system.analyzer.save_analysis_to_file(
                start_date, end_date, filename, output_dir=self.output_dir
            )

            analysis_report = self.immune_system.analyzer.generate_analysis_report(
                start_date, end_date
            )

            dashboard_paths = generate_dashboard(
                analysis_report,
                output_dir=self.output_dir,
            )

            immune_viz = create_visualizations(
                self,
                start_date,
                end_date,
                output_dir=self.output_dir,
            )

            print(f"분석 결과를 저장했습니다:")
            print(f"  JSON: {json_path}")
            print(f"  Markdown: {md_path}")
            print(f"  HTML Dashboard: {dashboard_paths['html_dashboard']}")
            print(
                f"\nHTML 대시보드에서 T-Cell/B-Cell 판단 근거를 직관적으로 확인할 수 있습니다!"
            )
            print(f"면역 시스템 반응 패턴 시각화는 기존 연구와의 차별화를 강조합니다!")

            return json_path, md_path, dashboard_paths["html_dashboard"]

        except Exception as e:
            print(f"분석 결과 저장 오류: {e}")
            return None, None, None

    def save_expertise_analysis(self, filename: str = None):
        """전문성 분석 결과 저장"""

        expertise_data = self.analyze_bcell_expertise()

        if "error" in expertise_data:
            print(expertise_data["error"])
            return None

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"expertise_analysis_{timestamp}"

        json_path = os.path.join(self.output_dir, f"{filename}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(expertise_data, f, ensure_ascii=False, indent=2)

        md_content = self._generate_expertise_markdown(expertise_data)
        md_path = os.path.join(self.output_dir, f"{filename}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"전문성 분석을 저장했습니다:")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {md_path}")

        return json_path, md_path

    def _generate_expertise_markdown(self, expertise_data: Dict) -> str:
        """전문성 분석 마크다운 생성"""

        md_content = "# B-세포 전문성 분석 보고서\n\n"

        for bcell_metrics in expertise_data["bcell_metrics"]:
            md_content += f"## {bcell_metrics['risk_type'].upper()} 전문가\n"
            md_content += (
                f"- 전문성 강도: {bcell_metrics['specialization_strength']:.3f}\n"
            )
            md_content += f"- 전문 경험: {bcell_metrics['specialist_experiences']}건\n"
            md_content += f"- 일반 경험: {bcell_metrics['general_experiences']}건\n"
            md_content += (
                f"- 전문화 비율: {bcell_metrics['specialization_ratio']:.1%}\n"
            )
            md_content += f"- 전문 분야 평균 보상: {bcell_metrics['specialist_avg_reward']:+.3f}\n"
            md_content += (
                f"- 일반 분야 평균 보상: {bcell_metrics['general_avg_reward']:+.3f}\n"
            )
            md_content += (
                f"- 전문성 우위: {bcell_metrics['expertise_advantage']:+.3f}\n\n"
            )

        md_content += "## 전체 시스템 현황\n"
        md_content += f"- 전체 전문화 비율: {expertise_data['overall_specialization_ratio']:.1%}\n"
        md_content += (
            f"- 총 전문 경험: {expertise_data['total_specialist_experiences']}건\n"
        )
        md_content += (
            f"- 총 일반 경험: {expertise_data['total_general_experiences']}건\n"
        )

        return md_content

    def save_model(self, immune_system, filename=None, output_dir=None):
        """모델 저장"""
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "models")
            os.makedirs(output_dir, exist_ok=True)

        if filename is None:
            if immune_system.use_learning_bcells:
                filename = "immune_system"
            else:
                filename = "legacy_immune_system"

        if immune_system.use_learning_bcells:
            model_dir = os.path.join(output_dir, filename)
            os.makedirs(model_dir, exist_ok=True)

            for i, bcell in enumerate(immune_system.bcells):
                if hasattr(bcell, "actor_network"):
                    network_path = os.path.join(
                        model_dir, f"bcell_{i}_{bcell.risk_type}.pth"
                    )
                    torch.save(bcell.actor_network.state_dict(), network_path)

            system_state = {
                "n_assets": immune_system.n_assets,
                "base_weights": immune_system.base_weights,
                "memory_cell": immune_system.memory_cell,
                "tcells": immune_system.tcells,
                "use_learning_bcells": True,
            }
            state_path = os.path.join(model_dir, "system_state.pkl")
            with open(state_path, "wb") as f:
                pickle.dump(system_state, f)

            print(f"Learning-based 모델을 저장했습니다: {model_dir}")
            return model_dir
        else:
            model_path = os.path.join(output_dir, f"{filename}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(immune_system, f)
            print(f"규칙 기반 모델을 저장했습니다: {model_path}")
            return model_path

    def save_results(self, metrics_df, filename=None, output_dir=None):
        """결과 저장"""
        if output_dir is None:
            output_dir = self.output_dir

        if filename is None:
            filename = "bipd_performance_metrics"

        csv_path = os.path.join(output_dir, f"{filename}.csv")
        metrics_df.to_csv(csv_path, index=False)

        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        metrics_df.boxplot(column=["Total Return"], ax=plt.gca())
        plt.title("Total Return Distribution")

        plt.subplot(2, 3, 2)
        metrics_df.boxplot(column=["Sharpe Ratio"], ax=plt.gca())
        plt.title("Sharpe Ratio Distribution")

        plt.subplot(2, 3, 3)
        metrics_df.boxplot(column=["Max Drawdown"], ax=plt.gca())
        plt.title("Max Drawdown Distribution")

        plt.subplot(2, 2, 3)
        correlation = metrics_df.corr()
        plt.imshow(correlation, cmap="coolwarm", aspect="auto")
        plt.colorbar()
        plt.title("Metrics Correlation")
        plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45)
        plt.yticks(range(len(correlation.columns)), correlation.columns)

        plt.subplot(2, 2, 4)
        plt.axis("off")
        summary_text = f"""
BIPD Backtest Results Summary

Total Return: {metrics_df['Total Return'].mean():.2%}
Volatility: {metrics_df['Volatility'].mean():.3f}
Max Drawdown: {metrics_df['Max Drawdown'].mean():.2%}
Sharpe Ratio: {metrics_df['Sharpe Ratio'].mean():.2f}
Calmar Ratio: {metrics_df['Calmar Ratio'].mean():.2f}
Initial Capital: {metrics_df['Initial Capital'].iloc[0]:,.0f}
Final Capital: {metrics_df['Final Value'].mean():,.0f}
        """
        plt.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment="center")

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"백테스트 결과를 저장했습니다:")
        print(f"  디렉토리: {output_dir}")
        print(f"  CSV: {os.path.basename(csv_path)}")
        print(f"  차트: {os.path.basename(plot_path)}")
        return csv_path, plot_path

    def run_multiple_backtests(
        self,
        n_runs=10,
        save_results=True,
        use_learning_bcells=True,
        use_hierarchical=True,
        use_curriculum=True,
        logging_level="sample",
        base_seed=None,
    ):
        """다중 백테스트 실행"""
        all_metrics = []
        best_immune_system = None
        best_sharpe = -np.inf
        rl_trackers = []

        print(f"\n=== BIPD 시스템 다중 백테스트 ({n_runs}회) 실행 ===")

        feature_status = []
        if use_learning_bcells:
            feature_status.append("적응형 신경망")
        if use_hierarchical:
            feature_status.append("계층적 제어")
        if use_curriculum:
            feature_status.append("커리큘럼 학습")

        if feature_status:
            print(f"시스템 유형: {' + '.join(feature_status)} 기반 BIPD 모델")
        else:
            print("시스템 유형: 규칙 기반 레거시 BIPD 모델")

        if base_seed is None:
            import time

            base_seed = int(time.time()) % 10000

        print(f"[설정] 기본 시드: {base_seed}")

        for run in range(n_runs):
            run_seed = base_seed + run * 1000
            print(f"\n{run + 1}/{n_runs}번째 실행 (시드: {run_seed})")

            # 각 실행별 RLTracker 디렉토리 생성
            run_rl_dir = os.path.join(self.rl_tracker_dir, f"run_{run+1}")
            os.makedirs(run_rl_dir, exist_ok=True)

            # 새로운 RLTracker 생성
            self.rl_tracker = RLTracker(output_dir=run_rl_dir)

            portfolio_returns, immune_system = self.backtest_single_run(
                seed=run_seed,
                return_model=True,
                use_learning_bcells=use_learning_bcells,
                use_hierarchical=use_hierarchical,
                use_curriculum=use_curriculum,
                logging_level=logging_level,
            )
            metrics = self.calculate_metrics(portfolio_returns)
            all_metrics.append(metrics)

            if metrics["Sharpe Ratio"] > best_sharpe:
                best_sharpe = metrics["Sharpe Ratio"]
                best_immune_system = immune_system

            rl_trackers.append(self.rl_tracker)

        metrics_df = pd.DataFrame(all_metrics)

        system_features = []
        if use_learning_bcells:
            system_features.append("Learning")
        if use_hierarchical:
            system_features.append("Hierarchical")
        if use_curriculum:
            system_features.append("Curriculum")

        system_type = "+".join(system_features) if system_features else "Rule-based"

        print(f"\n=== {system_type} 모델 성능 요약 ({n_runs}회 실행 평균) ===")
        print(f"총 수익률: {metrics_df['Total Return'].mean():.2%}")
        print(f"연평균 변동성: {metrics_df['Volatility'].mean():.3f}")
        print(f"최대 낙폭: {metrics_df['Max Drawdown'].mean():.2%}")
        print(f"샤프 지수: {metrics_df['Sharpe Ratio'].mean():.2f}")
        print(f"칼마 지수: {metrics_df['Calmar Ratio'].mean():.2f}")
        print(f"최종 자산: {metrics_df['Final Value'].mean():,.0f}원")

        # 다중 실행 결과 비교 시각화
        if use_learning_bcells and len(rl_trackers) > 1:
            self._create_multi_run_comparison(rl_trackers)

        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"bipd_{system_type}_{timestamp}"
            self.save_results(metrics_df, result_filename)

            if best_immune_system is not None:
                if use_learning_bcells:
                    model_filename = f"best_immune_system_{timestamp}"
                else:
                    model_filename = "best_legacy_immune_system.pkl"
                self.save_model(best_immune_system, model_filename)

        return metrics_df

    def _create_multi_run_comparison(self, rl_trackers):
        """다중 실행 결과 비교 시각화"""

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 실행별 보상 비교
        ax = axes[0, 0]
        for i, tracker in enumerate(rl_trackers):
            if tracker.episode_rewards:
                episodes = range(len(tracker.episode_rewards))
                ax.plot(
                    episodes,
                    tracker.episode_rewards,
                    alpha=0.7,
                    label=f"Run {i+1}",
                    linewidth=1,
                )
        ax.set_title("Reward Progression Comparison")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 최종 성과 분포
        ax = axes[0, 1]
        final_rewards = []
        for tracker in rl_trackers:
            if tracker.episode_rewards:
                final_rewards.append(np.mean(tracker.episode_rewards[-50:]))

        if final_rewards:
            ax.hist(
                final_rewards, bins=10, alpha=0.7, color="skyblue", edgecolor="black"
            )
            ax.axvline(
                np.mean(final_rewards),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(final_rewards):.3f}",
            )
            ax.set_title("Final Performance Distribution")
            ax.set_xlabel("Average Final Reward")
            ax.set_ylabel("Frequency")
            ax.legend()

        # 3. 학습 안정성 비교
        ax = axes[1, 0]
        for i, tracker in enumerate(rl_trackers):
            if len(tracker.episode_rewards) > 50:
                stds = []
                for j in range(50, len(tracker.episode_rewards), 10):
                    window_std = np.std(tracker.episode_rewards[j - 50 : j])
                    stds.append(window_std)
                if stds:
                    ax.plot(
                        range(len(stds)),
                        stds,
                        alpha=0.7,
                        label=f"Run {i+1}",
                        linewidth=1,
                    )
        ax.set_title("Learning Stability Comparison")
        ax.set_xlabel("Window")
        ax.set_ylabel("Reward Std")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. 수렴 속도 비교
        ax = axes[1, 1]
        convergence_episodes = []
        for i, tracker in enumerate(rl_trackers):
            if tracker.episode_rewards:
                target = np.mean(tracker.episode_rewards[-100:]) * 0.9
                for j in range(100, len(tracker.episode_rewards)):
                    if np.mean(tracker.episode_rewards[j - 20 : j]) >= target:
                        convergence_episodes.append(j)
                        break
                else:
                    convergence_episodes.append(len(tracker.episode_rewards))

        if convergence_episodes:
            runs = [f"Run {i+1}" for i in range(len(convergence_episodes))]
            ax.bar(runs, convergence_episodes, alpha=0.7, color="lightgreen")
            ax.set_title("Convergence Speed Comparison")
            ax.set_xlabel("Run")
            ax.set_ylabel("Episodes to Convergence")
            plt.setp(ax.get_xticklabels(), rotation=45)

        plt.tight_layout()
        comparison_path = os.path.join(self.rl_tracker_dir, "multi_run_comparison.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"다중 실행 비교 시각화 저장: {comparison_path}")
