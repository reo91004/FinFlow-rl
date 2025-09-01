# bipd/core/trainer.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import os
import time

from .environment import PortfolioEnvironment
from .system import ImmunePortfolioSystem
from data.features import FeatureExtractor
from utils.logger import BIPDLogger, get_session_directory
from utils.metrics import calculate_portfolio_metrics
from utils.visualization import create_episode_visualizations
from config import *

class BIPDTrainer:
    """
    BIPD 시스템 훈련자
    
    강화학습을 통해 면역 포트폴리오 시스템을 훈련
    """
    
    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame,
                 save_dir: Optional[str] = None):
        self.train_data = train_data
        self.test_data = test_data
        
        # 세션 디렉토리 내에 models 폴더 생성
        session_dir = get_session_directory()
        self.save_dir = save_dir if save_dir else os.path.join(session_dir, "models")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 컴포넌트 초기화
        self.feature_extractor = FeatureExtractor(LOOKBACK_WINDOW)
        self.train_env = PortfolioEnvironment(train_data, self.feature_extractor, 
                                            INITIAL_CAPITAL, TRANSACTION_COST)
        self.test_env = PortfolioEnvironment(test_data, self.feature_extractor, 
                                           INITIAL_CAPITAL, TRANSACTION_COST)
        
        self.immune_system = ImmunePortfolioSystem(
            n_assets=len(train_data.columns),
            state_dim=STATE_DIM,
            symbols=list(train_data.columns)  # 실제 종목 심볼 전달
        )
        
        # 훈련 통계
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'portfolio_values': [],
            'sharpe_ratios': [],
            'max_drawdowns': [],
            'crisis_levels': [],
            'selected_bcells': []
        }
        
        self.logger = BIPDLogger("Trainer")
        
        # 시각화 디렉토리 설정
        self.session_dir = get_session_directory()
        self.visualization_dir = os.path.join(self.session_dir, "visualizations")
        
        self.logger.info(
            f"BIPD 훈련자가 초기화되었습니다. "
            f"훈련데이터={len(train_data)}일, 테스트데이터={len(test_data)}일"
        )
        self.logger.info(f"시각화 결과 저장 위치: {self.visualization_dir}")
    
    def train(self, n_episodes: int = N_EPISODES, 
              save_interval: int = SAVE_INTERVAL) -> Dict:
        """
        메인 훈련 루프
        
        Args:
            n_episodes: 훈련 에피소드 수
            save_interval: 모델 저장 간격
            
        Returns:
            training_results: 훈련 결과 딕셔너리
        """
        self.logger.info(f"BIPD 시스템 훈련을 시작합니다. {n_episodes}개 에피소드")
        
        # T-Cell 사전 훈련
        self._pretrain_tcell()
        
        # 메인 훈련 루프
        start_time = time.time()
        
        # tqdm 진행바 설정
        pbar = tqdm(range(n_episodes), desc="BIPD Training")
        
        for episode in pbar:
            # Phase 2: B-Cell들의 에피소드 진행률 업데이트
            for bcell in self.immune_system.bcells.values():
                bcell.set_episode_progress(episode, n_episodes)
            
            # 에피소드 시작 로깅
            self.logger.info("=" * 25 + f" 에피소드 {episode + 1:,}/{n_episodes:,} 시작 " + "=" * 25)
            
            episode_results = self._run_episode(episode, training=True)
            
            # 통계 기록
            self._record_episode(episode, episode_results)
            
            # 에피소드 완료 로깅
            self._log_episode_completion(episode, episode_results)
            
            # XAI 시각화 생성 (매 10 에피소드마다)
            if (episode + 1) % 10 == 0 or episode == 0:  # 첫 에피소드와 10의 배수에서 시각화
                try:
                    visualization_files = create_episode_visualizations(
                        self, episode_results, self.visualization_dir
                    )
                    self.logger.info(
                        f"에피소드 {episode + 1} 시각화 완료: "
                        f"{len(visualization_files)}개 차트 생성"
                    )
                except Exception as e:
                    self.logger.warning(f"시각화 생성 실패: {e}")
            
            # 진행바 정보 업데이트
            current_reward = episode_results['avg_reward']
            avg_crisis = episode_results['crisis_stats']['avg_crisis']
            dominant_bcell = max(episode_results['bcell_usage'], key=episode_results['bcell_usage'].get)
            final_value = episode_results['final_value']
            
            # 최근 10 에피소드 평균 보상
            if len(self.training_history['rewards']) > 0:
                avg_reward_10 = np.mean(self.training_history['rewards'][-10:])
            else:
                avg_reward_10 = current_reward
            
            pbar.set_postfix({
                'Ep': episode + 1,
                'Reward': f"{current_reward:.3f}",
                'Avg10': f"{avg_reward_10:.3f}",
                'Crisis': f"{avg_crisis:.2f}",
                'Strategy': dominant_bcell[:3],  # 줄임
                'Value': f"{final_value/1000000:.1f}M"
            })
            
            # 주기적 로깅 (파일에만)
            if (episode + 1) % LOG_INTERVAL == 0:
                self._log_progress(episode, episode_results)
                
                # 환경 검증 통계 로깅 (매 50 에피소드마다)
                if (episode + 1) % 50 == 0:
                    validation_summary = self.train_env.get_validation_summary()
                    self.logger.info(f"환경 통계 요약 (에피소드 {episode + 1}): {validation_summary}")
            
            # 모델 저장
            if (episode + 1) % save_interval == 0:
                self._save_checkpoint(episode)
        
        pbar.close()
        
        training_time = time.time() - start_time
        
        # 훈련 완료
        self.logger.info(
            f"훈련이 완료되었습니다. "
            f"소요시간={training_time:.1f}초, "
            f"에피소드당 평균={training_time/n_episodes:.2f}초"
        )
        
        # 최종 모델 저장
        self._save_final_model()
        
        # 최종 시각화 생성
        self._create_final_visualizations()
        
        # 결과 요약
        training_summary = self._generate_training_summary()
        
        return training_summary
    
    def _pretrain_tcell(self) -> bool:
        """T-Cell 사전 훈련 (정상 시장 패턴 학습)"""
        self.logger.info("T-Cell 사전 훈련을 시작합니다.")
        
        # 훈련 데이터에서 특성 추출
        train_features = self.feature_extractor.extract_features_batch(self.train_data)
        
        if len(train_features) == 0:
            self.logger.error("T-Cell 훈련용 특성 추출 실패")
            return False
        
        # T-Cell 학습
        success = self.immune_system.fit_tcell(train_features)
        
        if success:
            self.logger.info(f"T-Cell 사전 훈련 완료: {len(train_features)}개 샘플 학습")
        
        return success
    
    def _run_episode(self, episode: int, training: bool = True) -> Dict:
        """단일 에피소드 실행"""
        env = self.train_env if training else self.test_env
        
        state = env.reset()
        total_reward = 0
        steps = 0
        
        # 에피소드 시작 디버그 로깅
        self.logger.debug(
            f"에피소드 {episode + 1} 시작: 초기 상태 크기={len(state)}, "
            f"환경 최대 스텝={env.max_steps}"
        )
        
        episode_data = {
            'rewards': [],
            'portfolio_values': [],
            'crisis_levels': [],
            'selected_bcells': [],
            'weights_history': [],
            'decision_info_history': []  # XAI 데이터 저장
        }
        
        while True:
            # 의사결정
            weights, decision_info = self.immune_system.decide(state, training)
            
            # 환경 스텝
            next_state, reward, done, env_info = env.step(weights)
            
            # 시스템 업데이트 (훈련 모드에서만)
            if training:
                self.immune_system.update(state, weights, reward, next_state, done)
            
            # 데이터 기록
            episode_data['rewards'].append(reward)
            episode_data['portfolio_values'].append(env_info['portfolio_value'])
            episode_data['crisis_levels'].append(decision_info['crisis_level'])
            episode_data['selected_bcells'].append(decision_info['selected_bcell'])
            episode_data['weights_history'].append(weights.copy())
            episode_data['decision_info_history'].append(decision_info)  # XAI 데이터 저장
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        # 에피소드 요약
        portfolio_metrics = env.get_portfolio_metrics()
        
        episode_summary = {
            'episode': episode,
            'total_reward': total_reward,
            'avg_reward': total_reward / steps,
            'steps': steps,
            'final_value': env.portfolio_value,
            'portfolio_metrics': portfolio_metrics,
            'crisis_stats': {
                'avg_crisis': np.mean(episode_data['crisis_levels']),
                'max_crisis': np.max(episode_data['crisis_levels']),
                'crisis_episodes': np.sum(np.array(episode_data['crisis_levels']) > 0.5)
            },
            'bcell_usage': {
                bcell: episode_data['selected_bcells'].count(bcell)
                for bcell in ['volatility', 'correlation', 'momentum', 'defensive', 'growth']
            },
            'episode_data': episode_data,
            # XAI 시각화를 위한 마지막 의사결정 데이터 (가장 중요한 의사결정 시점)
            'decision_data': episode_data['decision_info_history'][-1]['xai_data'] if episode_data['decision_info_history'] else {}
        }
        
        return episode_summary
    
    def _record_episode(self, episode: int, results: Dict) -> None:
        """에피소드 결과 기록"""
        history = self.training_history
        
        history['episodes'].append(episode)
        history['rewards'].append(results['total_reward'])
        history['portfolio_values'].append(results['final_value'])
        
        metrics = results['portfolio_metrics']
        history['sharpe_ratios'].append(metrics.get('sharpe_ratio', 0))
        history['max_drawdowns'].append(metrics.get('max_drawdown', 0))
        
        crisis_stats = results['crisis_stats']
        history['crisis_levels'].append(crisis_stats['avg_crisis'])
        
        # 가장 많이 사용된 B-Cell
        bcell_usage = results['bcell_usage']
        most_used = max(bcell_usage, key=bcell_usage.get)
        history['selected_bcells'].append(most_used)
    
    def _log_episode_completion(self, episode: int, results: Dict) -> None:
        """에피소드 완료 로깅"""
        metrics = results['portfolio_metrics']
        crisis_stats = results['crisis_stats']
        bcell_usage = results['bcell_usage']
        
        # B-Cell 사용률 계산
        total_decisions = sum(bcell_usage.values())
        bcell_percentages = {
            name: (count / total_decisions * 100) if total_decisions > 0 else 0
            for name, count in bcell_usage.items()
        }
        
        # 상세 에피소드 요약
        episode_summary = [
            f"에피소드 {episode + 1} 완료 요약:",
            f"  • 총 스텝: {results['steps']:,}단계",
            f"  • 총 보상: {results['total_reward']:,.2f}",
            f"  • 평균 보상: {results['avg_reward']:.4f}",
            f"  • 최종 가치: ₩{results['final_value']:,.0f}",
            f"  • 수익률: {metrics.get('total_return', 0):+.2%}",
            f"  • 샤프비율: {metrics.get('sharpe_ratio', 0):.3f}",
            f"  • 최대낙폭: {metrics.get('max_drawdown', 0):.2%}",
            f"  • 위기수준: 평균 {crisis_stats['avg_crisis']:.3f}, 최대 {crisis_stats['max_crisis']:.3f}",
            f"  • 위기상황: {crisis_stats['crisis_episodes']:,}회 / {results['steps']:,}단계 ({crisis_stats['crisis_episodes']/results['steps']*100:.1f}%)",
            "",
            "  B-Cell 전략 사용 비율:",
            f"    - 변동성(volatility): {bcell_percentages['volatility']:5.1f}%",
            f"    - 상관관계(correlation): {bcell_percentages['correlation']:5.1f}%",
            f"    - 모멘텀(momentum): {bcell_percentages['momentum']:5.1f}%",
            f"    - 방어전략(defensive): {bcell_percentages['defensive']:5.1f}%",
            f"    - 성장전략(growth): {bcell_percentages['growth']:5.1f}%"
        ]
        
        for line in episode_summary:
            self.logger.info(line)
        
        self.logger.info("=" * 60 + f" 에피소드 {episode + 1} 종료 " + "=" * 5)
    
    def _log_progress(self, episode: int, results: Dict) -> None:
        """진행 상황 로깅"""
        recent_rewards = self.training_history['rewards'][-LOG_INTERVAL:]
        recent_values = self.training_history['portfolio_values'][-LOG_INTERVAL:]
        recent_sharpes = self.training_history['sharpe_ratios'][-LOG_INTERVAL:]
        
        avg_reward = np.mean(recent_rewards)
        avg_value = np.mean(recent_values)
        avg_sharpe = np.mean(recent_sharpes)
        
        # B-Cell 사용 통계
        bcell_usage = results['bcell_usage']
        total_decisions = sum(bcell_usage.values())
        bcell_distribution = {
            name: count / total_decisions if total_decisions > 0 else 0
            for name, count in bcell_usage.items()
        }
        
        self.logger.info(
            f"에피소드 {episode + 1}: "
            f"평균보상={avg_reward:.3f}, "
            f"포트폴리오가치={avg_value:,.0f}, "
            f"샤프비율={avg_sharpe:.3f}"
        )
        
        self.logger.debug(
            f"B-Cell 사용률: "
            f"변동성={bcell_distribution['volatility']:.2%}, "
            f"상관관계={bcell_distribution['correlation']:.2%}, "
            f"모멘텀={bcell_distribution['momentum']:.2%}"
        )
    
    def _save_checkpoint(self, episode: int) -> None:
        """체크포인트 저장"""
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_episode_{episode}")
        success = self.immune_system.save_system(checkpoint_path)
        
        if success:
            self.logger.debug(f"체크포인트 저장: 에피소드 {episode}")
    
    def _save_final_model(self) -> None:
        """최종 모델 저장"""
        model_path = os.path.join(self.save_dir, "bipd_final_model")
        success = self.immune_system.save_system(model_path)
        
        if success:
            self.logger.info(f"최종 모델이 저장되었습니다: {model_path}")
        else:
            self.logger.error(f"최종 모델 저장 실패: {model_path}")
    
    def _generate_training_summary(self) -> Dict:
        """훈련 요약 생성"""
        history = self.training_history
        
        if not history['rewards']:
            return {'message': '훈련 데이터가 없습니다.'}
        
        # 성과 통계
        final_rewards = history['rewards'][-50:] if len(history['rewards']) >= 50 else history['rewards']
        
        summary = {
            'total_episodes': len(history['episodes']),
            'final_avg_reward': np.mean(final_rewards),
            'best_reward': np.max(history['rewards']),
            'worst_reward': np.min(history['rewards']),
            'final_portfolio_value': history['portfolio_values'][-1],
            'best_portfolio_value': np.max(history['portfolio_values']),
            'final_sharpe_ratio': history['sharpe_ratios'][-1],
            'best_sharpe_ratio': np.max(history['sharpe_ratios']),
            'avg_crisis_level': np.mean(history['crisis_levels']),
            'training_stability': np.std(final_rewards),
            'system_performance': self.immune_system.get_performance_summary()
        }
        
        return summary
    
    def evaluate(self, n_episodes: int = 10) -> Dict:
        """테스트 데이터로 평가"""
        self.logger.info(f"BIPD 시스템 평가를 시작합니다. {n_episodes}개 에피소드")
        
        evaluation_results = []
        
        for episode in tqdm(range(n_episodes), desc="Evaluation Episodes"):
            results = self._run_episode(episode, training=False)
            evaluation_results.append(results)
        
        # 평가 요약
        portfolio_values = [r['final_value'] for r in evaluation_results]
        sharpe_ratios = [r['portfolio_metrics']['sharpe_ratio'] for r in evaluation_results]
        max_drawdowns = [r['portfolio_metrics']['max_drawdown'] for r in evaluation_results]
        
        evaluation_summary = {
            'n_episodes': n_episodes,
            'avg_final_value': np.mean(portfolio_values),
            'std_final_value': np.std(portfolio_values),
            'avg_sharpe_ratio': np.mean(sharpe_ratios),
            'std_sharpe_ratio': np.std(sharpe_ratios),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'best_performance': np.max(portfolio_values),
            'worst_performance': np.min(portfolio_values),
            'success_rate': np.sum([v > INITIAL_CAPITAL for v in portfolio_values]) / n_episodes,
            'detailed_results': evaluation_results
        }
        
        self.logger.info(
            f"평가 완료: "
            f"평균 포트폴리오 가치={evaluation_summary['avg_final_value']:,.0f}, "
            f"평균 샤프비율={evaluation_summary['avg_sharpe_ratio']:.3f}, "
            f"성공률={evaluation_summary['success_rate']:.1%}"
        )
        
        return evaluation_summary
    
    def plot_training_results(self, save_path: Optional[str] = None) -> None:
        """훈련 결과 시각화"""
        if not self.training_history['rewards']:
            self.logger.warning("시각화할 훈련 데이터가 없습니다.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('BIPD System Training Results', fontsize=16)
        
        # 보상 추이
        axes[0, 0].plot(self.training_history['episodes'], self.training_history['rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 포트폴리오 가치 추이
        axes[0, 1].plot(self.training_history['episodes'], self.training_history['portfolio_values'])
        axes[0, 1].axhline(y=INITIAL_CAPITAL, color='r', linestyle='--', label='Initial Capital')
        axes[0, 1].set_title('Portfolio Value')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Portfolio Value')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 샤프 비율 추이
        axes[0, 2].plot(self.training_history['episodes'], self.training_history['sharpe_ratios'])
        axes[0, 2].set_title('Sharpe Ratio')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Sharpe Ratio')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 위기 수준 분포
        axes[1, 0].hist(self.training_history['crisis_levels'], bins=30, alpha=0.7)
        axes[1, 0].set_title('Crisis Level Distribution')
        axes[1, 0].set_xlabel('Crisis Level')
        axes[1, 0].set_ylabel('Frequency')
        
        # B-Cell 사용 분포
        bcell_counts = {}
        for bcell in self.training_history['selected_bcells']:
            bcell_counts[bcell] = bcell_counts.get(bcell, 0) + 1
        
        axes[1, 1].bar(bcell_counts.keys(), bcell_counts.values())
        axes[1, 1].set_title('B-Cell Usage Distribution')
        axes[1, 1].set_xlabel('B-Cell Type')
        axes[1, 1].set_ylabel('Usage Count')
        
        # 학습 진행도 (이동 평균)
        window_size = min(50, len(self.training_history['rewards']) // 10)
        if window_size > 1:
            rewards_ma = pd.Series(self.training_history['rewards']).rolling(window_size).mean()
            axes[1, 2].plot(self.training_history['episodes'], rewards_ma, label=f'MA({window_size})')
            axes[1, 2].plot(self.training_history['episodes'], self.training_history['rewards'], 
                           alpha=0.3, label='Raw Rewards')
            axes[1, 2].set_title('Learning Progress')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Reward')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"훈련 결과 그래프가 저장되었습니다: {save_path}")
        
        plt.show()
    
    def _require_keys(self, data: Dict, required_keys: List[str], context: str) -> None:
        """계약 검증: 필수 키 존재 확인 (연구용 표준)"""
        missing_keys = [key for key in required_keys if key not in data]
        assert not missing_keys, (
            f"[{context}] 필수 키 누락: {missing_keys}. "
            f"제공된 키: {list(data.keys())}"
        )
        
    def benchmark_comparison(self, benchmark_strategy: str = 'equal_weight') -> Dict:
        """벤치마크와 비교 (계약 검증 강화)"""
        self.logger.info(f"벤치마크 비교를 수행합니다: {benchmark_strategy}")
        
        # BIPD 시스템 평가
        bipd_results = self.evaluate(n_episodes=5)
        
        # BIPD 결과 계약 검증
        required_bipd_keys = ['avg_final_value', 'avg_sharpe_ratio', 'avg_max_drawdown']
        self._require_keys(bipd_results, required_bipd_keys, f"BIPD_results")
        
        # 벤치마크 전략 (균등 가중)
        benchmark_env = PortfolioEnvironment(self.test_data, self.feature_extractor,
                                           initial_capital=self.test_env.initial_capital,
                                           transaction_cost=self.test_env.transaction_cost)
        state = benchmark_env.reset()
        
        equal_weights = np.ones(self.train_data.shape[1]) / self.train_data.shape[1]
        
        while True:
            next_state, reward, done, info = benchmark_env.step(equal_weights)
            if done:
                break
            state = next_state
        
        benchmark_metrics = benchmark_env.get_portfolio_metrics()
        
        # 벤치마크 결과 계약 검증
        required_benchmark_keys = ['portfolio_value', 'total_return', 'sharpe_ratio', 'max_drawdown']
        self._require_keys(benchmark_metrics, required_benchmark_keys, f"benchmark_{benchmark_strategy}")
        
        # final_value 키 존재 확인 (중요: 기존 코드 호환성)
        if 'final_value' not in benchmark_metrics and 'portfolio_value' in benchmark_metrics:
            benchmark_metrics['final_value'] = benchmark_metrics['portfolio_value']
        
        # final_value 키 계약 검증 (KeyError 방지)
        self._require_keys(benchmark_metrics, ['final_value'], f"benchmark_{benchmark_strategy}_final")
        
        # 비교 결과
        comparison = {
            'bipd_performance': {
                'avg_final_value': bipd_results['avg_final_value'],
                'avg_sharpe_ratio': bipd_results['avg_sharpe_ratio'],
                'avg_max_drawdown': bipd_results['avg_max_drawdown']
            },
            'benchmark_performance': benchmark_metrics,
            'outperformance': {
                'value_improvement': (bipd_results['avg_final_value'] - benchmark_metrics['final_value']) / benchmark_metrics['final_value'],
                'sharpe_improvement': bipd_results['avg_sharpe_ratio'] - benchmark_metrics['sharpe_ratio'],
                'drawdown_improvement': benchmark_metrics['max_drawdown'] - bipd_results['avg_max_drawdown']
            }
        }
        
        # 결과 계약 검증
        required_comparison_keys = ['bipd_performance', 'benchmark_performance', 'outperformance']
        self._require_keys(comparison, required_comparison_keys, "benchmark_comparison_output")
        
        self.logger.info(
            f"벤치마크 비교 완료 (계약 검증 통과): "
            f"수익률 개선={comparison['outperformance']['value_improvement']:.2%}, "
            f"샤프비율 개선={comparison['outperformance']['sharpe_improvement']:.3f}"
        )
        
        return comparison
    
    def _create_final_visualizations(self) -> None:
        """최종 학습 결과 시각화 생성"""
        try:
            # matplotlib 기반 훈련 결과 차트 생성
            training_chart_path = os.path.join(self.visualization_dir, "training_results_detailed.png")
            self.plot_training_results(training_chart_path)
            self.logger.info(f"상세 훈련 결과 차트 저장: {training_chart_path}")
            
            self.logger.info("모든 시각화가 완료되었습니다.")
            
        except Exception as e:
            self.logger.error(f"최종 시각화 생성 실패: {e}")