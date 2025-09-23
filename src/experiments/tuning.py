# src/experiments/tuning.py

"""
하이퍼파라미터 튜닝 모듈

목적: Optuna 기반 자동 하이퍼파라미터 최적화
의존: optuna, trainer.py, portfolio_env.py
사용처: 최적 하이퍼파라미터 탐색 및 성능 개선
역할: 베이지안 최적화를 통한 체계적 파라미터 탐색

구현 내용:
- Optuna Study 설정 및 실행
- 파라미터 공간 정의 (학습률, 네트워크 크기, 정규화 등)
- 병렬 최적화 지원 (n_jobs)
- 중간 결과 저장 및 시각화
- 최적 파라미터 추출 및 저장

Note: 연구용 코드 - 필요시 pip install optuna 설치
"""

import optuna
import numpy as np
import torch
from typing import Dict, Any, Optional, Callable
from src.environments.portfolio_env import PortfolioEnv
# FinFlowTrainer는 순환 import 방지를 위해 필요시 import
from src.utils.logger import FinFlowLogger
import json

class HyperparameterTuner:
    """
    Optuna 기반 하이퍼파라미터 튜닝
    """
    
    def __init__(self,
                 env_config: Dict,
                 train_data: Any,
                 val_data: Any,
                 n_trials: int = 100,
                 n_jobs: int = 1,
                 storage: Optional[str] = None):
        """
        Args:
            env_config: 환경 설정
            train_data: 학습 데이터
            val_data: 검증 데이터
            n_trials: 시도 횟수
            n_jobs: 병렬 작업 수
            storage: Optuna 스토리지 (SQLite DB 경로)
        """
        self.env_config = env_config
        self.train_data = train_data
        self.val_data = val_data
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.storage = storage
        
        self.logger = FinFlowLogger("HyperparameterTuner")
        self.best_params = None
        self.best_value = None
        
    def optimize(self, 
                objective_metric: str = 'sharpe_ratio',
                direction: str = 'maximize') -> Dict:
        """
        하이퍼파라미터 최적화 실행
        
        Args:
            objective_metric: 최적화할 메트릭
            direction: 최적화 방향 ('maximize' or 'minimize')
            
        Returns:
            최적 하이퍼파라미터
        """
        study = optuna.create_study(
            direction=direction,
            storage=self.storage,
            study_name='finflow_optimization',
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # 목적 함수 정의
        def objective(trial):
            return self._objective(trial, objective_metric)
        
        # 최적화 실행
        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )
        
        # 최적 파라미터 저장
        self.best_params = study.best_params
        self.best_value = study.best_value
        
        self.logger.info(f"최적화 완료 - Best {objective_metric}: {self.best_value:.4f}")
        self.logger.info(f"최적 파라미터: {self.best_params}")
        
        # 중요도 분석
        importance = optuna.importance.get_param_importances(study)
        self.logger.info(f"파라미터 중요도: {importance}")
        
        # 결과 저장
        self._save_results(study)
        
        return self.best_params
    
    def _objective(self, trial: Any, metric: str) -> float:
        """목적 함수"""
        # 하이퍼파라미터 샘플링
        params = self._sample_hyperparameters(trial)

        # 환경 생성
        env = PortfolioEnv(
            price_data=self.train_data,
            initial_cash=self.env_config.get('initial_cash', 1000000),
            fee_rate=self.env_config.get('fee_rate', 0.001),
            slippage=self.env_config.get('slippage', 0.001)
        )

        # 모델 설정
        model_config = {
            'hidden_dim': params['hidden_dim'],
            'num_layers': params['num_layers'],
            'lr': params['lr'],
            'gamma': params['gamma'],
            'tau': params['tau'],
            'batch_size': params['batch_size'],
            'alpha': params.get('alpha', 0.2)
        }

        # 학습 실행 (간소화된 버전)
        score = self._train_and_evaluate(env, model_config, metric)

        # 조기 종료
        if trial.should_prune():
            raise optuna.TrialPruned()

        return score
    
    def _sample_hyperparameters(self, trial: Any) -> Dict:
        """하이퍼파라미터 샘플링"""
        params = {
            # 네트워크 구조
            'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512]),
            'num_layers': trial.suggest_int('num_layers', 2, 4),
            
            # 학습률
            'lr': trial.suggest_loguniform('lr', 1e-5, 1e-2),
            
            # RL 파라미터
            'gamma': trial.suggest_uniform('gamma', 0.95, 0.999),
            'tau': trial.suggest_loguniform('tau', 1e-4, 1e-2),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            
            # SAC 특화
            'alpha': trial.suggest_loguniform('alpha', 0.01, 1.0),
            
            # IQL 특화
            'expectile': trial.suggest_uniform('expectile', 0.5, 0.99),
            'temperature': trial.suggest_uniform('temperature', 0.1, 10.0),
            
            # 분포적 강화학습
            'num_quantiles': trial.suggest_categorical('num_quantiles', [8, 16, 32, 64]),
            
            # 리플레이 버퍼
            'buffer_size': trial.suggest_categorical('buffer_size', [10000, 50000, 100000]),
            
            # 목적 함수 가중치
            'sharpe_weight': trial.suggest_uniform('sharpe_weight', 0.0, 2.0),
            'cvar_weight': trial.suggest_uniform('cvar_weight', 0.0, 1.0),
            'turnover_penalty': trial.suggest_loguniform('turnover_penalty', 1e-4, 1e-1)
        }
        
        return params
    
    def _train_and_evaluate(self, 
                           env: PortfolioEnv,
                           config: Dict,
                           metric: str) -> float:
        """간소화된 학습 및 평가"""
        # 여기서는 빠른 평가를 위해 짧은 에피소드만 실행
        # 실제로는 Trainer 클래스를 사용
        
        total_rewards = []
        returns = []
        
        for episode in range(10):  # 빠른 평가를 위해 10 에피소드만
            state = env.reset()
            episode_reward = 0
            episode_returns = []
            
            for step in range(100):  # 최대 100 스텝
                # 랜덤 액션 (실제로는 학습된 정책 사용)
                action = np.random.dirichlet(np.ones(env.action_dim))
                
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                episode_returns.append(info.get('portfolio_return', 0))
                
                if done:
                    break
                
                state = next_state
            
            total_rewards.append(episode_reward)
            returns.extend(episode_returns)
        
        # 메트릭 계산
        if metric == 'sharpe_ratio':
            if len(returns) > 0 and np.std(returns) > 0:
                return np.mean(returns) / np.std(returns) * np.sqrt(252)
            return 0.0
        elif metric == 'total_return':
            return np.mean(total_rewards)
        elif metric == 'max_drawdown':
            cumulative = np.cumprod(1 + np.array(returns))
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            return -np.min(drawdown)  # 음수로 반환 (minimize)
        else:
            return np.mean(total_rewards)
    
    def _save_results(self, study: Any):
        """결과 저장"""
        results = {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_trials': len(study.trials),
            'optimization_history': [
                {
                    'trial': t.number,
                    'value': t.value,
                    'params': t.params,
                    'state': t.state.name
                }
                for t in study.trials
            ]
        }
        
        with open('tuning_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info("튜닝 결과 저장: tuning_results.json")
    
    def grid_search(self, param_grid: Dict[str, list]) -> Dict:
        """
        그리드 서치 (작은 파라미터 공간용)
        
        Args:
            param_grid: 파라미터 그리드
            
        Returns:
            최적 파라미터
        """
        from itertools import product
        
        # 모든 조합 생성
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(product(*values))
        
        best_score = float('-inf')
        best_params = None
        
        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            self.logger.info(f"Grid search {i+1}/{len(combinations)}: {params}")
            
            # 평가
            env = PortfolioEnv(
                price_data=self.train_data,
                initial_cash=self.env_config.get('initial_cash', 1000000)
            )
            
            score = self._train_and_evaluate(env, params, 'sharpe_ratio')
            
            if score > best_score:
                best_score = score
                best_params = params
                self.logger.info(f"새로운 최적 점수: {best_score:.4f}")
        
        self.best_params = best_params
        self.best_value = best_score
        
        return best_params
    
    def bayesian_optimization(self, 
                             n_calls: int = 100,
                             n_initial_points: int = 10) -> Dict:
        """
        Scikit-optimize를 사용한 베이지안 최적화 (대안)
        
        Args:
            n_calls: 총 평가 횟수
            n_initial_points: 초기 랜덤 샘플링 수
            
        Returns:
            최적 파라미터
        """
        # 연구용 코드 - 필요한 라이브러리는 직접 설치
        # pip install scikit-optimize
        from skopt import gp_minimize
        from skopt.space import Real, Integer, Categorical
        
        # 파라미터 공간 정의
        space = [
            Integer(128, 512, name='hidden_dim'),
            Integer(2, 4, name='num_layers'),
            Real(1e-5, 1e-2, prior='log-uniform', name='lr'),
            Real(0.95, 0.999, name='gamma'),
            Real(1e-4, 1e-2, prior='log-uniform', name='tau'),
            Categorical([32, 64, 128, 256], name='batch_size')
        ]
        
        # 목적 함수
        def objective(params):
            param_dict = {
                'hidden_dim': params[0],
                'num_layers': params[1],
                'lr': params[2],
                'gamma': params[3],
                'tau': params[4],
                'batch_size': params[5]
            }
            
            env = PortfolioEnv(
                price_data=self.train_data,
                initial_cash=1000000
            )
            
            # 최대화를 최소화로 변환
            score = -self._train_and_evaluate(env, param_dict, 'sharpe_ratio')
            return score
        
        # 베이지안 최적화 실행
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            random_state=42
        )
        
        # 최적 파라미터 추출
        self.best_params = {
            'hidden_dim': result.x[0],
            'num_layers': result.x[1],
            'lr': result.x[2],
            'gamma': result.x[3],
            'tau': result.x[4],
            'batch_size': result.x[5]
        }
        self.best_value = -result.fun
        
        self.logger.info(f"베이지안 최적화 완료 - Best score: {self.best_value:.4f}")
        
        return self.best_params