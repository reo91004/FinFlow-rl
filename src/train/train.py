"""
강화학습 학습 모듈

PPO 알고리즘을 이용한 포트폴리오 최적화 모델 학습 기능을 제공합니다.
에이전트 학습, 검증, 앙상블 모델 관리 등의 기능을 포함합니다.
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import logging
import pickle
import gc
import glob

from src.models.ppo import PPO
from src.models.memory import Memory
from src.environment.portfolio_env import StockPortfolioEnv
from src.constants import (
    NUM_EPISODES,
    ENSEMBLE_SIZE,
    DEVICE,
    VALIDATION_INTERVAL,
    PPO_UPDATE_TIMESTEP,
    NORMALIZE_STATES,
    RESULTS_BASE_PATH
)

def create_results_dir(mode="train", ensemble_id=None):
    """
    학습 결과를 저장할 디렉토리를 생성합니다.
    
    Args:
        mode (str): 학습 모드 (train, test 등)
        ensemble_id (int, optional): 앙상블 모델 ID
        
    Returns:
        str: 생성된 디렉토리 경로
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 앙상블 모델인 경우 앙상블 ID 추가
    if ensemble_id is not None:
        run_dir = os.path.join(
            RESULTS_BASE_PATH, f"{mode}_{timestamp}_ensemble_{ensemble_id}"
        )
    else:
        run_dir = os.path.join(RESULTS_BASE_PATH, f"{mode}_{timestamp}")
    
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def train_agent(
    env,
    ppo_agent,
    num_episodes=NUM_EPISODES,
    validate_env=None,
    run_dir=None,
    logger=None
):
    """
    단일 PPO 에이전트를 학습합니다.
    
    Args:
        env (StockPortfolioEnv): 학습 환경
        ppo_agent (PPO): 학습할 PPO 에이전트
        num_episodes (int): 학습할 에피소드 수
        validate_env (StockPortfolioEnv, optional): 검증 환경
        run_dir (str, optional): 결과 저장 디렉토리
        logger (Logger, optional): 로거 객체
        
    Returns:
        dict: 학습 결과 (보상 이력, 포트폴리오 가치 이력 등)
    """
    if logger is None:
        logger = logging.getLogger("PortfolioRL")
    
    # 결과 저장 디렉토리가 없으면 생성
    if run_dir is None:
        run_dir = create_results_dir()
        logger.info(f"결과 디렉토리 생성: {run_dir}")
    
    # 학습 메모리 생성
    memory = Memory()
    
    # 학습 이력 추적 변수들 초기화
    reward_history = []
    portfolio_value_history = []
    loss_history = []
    validation_reward_history = []
    episode_lengths = []
    best_reward = -float("inf")
    
    # 타임스텝 카운터
    time_step = 0
    update_step = 0  # PPO 업데이트 횟수 추적
    episode_reward_threshold = -10.0  # 조기 종료 임계값
    
    # 환경 상태 정규화 관련 변수
    if NORMALIZE_STATES and env.obs_rms is not None:
        ppo_agent.obs_rms = env.obs_rms  # 정규화 상태 공유
    
    # 탐색(exploration) 매개변수
    epsilon_start = 0.3  # 초기 탐색 확률
    epsilon_final = 0.01  # 최종 탐색 확률
    epsilon_decay = 0.995  # 탐색 확률 감쇠율
    current_epsilon = epsilon_start  # 현재 탐색 확률
    
    # 학습 시작 로그
    logger.info(f"학습 시작: {num_episodes} 에피소드")
    logger.info(f"앙상블 에이전트: {ENSEMBLE_SIZE}개, 디바이스: {DEVICE}")
    logger.info(f"PPO 업데이트 주기: {PPO_UPDATE_TIMESTEP} 타임스텝")
    logger.info(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    training_start_time = time.time()
    
    # 에피소드별 학습
    for episode in range(1, num_episodes + 1):
        episode_start_time = time.time()
        
        # 환경 초기화 (무작위 시작 인덱스)
        state, _ = env.reset()
        current_episode_reward = 0
        step_count = 0
        done = False
        truncated = False
        portfolio_values = []
        
        # 에피소드 진행
        while not (done or truncated):
            # 메모리에 저장할 에피소드 길이 제한
            if step_count >= env.max_episode_length:
                truncated = True
                break
                
            # 액션 선택 (입실론-그리디 탐색 적용)
            if np.random.random() < current_epsilon:
                # 임의 행동 샘플링 (탐색)
                random_action = np.random.random(env.action_space.shape)
                random_action /= random_action.sum()  # 합이 1이 되도록 정규화
                action = random_action
                
                # 메모리에 저장하기 위한 log_prob과 value 계산
                _, log_prob, value = ppo_agent.policy_old.act(state)
            else:
                # 정책에 따른 행동 선택 (활용)
                action, log_prob, value = ppo_agent.policy_old.act(state)
            
            # 환경에서 한 스텝 진행
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # 메모리에 경험 저장
            memory.add_experience(state, action, log_prob, reward, terminated, value)
            
            # 타임스텝 카운터 증가 및 PPO 업데이트 체크
            time_step += 1
            
            # PPO 업데이트 주기 체크 및 실행
            if time_step % PPO_UPDATE_TIMESTEP == 0:
                if len(memory.states) > 100:  # 최소 100개 이상의 샘플이 있을 때만 업데이트
                    logger.debug(f"PPO 업데이트 실행 (타임스텝: {time_step}, 에피소드: {episode})")
                    loss = ppo_agent.update(memory)
                    loss_history.append(loss)
                    memory.clear_memory()
                    update_step += 1
                    
                    # PPO 업데이트 후에 학습률 스케줄러 업데이트
                    current_lr = ppo_agent.update_lr_scheduler()
                    if current_lr is not None:
                        logger.debug(f"학습률 업데이트: {current_lr:.7f}")
            
            # 내부 상태 업데이트
            state = next_state
            portfolio_values.append(info["portfolio_value"])
            current_episode_reward += info.get("raw_reward", reward)
            step_count += 1
            
            # 종료 조건 체크
            if terminated:
                done = True
                
            # 너무 낮은 보상값을 보이는 경우 조기 종료 (학습 시간 절약)
            if current_episode_reward < episode_reward_threshold * (step_count / 10):
                logger.warning(
                    f"에피소드 {episode}가 낮은 누적 보상으로 조기 종료됨"
                )
                truncated = True
        
        # 에피소드 통계 기록
        episode_lengths.append(step_count)
        reward_history.append(current_episode_reward)
        portfolio_value_history.append(portfolio_values)
        
        # 탐색 확률 감소
        current_epsilon = max(epsilon_final, current_epsilon * epsilon_decay)
        
        # 에피소드가 끝났지만 메모리에 데이터가 충분하고 일정 주기가 지났으면 추가 업데이트
        if episode % 5 == 0 and len(memory.states) > 200:
            logger.debug(f"에피소드 종료 후 추가 PPO 업데이트 (에피소드: {episode})")
            loss = ppo_agent.update(memory)
            loss_history.append(loss)
            memory.clear_memory()
            update_step += 1
        
        # 에피소드 정보 출력
        episode_time = time.time() - episode_start_time
        logger.info(
            f"=== 에피소드 {episode}/{num_episodes} 완료 "
            f"(보상: {current_episode_reward:.4f}, "
            f"포트폴리오: {portfolio_values[-1]:.2f}, "
            f"스텝: {step_count}, "
            f"탐색: {current_epsilon:.3f}, "
            f"시간: {episode_time:.2f}초)"
        )
        
        # 주기적 검증 수행
        if validate_env is not None and episode % VALIDATION_INTERVAL == 0:
            validation_reward = ppo_agent.validate(validate_env)
            validation_reward_history.append((episode, validation_reward))
            
            logger.info(
                f"검증 결과 (에피소드 {episode}): 평균 보상 {validation_reward:.4f}"
            )
            
            # 가장 좋은 모델 저장
            if validation_reward > best_reward:
                best_reward = validation_reward
                model_file = os.path.join(run_dir, "best_model.pth")
                ppo_agent.save_model(episode, validation_reward)
                logger.info(
                    f"새로운 최고 성능! 보상: {validation_reward:.4f} -> {model_file}"
                )
                
            # Early Stopping 체크
            if ppo_agent.check_early_stopping(validation_reward):
                logger.warning(
                    f"Early Stopping 발동: {ppo_agent.early_stopping_patience} 에피소드 동안 성능 향상 없음"
                )
                break
                
    # 총 학습 시간 출력
    total_training_time = time.time() - training_start_time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(
        f"학습 완료! 총 소요 시간: {int(hours)}시간 {int(minutes)}분 {seconds:.2f}초, "
        f"총 업데이트 횟수: {update_step}회"
    )
    
    # 학습 결과 저장
    try:
        results = {
            "reward_history": reward_history,
            "portfolio_value_history": portfolio_value_history,
            "loss_history": loss_history,
            "validation_reward_history": validation_reward_history,
            "episode_lengths": episode_lengths,
            "total_training_time": total_training_time,
            "update_count": update_step,
        }
        
        with open(os.path.join(run_dir, "training_results.pkl"), "wb") as f:
            pickle.dump(results, f)
            
        logger.info(f"학습 결과 저장 완료: {run_dir}/training_results.pkl")
        
        # 학습 곡선 저장
        plot_training_curves(reward_history, loss_history, validation_reward_history, run_dir)
        logger.info(f"학습 곡선 저장 완료: {run_dir}/learning_curves.png")
    except Exception as e:
        logger.error(f"결과 저장 중 오류 발생: {e}")
    
    # 마지막 모델 저장
    model_file = os.path.join(run_dir, "final_model.pth")
    ppo_agent.save_model(num_episodes, reward_history[-1] if reward_history else -float("inf"))
    logger.info(f"최종 모델 저장 완료: {model_file}")
    
    # 메모리 정리
    del memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return results

def train_ensemble(
    train_data,
    test_data,
    train_dates,
    test_dates,
    ensemble_size=ENSEMBLE_SIZE,
    ensemble_dir=None,
    logger=None
):
    """
    앙상블 PPO 에이전트들을 학습합니다.
    
    Args:
        train_data (np.ndarray): 학습 데이터
        test_data (np.ndarray): 테스트 데이터
        train_dates (pd.DatetimeIndex): 학습 데이터 날짜
        test_dates (pd.DatetimeIndex): 테스트 데이터 날짜
        ensemble_size (int): 앙상블 모델 수
        ensemble_dir (str, optional): 앙상블 결과 저장 디렉토리
        logger (Logger, optional): 로거 객체
        
    Returns:
        list: 학습된, PPO 에이전트 리스트
    """
    if logger is None:
        logger = logging.getLogger("PortfolioRL")
    
    # 앙상블 결과 디렉토리가 없으면 생성
    if ensemble_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ensemble_dir = os.path.join(RESULTS_BASE_PATH, f"ensemble_{timestamp}")
        os.makedirs(ensemble_dir, exist_ok=True)
    
    # 앙상블 모델 폴더 생성 - 좀 더 체계적인 구조로
    models_dir = os.path.join(ensemble_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    logger.info(f"앙상블 학습 시작: {ensemble_size}개 모델")
    ensemble_agents = []
    seed_values = []
    
    # 기준 시드 설정
    base_seed = int(time.time()) % 10000
    logger.info(f"앙상블 기준 시드: {base_seed}")
    
    # 반드시 ensemble_size만큼 모델이 생성되도록 함
    for i in range(ensemble_size):
        # 각 모델마다 명확하게 다른 시드 패턴 설정
        model_seed = base_seed + (i * 111)  # 명확한 간격으로 시드 분리
        seed_values.append(model_seed)
        
        # 시드 설정 로깅
        logger.info(f"앙상블 모델 {i+1}/{ensemble_size} 학습 시작 (시드: {model_seed})")
        
        # 시드 설정
        torch.manual_seed(model_seed)
        np.random.seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(model_seed)
            torch.cuda.manual_seed_all(model_seed)
        
        # 모델별 개별 결과 디렉토리 생성 (체계적인 이름)
        run_dir = os.path.join(models_dir, f"model_{i}")
        os.makedirs(run_dir, exist_ok=True)
        
        # 학습/검증용 환경 생성
        train_env = StockPortfolioEnv(train_data)
        val_env = StockPortfolioEnv(test_data)
        
        # PPO 에이전트 생성
        n_assets = train_data.shape[1]
        n_features = train_data.shape[2]
        ppo_agent = PPO(
            n_assets=n_assets,
            n_features=n_features,
            model_path=run_dir,
            logger=logger,
        )
        
        # 에이전트 학습
        train_agent(
            env=train_env, 
            ppo_agent=ppo_agent, 
            validate_env=val_env, 
            run_dir=run_dir, 
            logger=logger
        )
        
        # 앙상블에 추가
        ensemble_agents.append(ppo_agent)
        
        # 메모리 정리
        del train_env, val_env
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 모델 수 확인
    if len(ensemble_agents) != ensemble_size:
        logger.warning(f"생성된 모델 수가 요청된 앙상블 크기와 다릅니다: {len(ensemble_agents)} vs {ensemble_size}")
    
    # 앙상블 메타데이터 저장
    meta_file = os.path.join(ensemble_dir, "ensemble_meta.pkl")
    meta_data = {
        "ensemble_size": ensemble_size,
        "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_paths": [os.path.join(models_dir, f"model_{i}") for i in range(ensemble_size)],
        "seed_values": seed_values,  # 실제 사용된 시드값 저장
        "base_seed": base_seed
    }
    
    with open(meta_file, "wb") as f:
        pickle.dump(meta_data, f)
    
    # 앙상블 모델을 결합하는 시도
    try:
        # 앙상블 투표 모델 가중치 설정 - 가중치 평균 접근법
        if len(ensemble_agents) > 0:
            logger.info("앙상블 결합 모델 생성 중...")
            combined_model_dir = os.path.join(ensemble_dir, "combined_model")
            os.makedirs(combined_model_dir, exist_ok=True)
            
            # 각 에이전트 모델의 상태 사전 로드
            combined_state_dict = None
            
            for idx, agent in enumerate(ensemble_agents):
                if hasattr(agent.policy, 'state_dict'):
                    if combined_state_dict is None:
                        # 첫 모델의 상태 사전으로 초기화
                        combined_state_dict = {}
                        for key, param in agent.policy.state_dict().items():
                            combined_state_dict[key] = param.clone()
                    else:
                        # 앙상블 모델의 파라미터 누적 (평균 계산 위함)
                        for key, param in agent.policy.state_dict().items():
                            combined_state_dict[key] += param
            
            # 평균 계산
            if combined_state_dict:
                for key in combined_state_dict:
                    combined_state_dict[key] /= len(ensemble_agents)
                
                # 앙상블 기준 모델 생성 (첫 번째 모델 구조 사용)
                base_agent = ensemble_agents[0]
                combined_agent = PPO(
                    n_assets=n_assets,
                    n_features=n_features,
                    model_path=combined_model_dir,
                    logger=logger
                )
                
                # 결합된 가중치로 모델 초기화
                combined_agent.policy.load_state_dict(combined_state_dict)
                combined_agent.policy_old.load_state_dict(combined_state_dict)
                
                # 결합 모델 저장
                model_file = os.path.join(combined_model_dir, "ensemble_model.pth")
                combined_agent.save_model(0, 0.0)
                logger.info(f"앙상블 결합 모델 저장 완료: {combined_model_dir}/ensemble_model.pth")
                
                # 메타데이터에 결합 모델 경로 추가
                meta_data["combined_model_path"] = combined_model_dir
                with open(meta_file, "wb") as f:
                    pickle.dump(meta_data, f)
    except Exception as e:
        logger.warning(f"앙상블 결합 모델 생성 중 오류 발생: {e}")
    
    logger.info(f"앙상블 학습 완료: {len(ensemble_agents)}/{ensemble_size}개 모델")
    return ensemble_agents

def load_ensemble(ensemble_dir, logger=None):
    """
    저장된 앙상블 모델들을 로드합니다.
    
    Args:
        ensemble_dir (str): 앙상블 모델이 저장된 디렉토리
        logger (Logger, optional): 로거 객체
        
    Returns:
        list: 로드된 PPO 에이전트 리스트
    """
    if logger is None:
        logger = logging.getLogger("PortfolioRL")
    
    # 앙상블 메타데이터 로드
    meta_file = os.path.join(ensemble_dir, "ensemble_meta.pkl")
    if not os.path.exists(meta_file):
        # 메타데이터 없음 - 디렉토리 탐색하여 추론
        # 이전 버전과의 호환성을 위해 agent_* 와 model_* 모두 확인
        agent_dirs = glob.glob(os.path.join(ensemble_dir, "agent_*"))
        model_dirs = glob.glob(os.path.join(ensemble_dir, "models/model_*"))
        
        if not agent_dirs and not model_dirs:
            logger.error(f"앙상블 디렉토리에 에이전트 또는 모델 폴더가 없음: {ensemble_dir}")
            return []
        
        # 새 형식과 이전 형식 중 더 많은 모델이 있는 것 사용
        if len(model_dirs) >= len(agent_dirs):
            ensemble_size = len(model_dirs)
            agent_paths = sorted(model_dirs)
            logger.warning(f"메타데이터 없음, 디렉토리 탐색 결과: {ensemble_size}개 모델 발견 (새 형식)")
        else:
            ensemble_size = len(agent_dirs)
            agent_paths = sorted(agent_dirs)
            logger.warning(f"메타데이터 없음, 디렉토리 탐색 결과: {ensemble_size}개 모델 발견 (이전 형식)")
    else:
        # 메타데이터 로드
        with open(meta_file, "rb") as f:
            meta_data = pickle.load(f)
        
        ensemble_size = meta_data["ensemble_size"]
        
        # 모델 경로 호환성 처리
        if "model_paths" in meta_data:
            agent_paths = meta_data["model_paths"]
        else:
            agent_paths = meta_data.get("agent_paths", [])
            
        # 시드 정보 출력 (있는 경우)
        if "seed_values" in meta_data:
            logger.info(f"앙상블 생성에 사용된 시드: {meta_data['seed_values']}")
        
        logger.info(f"앙상블 메타데이터 로드: {ensemble_size}개 모델")
    
    # 각 에이전트 로드
    ensemble_agents = []
    for i, agent_path in enumerate(agent_paths):
        logger.info(f"앙상블 모델 {i+1}/{ensemble_size} 로드 중: {agent_path}")
        
        # 모델 파일 경로 (최고 성능 모델 또는 최종 모델)
        best_model_file = os.path.join(agent_path, "best_model.pth")
        final_model_file = os.path.join(agent_path, "final_model.pth")
        
        model_file = (
            best_model_file if os.path.exists(best_model_file) else final_model_file
        )
        
        if not os.path.exists(model_file):
            logger.warning(f"모델 파일 없음: {model_file}, 건너뛰었습니다.")
            continue
        
        # 모델 파일에서 n_assets, n_features 추출
        try:
            checkpoint = torch.load(model_file, map_location=DEVICE)
            state_dict = checkpoint["model_state_dict"]
            
            # LSTM 레이어 가중치에서 차원 추출
            lstm_weight_key = next(k for k in state_dict.keys() if "lstm.weight" in k)
            lstm_shape = state_dict[lstm_weight_key].shape
            n_features = lstm_shape[1]  # LSTM input_size
            
            # Actor 출력 레이어에서 자산 수 추출
            actor_weight_key = next(k for k in state_dict.keys() if "actor_head.weight" in k)
            actor_shape = state_dict[actor_weight_key].shape
            n_assets = actor_shape[0] - 1  # 현금 제외
            
            logger.debug(f"모델 파일에서 추출: n_assets={n_assets}, n_features={n_features}")
        except Exception as e:
            logger.warning(f"모델 파일에서 차원 추출 실패: {e}, 기본값 사용")
            n_assets = 10  # 기본값
            n_features = 10  # 기본값
        
        # PPO 에이전트 생성 및 가중치 로드
        ppo_agent = PPO(
            n_assets=n_assets,
            n_features=n_features,
            model_path=agent_path,
            logger=logger,
        )
        
        success = ppo_agent.load_model(model_file)
        if success:
            ensemble_agents.append(ppo_agent)
            logger.info(f"앙상블 모델 {i+1} 로드 성공: {model_file}")
        else:
            logger.warning(f"앙상블 모델 {i+1} 로드 실패: {model_file}")
    
    # 결합 모델 로드 시도
    combined_model_path = os.path.join(ensemble_dir, "combined_model")
    if os.path.exists(combined_model_path):
        combined_model_file = os.path.join(combined_model_path, "ensemble_model.pth")
        if os.path.exists(combined_model_file):
            logger.info(f"앙상블 결합 모델 발견: {combined_model_file}")
    
    logger.info(f"앙상블 로드 완료: {len(ensemble_agents)}/{ensemble_size}개 모델 로드됨")
    return ensemble_agents

def plot_training_curves(reward_history, loss_history, validation_reward_history, save_dir=None):
    """
    학습 곡선을 그리고 저장합니다.
    
    Args:
        reward_history (list): 에피소드별 보상 이력
        loss_history (list): 타임스텝별 손실 이력
        validation_reward_history (list): 검증 보상 이력 (에피소드, 보상) 튜플 리스트
        save_dir (str, optional): 저장 디렉토리
    """
    if not reward_history:
        return
    
    plt.figure(figsize=(15, 10))
    
    # 보상 곡선
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(reward_history) + 1), reward_history, label='Training Reward')
    
    # 검증 보상 곡선 (있는 경우)
    if validation_reward_history:
        val_episodes, val_rewards = zip(*validation_reward_history)
        plt.plot(val_episodes, val_rewards, 'ro-', label='Validation Reward')
    
    plt.title('Training and Validation Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.legend()
    
    # 손실 곡선
    if loss_history:
        plt.subplot(2, 1, 2)
        plt.plot(loss_history)
        plt.title('PPO Loss')
        plt.xlabel('Update Steps')
        plt.ylabel('Loss')
        plt.grid(True)
    
    plt.tight_layout()
    
    # 저장
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'learning_curves.png'))
    
    plt.close() 