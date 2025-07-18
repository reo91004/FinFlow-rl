# core/backtester.py

import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import matplotlib.pyplot as plt
import torch
from datetime import datetime, timedelta
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional
from .system import ImmunePortfolioSystem
from xai import generate_dashboard, create_visualizations
from constant import *

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

        # 데이터 로드
        data_filename = f"market_data_{'_'.join(symbols)}_{train_start}_{test_end}.pkl"
        self.data_path = os.path.join(DATA_DIR, data_filename)

        if os.path.exists(self.data_path):
            print(f"기존 데이터 로드 중: {data_filename}")
            with open(self.data_path, "rb") as f:
                self.data = pickle.load(f)
        else:
            print("포괄적 시장 데이터 다운로드 중...")
            raw_data = yf.download(
                symbols, start="2007-12-01", end="2025-01-01", progress=True
            )

            # 다중 지표 데이터 처리
            self.data = self._process_comprehensive_data(raw_data, symbols)

            # 데이터 전처리는 _process_comprehensive_data에서 이미 완료됨
            print("데이터 전처리 완료")

            # 데이터 저장
            with open(self.data_path, "wb") as f:
                pickle.dump(self.data, f)
            print(f"포괄적 시장 데이터 저장 완료: {data_filename}")
            print(f"데이터 구조: {list(self.data.keys())}")

        # 데이터 분할
        self.train_data = self.data["prices"][train_start:train_end]
        self.test_data = self.data["prices"][test_start:test_end]
        self.train_features = self.data["features"][train_start:train_end]
        self.test_features = self.data["features"][test_start:test_end]

        # 기존 코드 호환성을 위한 추가 정리
        self.train_data = self._clean_data(self.train_data)
        self.test_data = self._clean_data(self.test_data)

    def _process_comprehensive_data(self, raw_data, symbols):
        """포괄적인 시장 데이터 처리"""
        print("다중 지표 데이터 처리 중...")

        # 기본 가격 데이터 추출
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

        # 추가 지표 계산
        features = self._calculate_technical_indicators(raw_data, symbols)

        # 데이터 정리
        prices = self._clean_data(prices)
        features = self._clean_data(features)

        return {"prices": prices, "features": features, "raw_data": raw_data}

    def _calculate_technical_indicators(self, raw_data, symbols):
        """기술적 지표 계산"""
        print("기술적 지표 계산 중...")

        features = {}

        for symbol in symbols:
            try:
                # 가격 데이터 추출
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

                # 기술적 지표 계산
                symbol_features = pd.DataFrame(index=close.index)

                # 1. 가격 기반 지표
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

                # 2. 모멘텀 지표
                symbol_features[f"{symbol}_rsi"] = self._calculate_rsi(close, 14)
                symbol_features[f"{symbol}_momentum"] = close / close.shift(10) - 1

                # 3. 볼린저 밴드
                bb_upper, bb_lower = self._calculate_bollinger_bands(close, 20, 2)
                symbol_features[f"{symbol}_bb_position"] = (close - bb_lower) / (
                    bb_upper - bb_lower
                )

                # 4. 거래량 지표
                symbol_features[f"{symbol}_volume_sma"] = volume.rolling(20).mean()
                symbol_features[f"{symbol}_volume_ratio"] = (
                    volume / symbol_features[f"{symbol}_volume_sma"]
                )

                # 5. 변동성 지표
                symbol_features[f"{symbol}_high_low_ratio"] = (high - low) / close
                symbol_features[f"{symbol}_price_range"] = (high - low) / close.rolling(
                    20
                ).mean()

                features[symbol] = symbol_features

            except Exception as e:
                print(f"[경고] {symbol} 기술적 지표 계산 중 오류 발생: {e}")
                continue

        # 전체 특성 데이터프레임 생성
        all_features = pd.concat(features.values(), axis=1)

        # 시장 전체 지표 추가
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
        print("시장 전체 지표 계산 중...")

        try:
            # 시장 전체 수익률 (동일 가중)
            return_cols = [col for col in features.columns if "_returns" in col]
            if return_cols:
                features["market_return"] = features[return_cols].mean(axis=1)
                features["market_volatility"] = features[return_cols].std(axis=1)
                # 상관계수 계산 개선
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

            # VIX 대용 지표 (변동성의 변동성)
            vol_cols = [col for col in features.columns if "_volatility" in col]
            if vol_cols:
                features["vix_proxy"] = (
                    features[vol_cols].mean(axis=1).rolling(10).std()
                )

            # 시장 스트레스 지수
            rsi_cols = [col for col in features.columns if "_rsi" in col]
            if rsi_cols:
                features["market_stress"] = features[rsi_cols].apply(
                    lambda x: (x < 30).sum() + (x > 70).sum(), axis=1
                )
            else:
                features["market_stress"] = 0

            # 결측값 처리
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
            # 기본값 설정
            features["market_return"] = 0.0
            features["market_volatility"] = 0.1
            features["market_correlation"] = 0.5
            features["vix_proxy"] = 0.1
            features["market_stress"] = 0.0

        return features

    def _clean_data(self, data):
        """데이터 정리"""
        print("데이터 전처리 중...")

        if data.isnull().values.any():
            print("결측값 발견, 전방향/후방향 채우기 적용")
            data = data.fillna(method="ffill").fillna(method="bfill")

        if data.isnull().values.any():
            print("잔여 결측값을 0으로 채움")
            data = data.fillna(0)

        if np.isinf(data.values).any():
            print("무한대 값 발견, 유한값으로 변환")
            data = data.replace([np.inf, -np.inf], 0)

        if data.isnull().values.any() or np.isinf(data.values).any():
            print("최종 데이터 정리 중...")
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
            logging_level=logging_level,
            output_dir=self.output_dir,
        )

        # 사전 훈련
        if use_learning_bcells:
            immune_system.pretrain_bcells(self.train_data, episodes=500)

        # 훈련 단계
        print("적응형 학습 진행 중...")
        train_returns = self.train_data.pct_change().dropna()
        portfolio_values = [1.0]

        for i in tqdm(range(len(train_returns)), desc="적응형 학습"):
            current_data = self.train_data.iloc[: i + 1]
            market_features = immune_system.extract_market_features(current_data)

            # 면역 반응 실행
            weights, response_type, bcell_decisions = immune_system.immune_response(
                market_features, training=True
            )

            portfolio_return = np.sum(weights * train_returns.iloc[i])
            portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))

            # 로깅
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

            # 학습 로직
            if len(portfolio_values) > 20:
                base_reward = portfolio_return * 100
                if portfolio_return < 0:
                    base_reward -= (portfolio_return * 150) ** 2

                running_max = np.maximum.accumulate(portfolio_values)
                drawdown = (portfolio_values[-1] - running_max[-1]) / (
                    running_max[-1] + 1e-8
                )
                if drawdown < 0:
                    base_reward += drawdown * 50

                # 지배적 위험 판단
                risk_features = market_features[:5]
                dominant_risk_idx = np.argmax(
                    np.abs(risk_features - np.mean(risk_features))
                )
                risk_map = {
                    0: "volatility",
                    1: "correlation",
                    2: "momentum",
                    3: "liquidity",
                    4: "macro",
                }
                dominant_risk = risk_map.get(dominant_risk_idx, "volatility")

                # B-세포 학습
                if use_learning_bcells:
                    for bcell in immune_system.bcells:
                        if hasattr(bcell, "last_strategy"):
                            # 전문 분야 보상 조정
                            is_specialist_today = bcell.is_my_specialty_situation(
                                market_features, immune_system.crisis_level
                            )

                            if is_specialist_today:
                                specialist_reward = base_reward * 2.0
                            else:
                                specialist_reward = base_reward * 0.8

                            final_reward = np.clip(specialist_reward, -2, 2)

                            # 경험 추가
                            bcell.add_experience(
                                market_features,
                                immune_system.crisis_level,
                                bcell.last_strategy.numpy(),
                                final_reward,
                            )

                            # 주기적 전문가 학습
                            if i % 20 == 0:
                                bcell.learn_from_specialized_experience()

                # 기억 세포 업데이트 (더 민감한 임계값)
                if immune_system.crisis_level > 0.15:
                    immune_system.update_memory(
                        market_features, weights, np.clip(base_reward, -1, 1)
                    )

        # 에피소드 종료
        if use_learning_bcells:
            for bcell in immune_system.bcells:
                bcell.end_episode()

        # 테스트 단계
        print("테스트 데이터 기반 성능 평가 진행 중...")
        test_returns = self.test_data.pct_change().dropna()
        test_portfolio_returns = []

        for i in tqdm(range(len(test_returns)), desc="성능 평가"):
            current_data = self.test_data.iloc[: i + 1]
            market_features = immune_system.extract_market_features(current_data)

            weights, response_type, bcell_decisions = immune_system.immune_response(
                market_features, training=False
            )

            portfolio_return = np.sum(weights * test_returns.iloc[i])
            test_portfolio_returns.append(portfolio_return)

            # 테스트 로깅 (로깅 레벨에 따라 조정)
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

        if return_model:
            return (
                pd.Series(test_portfolio_returns, index=test_returns.index),
                immune_system,
            )
        else:
            return pd.Series(test_portfolio_returns, index=test_returns.index)

    def analyze_bcell_expertise(self):
        """B-세포 전문성 분석"""

        if (
            not hasattr(self, "immune_system")
            or not self.immune_system.use_learning_bcells
        ):
            return {"error": "Learning-based system is not available."}

        print("B-세포 전문화 시스템 분석 중...")

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
        """통합 분석 결과 저장 (의사결정 분석 + 전문성 분석)"""

        if output_dir is None:
            output_dir = self.output_dir  # 전역 output_dir 사용

        if filename is None:
            filename = f"bipd_comprehensive_{start_date}_{end_date}"

        # 의사결정 분석
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

        # 전문성 분석
        expertise_analysis = self.analyze_bcell_expertise()

        # 통합 데이터 구조
        comprehensive_data = {
            "metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "analysis_period": {"start": start_date, "end": end_date},
                "system_type": (
                    "Learning-based"
                    if (
                        hasattr(self, "immune_system")
                        and self.immune_system.use_learning_bcells
                    )
                    else "규칙 기반"
                ),
            },
            "decision_analysis": decision_analysis,
            "expertise_analysis": expertise_analysis,
        }

        # JSON 저장
        json_path = os.path.join(output_dir, f"{filename}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(comprehensive_data, f, ensure_ascii=False, indent=2)

        # Markdown 저장
        md_content = self._generate_comprehensive_markdown(comprehensive_data)
        md_path = os.path.join(output_dir, f"{filename}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"Comprehensive analysis results saved:")
        print(f"  Directory: {output_dir}")
        print(f"  JSON: {os.path.basename(json_path)}")
        print(f"  Markdown: {os.path.basename(md_path)}")

        return json_path, md_path

    def _generate_comprehensive_markdown(self, comprehensive_data: Dict) -> str:
        """통합 분석 마크다운 생성"""

        metadata = comprehensive_data["metadata"]
        decision_data = comprehensive_data["decision_analysis"]
        expertise_data = comprehensive_data["expertise_analysis"]

        md_content = f"""# BIPD 시스템 통합 분석 보고서

## 분석 메타데이터
- 분석 시간: {metadata['analysis_timestamp']}
- 시스템 유형: {metadata['system_type']}
- 분석 기간: {metadata['analysis_period']['start']} ~ {metadata['analysis_period']['end']}

---

"""

        # 의사결정 분석 섹션
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

        # 전문성 분석 섹션
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
        """분석 결과 저장 (HTML 대시보드 포함)"""

        if not hasattr(self, "immune_system") or not hasattr(
            self.immune_system, "analyzer"
        ):
            print("Analysis system is not available.")
            return None, None, None

        try:
            # 기존 JSON/MD 파일 생성
            json_path, md_path = self.immune_system.analyzer.save_analysis_to_file(
                start_date, end_date, filename, output_dir=self.output_dir
            )

            # 분석 보고서 데이터 가져오기
            analysis_report = self.immune_system.analyzer.generate_analysis_report(
                start_date, end_date
            )

            # HTML 대시보드 생성
            dashboard_paths = generate_dashboard(
                analysis_report,
                output_dir=self.output_dir,
            )

            # 면역 시스템 시각화 생성
            immune_viz = create_visualizations(
                self,
                start_date,
                end_date,
                output_dir=self.output_dir,
            )

            print(f"Analysis results saved:")
            print(f"  JSON: {json_path}")
            print(f"  Markdown: {md_path}")
            print(f"  HTML Dashboard: {dashboard_paths['html_dashboard']}")
            print(
                f"\nYou can intuitively check T-Cell/B-Cell decision basis in HTML dashboard!"
            )
            print(
                f"Immune system response pattern visualization emphasizes differentiation from existing research!"
            )

            return json_path, md_path, dashboard_paths["html_dashboard"]

        except Exception as e:
            print(f"Analysis results save error: {e}")
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

        # JSON 저장
        json_path = os.path.join(self.output_dir, f"{filename}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(expertise_data, f, ensure_ascii=False, indent=2)

        # Markdown 저장
        md_content = self._generate_expertise_markdown(expertise_data)
        md_path = os.path.join(self.output_dir, f"{filename}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        print(f"전문성 분석 저장 완료:")
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

            # B-세포 신경망 저장
            for i, bcell in enumerate(immune_system.bcells):
                if hasattr(bcell, "strategy_network"):
                    network_path = os.path.join(
                        model_dir, f"bcell_{i}_{bcell.risk_type}.pth"
                    )
                    torch.save(bcell.strategy_network.state_dict(), network_path)

            # 시스템 상태 저장
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

            print(f"Learning-based model saved: {model_dir}")
            return model_dir
        else:
            model_path = os.path.join(output_dir, f"{filename}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(immune_system, f)
            print(f"규칙 기반 모델 저장 완료: {model_path}")
            return model_path

    def save_results(self, metrics_df, filename=None, output_dir=None):
        """결과 저장"""
        if output_dir is None:
            output_dir = self.output_dir  # 전역 output_dir 사용

        if filename is None:
            filename = "bipd_performance_metrics"

        # CSV 저장
        csv_path = os.path.join(output_dir, f"{filename}.csv")
        metrics_df.to_csv(csv_path, index=False)

        # 시각화
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

        print(f"Backtest results saved:")
        print(f"  Directory: {output_dir}")
        print(f"  CSV: {os.path.basename(csv_path)}")
        print(f"  Chart: {os.path.basename(plot_path)}")
        return csv_path, plot_path

    def run_multiple_backtests(
        self,
        n_runs=10,
        save_results=True,
        use_learning_bcells=True,
        logging_level="sample",
        base_seed=None,
    ):
        """다중 백테스트 실행"""
        all_metrics = []
        best_immune_system = None
        best_sharpe = -np.inf

        print(f"\n=== BIPD 시스템 다중 백테스트 ({n_runs}회) 실행 ===")
        if use_learning_bcells:
            print("시스템 유형: 적응형 신경망 기반 BIPD 모델")
        else:
            print("시스템 유형: 규칙 기반 레거시 BIPD 모델")

        # 시드 설정
        if base_seed is None:
            import time

            base_seed = int(time.time()) % 10000

        print(f"[설정] 기본 시드: {base_seed}")

        for run in range(n_runs):
            run_seed = base_seed + run * 1000  # 각 실행마다 다른 시드
            print(f"\n{run + 1}/{n_runs}번째 실행 (시드: {run_seed})")

            portfolio_returns, immune_system = self.backtest_single_run(
                seed=run_seed,
                return_model=True,
                use_learning_bcells=use_learning_bcells,
                logging_level=logging_level,
            )
            metrics = self.calculate_metrics(portfolio_returns)
            all_metrics.append(metrics)

            if metrics["Sharpe Ratio"] > best_sharpe:
                best_sharpe = metrics["Sharpe Ratio"]
                best_immune_system = immune_system

        metrics_df = pd.DataFrame(all_metrics)

        system_type = "Learning-based" if use_learning_bcells else "Rule-based"
        print(f"\n=== {system_type} 모델 성능 요약 ({n_runs}회 실행 평균) ===")
        print(f"총 수익률: {metrics_df['Total Return'].mean():.2%}")
        print(f"연평균 변동성: {metrics_df['Volatility'].mean():.3f}")
        print(f"최대 낙폭: {metrics_df['Max Drawdown'].mean():.2%}")
        print(f"샤프 지수: {metrics_df['Sharpe Ratio'].mean():.2f}")
        print(f"칼마 지수: {metrics_df['Calmar Ratio'].mean():.2f}")
        print(f"최종 자산: {metrics_df['Final Value'].mean():,.0f}원")

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
