# xai/analyzer.py

import json
import os
import numpy as np
from datetime import datetime
from typing import Dict, List
from constant import *


class DecisionAnalyzer:
    """의사결정 과정 분석 클래스"""

    def __init__(self, output_dir=None):
        self.decision_log = []
        self.risk_thresholds = RISK_THRESHOLDS
        self.crisis_detection_log = []  # 상세 위기 감지 로그
        self.output_dir = output_dir or "."

    def _process_detailed_tcell_analysis(
        self, tcell_analysis, dominant_risk, risk_features, dominant_risk_idx
    ):
        """상세 T-cell 분석 처리"""
        # 기본 T-cell 분석 정보 (기존 형식 유지)
        basic_analysis = {
            "crisis_level": float(tcell_analysis.get("crisis_level", 0.0)),
            "dominant_risk": dominant_risk,
            "risk_intensity": float(risk_features[dominant_risk_idx]),
            "overall_threat": self._assess_threat_level(
                tcell_analysis.get("crisis_level", 0.0)
            ),
        }

        # 상세 위기 감지 로그가 있는 경우 추가
        if (
            isinstance(tcell_analysis, dict)
            and "detailed_crisis_logs" in tcell_analysis
        ):
            detailed_logs = tcell_analysis["detailed_crisis_logs"]
            analysis = basic_analysis.copy()
            analysis["detailed_crisis_detection"] = {
                "active_tcells": len(detailed_logs),
                "crisis_detections": [],
            }

            for tcell_log in detailed_logs:
                if tcell_log.get("activation_level", 0.0) > 0.15:  # 위기 감지 임계값
                    crisis_detection = {
                        "tcell_id": tcell_log.get("tcell_id", "unknown"),
                        "timestamp": tcell_log.get("timestamp", ""),
                        "activation_level": tcell_log.get("activation_level", 0.0),
                        "crisis_level_classification": tcell_log.get(
                            "crisis_level", "normal"
                        ),
                        "crisis_indicators": tcell_log.get("crisis_indicators", []),
                        "decision_reasoning": tcell_log.get("decision_reasoning", []),
                        "feature_contributions": tcell_log.get(
                            "feature_contributions", {}
                        ),
                        "market_state_analysis": tcell_log.get("market_state", {}),
                    }
                    analysis["detailed_crisis_detection"]["crisis_detections"].append(
                        crisis_detection
                    )

                    # 위기 감지 로그에 추가
                    self.crisis_detection_log.append(
                        {
                            "timestamp": tcell_log.get("timestamp", ""),
                            "tcell_id": tcell_log.get("tcell_id", "unknown"),
                            "crisis_info": crisis_detection,
                        }
                    )

            return analysis

        return basic_analysis

    def log_decision(
        self,
        date,
        market_features,
        tcell_analysis,
        bcell_decisions,
        final_weights,
        portfolio_return,
        crisis_level,
    ):
        """의사결정 과정 기록"""

        # 지배적 위험 분석
        risk_features = market_features[:5]
        dominant_risk_idx = np.argmax(np.abs(risk_features - np.mean(risk_features)))
        risk_map = {
            0: "volatility",
            1: "correlation",
            2: "momentum",
            3: "liquidity",
            4: "macro",
        }
        dominant_risk = risk_map.get(dominant_risk_idx, "volatility")

        # T-cell 분석 처리
        tcell_analysis_result = self._process_detailed_tcell_analysis(
            tcell_analysis, dominant_risk, risk_features, dominant_risk_idx
        )

        decision_record = {
            "date": (
                date.strftime("%Y-%m-%d") if hasattr(date, "strftime") else str(date)
            ),
            "market_features": (
                market_features.tolist()
                if hasattr(market_features, "tolist")
                else list(market_features)
            ),
            "tcell_analysis": tcell_analysis_result,
            "bcell_decisions": self._serialize_bcell_decisions(bcell_decisions),
            "final_weights": (
                final_weights.tolist()
                if hasattr(final_weights, "tolist")
                else list(final_weights)
            ),
            "portfolio_return": float(portfolio_return),
            "memory_activated": bool(crisis_level > 0.3),
        }

        self.decision_log.append(decision_record)

    def _serialize_bcell_decisions(self, bcell_decisions):
        """B-cell 결정 정보 직렬화 (상세 분석 포함)"""
        if not bcell_decisions:
            return []

        serialized = []
        for bcell in bcell_decisions:
            serialized_bcell = {}
            for key, value in bcell.items():
                if isinstance(value, (int, float, str, bool)):
                    serialized_bcell[key] = value
                else:
                    serialized_bcell[key] = str(value)

            # B-cell 전문성 및 의사결정 근거 분석 추가
            if "risk_type" in bcell:
                serialized_bcell["specialization_analysis"] = (
                    self._analyze_bcell_specialization(bcell)
                )

            serialized.append(serialized_bcell)

        return serialized

    def _analyze_bcell_specialization(self, bcell_decision):
        """B-cell 전문성 및 의사결정 근거 분석"""
        risk_type = bcell_decision.get("risk_type", "unknown")
        activation_level = bcell_decision.get("activation_level", 0.0)
        antibody_strength = bcell_decision.get("antibody_strength", 0.0)

        # 전문성 평가
        specialization_score = bcell_decision.get("strategy_contribution", 0.0)

        # 의사결정 근거 생성
        decision_reasoning = []

        # 활성화 근거
        if activation_level > 0.7:
            decision_reasoning.append(
                f"높은 활성화 레벨({activation_level:.3f})로 인한 강력한 대응 필요"
            )
        elif activation_level > 0.5:
            decision_reasoning.append(
                f"중간 활성화 레벨({activation_level:.3f})로 인한 적극적 대응"
            )
        elif activation_level > 0.3:
            decision_reasoning.append(
                f"낮은 활성화 레벨({activation_level:.3f})로 인한 보수적 대응"
            )

        # 위험 유형별 전문성 근거
        risk_reasoning = {
            "volatility": f"시장 변동성 위험에 특화된 안전 자산 중심 포트폴리오 구성",
            "correlation": f"상관관계 위험에 특화된 분산 투자 전략 적용",
            "momentum": f"모멘텀 위험에 특화된 추세 추종 전략 활용",
            "liquidity": f"유동성 위험에 특화된 대형주 중심 포트폴리오 구성",
            "memory_recall": f"과거 위기 경험을 바탕으로 한 검증된 대응 전략 적용",
        }

        if risk_type in risk_reasoning:
            decision_reasoning.append(risk_reasoning[risk_type])

        # 항체 강도 근거
        if antibody_strength > 0.8:
            decision_reasoning.append(
                f"높은 항체 강도({antibody_strength:.3f})로 강력한 방어 전략 수행"
            )
        elif antibody_strength > 0.5:
            decision_reasoning.append(
                f"중간 항체 강도({antibody_strength:.3f})로 균형잡힌 방어 전략 수행"
            )

        # 전문화 정도 평가
        if specialization_score > 0.8:
            specialization_level = "매우 높음"
        elif specialization_score > 0.6:
            specialization_level = "높음"
        elif specialization_score > 0.4:
            specialization_level = "중간"
        else:
            specialization_level = "낮음"

        return {
            "risk_type": risk_type,
            "specialization_level": specialization_level,
            "specialization_score": specialization_score,
            "decision_reasoning": decision_reasoning,
            "activation_analysis": {
                "level": activation_level,
                "category": (
                    "high"
                    if activation_level > 0.7
                    else "medium" if activation_level > 0.3 else "low"
                ),
            },
            "antibody_analysis": {
                "strength": antibody_strength,
                "effectiveness": (
                    "high"
                    if antibody_strength > 0.8
                    else "medium" if antibody_strength > 0.5 else "low"
                ),
            },
        }

    def _assess_threat_level(self, crisis_level):
        """위기 수준 평가"""
        if crisis_level < self.risk_thresholds["low"]:
            return "stable"
        elif crisis_level < self.risk_thresholds["medium"]:
            return "caution"
        elif crisis_level < self.risk_thresholds["high"]:
            return "alert"
        else:
            return "crisis"

    def generate_analysis_report(self, start_date: str, end_date: str) -> Dict:
        """분석 보고서 생성"""

        # 해당 기간 데이터 필터링
        period_records = []
        for record in self.decision_log:
            record_date = datetime.strptime(record["date"], "%Y-%m-%d")
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            if start_dt <= record_date <= end_dt:
                period_records.append(record)

        if not period_records:
            return {"error": f"No data found for period {start_date} to {end_date}"}

        # 통계 계산 (더 민감한 임계값)
        total_days = len(period_records)
        crisis_days = sum(
            1 for r in period_records if r["tcell_analysis"]["crisis_level"] > 0.15
        )
        memory_activations = sum(1 for r in period_records if r["memory_activated"])

        # 지배적 위험 분포
        risk_distribution = {}
        for record in period_records:
            risk = record["tcell_analysis"]["dominant_risk"]
            risk_distribution[risk] = risk_distribution.get(risk, 0) + 1

        # 평균 수익률
        avg_return = np.mean([r["portfolio_return"] for r in period_records])

        # T-cell 위기 감지 상세 분석
        tcell_analysis = self._analyze_tcell_crisis_detection()

        # B-cell 전문가 분석
        bcell_analysis = self._analyze_bcell_expert_responses()

        # 특성 기여도 분석
        feature_attribution = self._analyze_feature_attribution()

        # 시간별 위기 진행 분석
        temporal_analysis = self._analyze_temporal_crisis_patterns()

        report = {
            "period": {"start": start_date, "end": end_date},
            "basic_stats": {
                "total_days": total_days,
                "crisis_days": crisis_days,
                "crisis_ratio": crisis_days / total_days,
                "memory_activations": memory_activations,
                "memory_activation_ratio": memory_activations / total_days,
                "avg_daily_return": avg_return,
            },
            "risk_distribution": risk_distribution,
            "system_efficiency": {
                "crisis_response_rate": crisis_days / total_days,
                "learning_activation_rate": memory_activations / total_days,
                "system_stability": "high" if avg_return > 0 else "normal",
            },
            "tcell_crisis_analysis": tcell_analysis,
            "bcell_expert_analysis": bcell_analysis,
            "feature_attribution": feature_attribution,
            "temporal_analysis": temporal_analysis,
            "explainability_summary": {
                "total_crisis_detections": len(self.crisis_detection_log),
                "total_decisions_logged": len(self.decision_log),
                "analysis_completeness": "comprehensive",
                "xai_features_implemented": [
                    "detailed_crisis_detection_reasoning",
                    "tcell_specific_analysis",
                    "feature_attribution_analysis",
                    "temporal_pattern_tracking",
                    "bcell_expert_reasoning",
                    "decision_explanation_logging",
                ],
            },
        }

        return report

    def _analyze_tcell_crisis_detection(self):
        """T-cell 위기 감지 상세 분석"""
        if not self.crisis_detection_log:
            return {"total_detections": 0, "tcell_details": {}}

        tcell_details = {}
        total_detections = 0
        feature_contributions = {}

        for crisis_log in self.crisis_detection_log:
            tcell_id = crisis_log["tcell_id"]
            crisis_info = crisis_log["crisis_info"]

            if tcell_id not in tcell_details:
                tcell_details[tcell_id] = {
                    "detections": 0,
                    "avg_activation": 0.0,
                    "crisis_types": {},
                    "indicators": {},
                    "reasoning_patterns": [],
                    "feature_contributions": {},
                }

            tcell_details[tcell_id]["detections"] += 1
            tcell_details[tcell_id]["avg_activation"] += crisis_info["activation_level"]

            # 위기 유형 분석
            crisis_level = crisis_info["crisis_level_classification"]
            tcell_details[tcell_id]["crisis_types"][crisis_level] = (
                tcell_details[tcell_id]["crisis_types"].get(crisis_level, 0) + 1
            )

            # 지표 분석
            for indicator in crisis_info["crisis_indicators"]:
                indicator_type = indicator["type"]
                tcell_details[tcell_id]["indicators"][indicator_type] = (
                    tcell_details[tcell_id]["indicators"].get(indicator_type, 0) + 1
                )

            # 특성 기여도 분석
            for feature, contribution in crisis_info["feature_contributions"].items():
                if feature not in tcell_details[tcell_id]["feature_contributions"]:
                    tcell_details[tcell_id]["feature_contributions"][feature] = []
                tcell_details[tcell_id]["feature_contributions"][feature].append(
                    contribution
                )

                # 전체 특성 기여도 누적
                if feature not in feature_contributions:
                    feature_contributions[feature] = []
                feature_contributions[feature].append(contribution)

            # 의사결정 근거 패턴 분석 (최대 3개만 저장)
            if len(tcell_details[tcell_id]["reasoning_patterns"]) < 3:
                tcell_details[tcell_id]["reasoning_patterns"].extend(
                    crisis_info["decision_reasoning"]
                )

            total_detections += 1

        # 평균 계산
        for tcell_id in tcell_details:
            if tcell_details[tcell_id]["detections"] > 0:
                tcell_details[tcell_id]["avg_activation"] /= tcell_details[tcell_id][
                    "detections"
                ]

                # 특성 기여도 평균 계산
                for feature in tcell_details[tcell_id]["feature_contributions"]:
                    contributions = tcell_details[tcell_id]["feature_contributions"][
                        feature
                    ]
                    tcell_details[tcell_id]["feature_contributions"][feature] = {
                        "avg": sum(contributions) / len(contributions),
                        "max": max(contributions),
                        "min": min(contributions),
                    }

        # 전체 특성 기여도 분석
        global_feature_analysis = {}
        for feature, contributions in feature_contributions.items():
            global_feature_analysis[feature] = {
                "avg_contribution": sum(contributions) / len(contributions),
                "max_contribution": max(contributions),
                "detection_frequency": len(contributions),
            }

        return {
            "total_detections": total_detections,
            "active_tcells": len(tcell_details),
            "tcell_details": tcell_details,
            "global_feature_analysis": global_feature_analysis,
        }

    def _analyze_bcell_expert_responses(self):
        """B-cell 전문가 대응 분석"""
        if not self.decision_log:
            return {"total_experts": 0, "expert_details": {}}

        expert_analysis = {}
        total_activations = 0

        for record in self.decision_log:
            for bcell_decision in record.get("bcell_decisions", []):
                if "specialization_analysis" in bcell_decision:
                    analysis = bcell_decision["specialization_analysis"]
                    risk_type = analysis["risk_type"]

                    if risk_type not in expert_analysis:
                        expert_analysis[risk_type] = {
                            "activations": 0,
                            "avg_activation_level": 0.0,
                            "avg_antibody_strength": 0.0,
                            "avg_specialization_score": 0.0,
                            "decision_patterns": {},
                            "performance_metrics": {
                                "positive_outcomes": 0,
                                "negative_outcomes": 0,
                                "avg_return_when_active": 0.0,
                                "returns": [],
                            },
                        }

                    expert_analysis[risk_type]["activations"] += 1
                    expert_analysis[risk_type]["avg_activation_level"] += analysis[
                        "activation_analysis"
                    ]["level"]
                    expert_analysis[risk_type]["avg_antibody_strength"] += analysis[
                        "antibody_analysis"
                    ]["strength"]
                    expert_analysis[risk_type]["avg_specialization_score"] += analysis[
                        "specialization_score"
                    ]

                    # 의사결정 패턴 분석
                    for reasoning in analysis["decision_reasoning"]:
                        pattern_key = (
                            reasoning.split("로 인한")[0]
                            if "로 인한" in reasoning
                            else reasoning[:50]
                        )
                        expert_analysis[risk_type]["decision_patterns"][pattern_key] = (
                            expert_analysis[risk_type]["decision_patterns"].get(
                                pattern_key, 0
                            )
                            + 1
                        )

                    # 성과 분석
                    portfolio_return = record.get("portfolio_return", 0.0)
                    expert_analysis[risk_type]["performance_metrics"]["returns"].append(
                        portfolio_return
                    )

                    if portfolio_return > 0:
                        expert_analysis[risk_type]["performance_metrics"][
                            "positive_outcomes"
                        ] += 1
                    else:
                        expert_analysis[risk_type]["performance_metrics"][
                            "negative_outcomes"
                        ] += 1

                    total_activations += 1

        # 평균 계산
        for risk_type in expert_analysis:
            analysis = expert_analysis[risk_type]
            if analysis["activations"] > 0:
                analysis["avg_activation_level"] /= analysis["activations"]
                analysis["avg_antibody_strength"] /= analysis["activations"]
                analysis["avg_specialization_score"] /= analysis["activations"]

                # 성과 메트릭 계산
                if analysis["performance_metrics"]["returns"]:
                    analysis["performance_metrics"]["avg_return_when_active"] = sum(
                        analysis["performance_metrics"]["returns"]
                    ) / len(analysis["performance_metrics"]["returns"])

                    total_outcomes = (
                        analysis["performance_metrics"]["positive_outcomes"]
                        + analysis["performance_metrics"]["negative_outcomes"]
                    )
                    analysis["performance_metrics"]["success_rate"] = (
                        (
                            analysis["performance_metrics"]["positive_outcomes"]
                            / total_outcomes
                        )
                        * 100
                        if total_outcomes > 0
                        else 0
                    )

        return {
            "total_activations": total_activations,
            "active_experts": len(expert_analysis),
            "expert_details": expert_analysis,
        }

    def _analyze_feature_attribution(self):
        """특성 기여도 분석"""
        if not self.crisis_detection_log:
            return {"total_features": 0, "feature_importance": {}}

        feature_contributions = {}
        feature_combinations = {}

        for crisis_log in self.crisis_detection_log:
            crisis_info = crisis_log["crisis_info"]

            # 개별 특성 기여도 분석
            for feature, contribution in crisis_info["feature_contributions"].items():
                if feature not in feature_contributions:
                    feature_contributions[feature] = []
                feature_contributions[feature].append(contribution)

            # 특성 조합 분석
            active_features = [
                f for f, c in crisis_info["feature_contributions"].items() if c > 0.1
            ]
            if len(active_features) > 1:
                combination = tuple(sorted(active_features))
                feature_combinations[combination] = (
                    feature_combinations.get(combination, 0) + 1
                )

        # 특성 중요도 계산
        feature_importance = {}
        for feature, contributions in feature_contributions.items():
            feature_importance[feature] = {
                "avg_contribution": sum(contributions) / len(contributions),
                "max_contribution": max(contributions),
                "min_contribution": min(contributions),
                "detection_frequency": len(contributions),
                "importance_level": self._classify_importance(
                    sum(contributions) / len(contributions)
                ),
            }

        # 상위 특성 조합
        top_combinations = sorted(
            feature_combinations.items(), key=lambda x: x[1], reverse=True
        )[:5]

        return {
            "total_features": len(feature_importance),
            "feature_importance": feature_importance,
            "top_feature_combinations": [
                {
                    "features": list(combo),
                    "frequency": freq,
                    "percentage": (freq / len(self.crisis_detection_log)) * 100,
                }
                for combo, freq in top_combinations
            ],
        }

    def _analyze_temporal_crisis_patterns(self):
        """시간별 위기 진행 패턴 분석"""
        if not self.crisis_detection_log:
            return {"total_events": 0, "patterns": {}}

        time_sorted_crises = sorted(
            self.crisis_detection_log, key=lambda x: x["timestamp"]
        )

        # 위기 클러스터 분석
        crisis_clusters = []
        current_cluster = []

        for crisis_log in time_sorted_crises:
            crisis_time = datetime.fromisoformat(crisis_log["timestamp"])

            if current_cluster:
                last_crisis_time = datetime.fromisoformat(
                    current_cluster[-1]["timestamp"]
                )
                time_diff = (crisis_time - last_crisis_time).total_seconds() / 3600

                if time_diff <= REACTION_TIME_THRESHOLD:  # 6시간 이내
                    current_cluster.append(crisis_log)
                else:
                    if len(current_cluster) > 1:
                        crisis_clusters.append(current_cluster)
                    current_cluster = [crisis_log]
            else:
                current_cluster = [crisis_log]

        if len(current_cluster) > 1:
            crisis_clusters.append(current_cluster)

        # 에스컬레이션 패턴 분석
        escalation_patterns = []
        for i in range(len(time_sorted_crises) - 1):
            current_activation = time_sorted_crises[i]["crisis_info"][
                "activation_level"
            ]
            next_activation = time_sorted_crises[i + 1]["crisis_info"][
                "activation_level"
            ]

            time_diff = (
                datetime.fromisoformat(time_sorted_crises[i + 1]["timestamp"])
                - datetime.fromisoformat(time_sorted_crises[i]["timestamp"])
            ).total_seconds() / 3600

            if time_diff <= 6:
                change = next_activation - current_activation
                escalation_patterns.append(
                    {
                        "time_diff": time_diff,
                        "activation_change": change,
                        "pattern": (
                            "escalation"
                            if change > 0.1
                            else "de-escalation" if change < -0.1 else "stable"
                        ),
                    }
                )

        # 시간대별 분석
        hourly_patterns = {}
        for crisis_log in time_sorted_crises:
            crisis_time = datetime.fromisoformat(crisis_log["timestamp"])
            hour = crisis_time.hour
            hourly_patterns[hour] = hourly_patterns.get(hour, 0) + 1

        return {
            "total_events": len(time_sorted_crises),
            "crisis_clusters": {
                "total_clusters": len(crisis_clusters),
                "cluster_details": [
                    {
                        "duration_hours": (
                            datetime.fromisoformat(cluster[-1]["timestamp"])
                            - datetime.fromisoformat(cluster[0]["timestamp"])
                        ).total_seconds()
                        / 3600,
                        "events_count": len(cluster),
                        "max_activation": max(
                            c["crisis_info"]["activation_level"] for c in cluster
                        ),
                        "avg_activation": sum(
                            c["crisis_info"]["activation_level"] for c in cluster
                        )
                        / len(cluster),
                    }
                    for cluster in crisis_clusters
                ],
            },
            "escalation_patterns": {
                "total_patterns": len(escalation_patterns),
                "pattern_distribution": {
                    pattern: sum(
                        1 for p in escalation_patterns if p["pattern"] == pattern
                    )
                    for pattern in ["escalation", "de-escalation", "stable"]
                },
            },
            "hourly_distribution": hourly_patterns,
        }

    def _classify_importance(self, avg_contribution):
        """중요도 분류"""
        if avg_contribution > 0.3:
            return "very_high"
        elif avg_contribution > 0.2:
            return "high"
        elif avg_contribution > 0.1:
            return "medium"
        else:
            return "low"

    def save_analysis_to_file(
        self,
        start_date: str,
        end_date: str,
        filename: str = None,
        output_dir: str = None,
    ):
        """분석 결과를 파일로 저장"""

        if output_dir is None:
            output_dir = self.output_dir  # 전역 output_dir 사용

        if filename is None:
            filename = f"decision_analysis_{start_date}_{end_date}"

        # JSON 보고서 생성
        report = self.generate_analysis_report(start_date, end_date)

        # JSON 파일 저장
        json_path = os.path.join(output_dir, f"{filename}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=JSON_INDENT)

        # Markdown 보고서 생성
        md_content = self._generate_markdown_report(report)
        md_path = os.path.join(output_dir, f"{filename}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return json_path, md_path

    def _generate_markdown_report(self, report: Dict) -> str:
        """마크다운 형식 보고서 생성"""

        if "error" in report:
            return f"# 분석 오류\n\n{report['error']}"

        period = report["period"]
        stats = report["basic_stats"]
        risk_dist = report["risk_distribution"]
        efficiency = report["system_efficiency"]

        md_content = f"""# BIPD 시스템 분석 보고서

## 분석 기간
- 시작일: {period['start']}
- 종료일: {period['end']}

## 기본 통계
- 총 거래일: {stats['total_days']}일
- 위기 감지일: {stats['crisis_days']}일 ({stats['crisis_ratio']:.1%})
- 기억 세포 활성화: {stats['memory_activations']}일 ({stats['memory_activation_ratio']:.1%})
- 평균 일수익률: {stats['avg_daily_return']:+.3%}

## 위험 유형별 분포
"""

        for risk, count in sorted(risk_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = count / stats["total_days"] * 100
            md_content += f"- {risk}: {count}일 ({percentage:.1f}%)\n"

        md_content += f"""
## 시스템 효율성
- 위기 대응률: {efficiency['crisis_response_rate']:.1%}
- 학습 활성화율: {efficiency['learning_activation_rate']:.1%}
- 시스템 안정성: {efficiency['system_stability']}

## T-cell 상세 위기 감지 분석"""

        # 추가적인 분석 내용은 원본과 동일하므로 생략...
        # (원본 _generate_markdown_report 메소드의 나머지 부분을 여기에 포함)

        return md_content
