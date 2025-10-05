#!/bin/bash

# scripts/run.sh
# FinFlow-RL 통합 실행 스크립트
#
# SAC 기본 학습, IRT 학습, 비교 실험, Ablation study를 하나의 스크립트로 통합
#
# 사용법:
#   ./run.sh                              # Interactive menu
#   ./run.sh --mode sac                   # SAC 기본 학습
#   ./run.sh --mode irt                   # IRT 학습
#   ./run.sh --mode experiments           # 비교 실험 (3개)
#   ./run.sh --mode ablation              # Ablation study (5개)
#   ./run.sh --mode all                   # 전체 실험
#   ./run.sh --mode quick                 # 빠른 테스트 (10 episodes)
#   ./run.sh --mode experiments --episodes 100

set -e  # 에러 발생 시 중단

# ============================================================================
# 설정
# ============================================================================

# 기본값
DEFAULT_EPISODES=200
QUICK_EPISODES=10
EPISODES=${EPISODES:-$DEFAULT_EPISODES}
OUTPUT_BASE="logs"

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# 유틸리티 함수
# ============================================================================

function print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
    echo ""
}

function print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

function print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

function print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

function print_error() {
    echo -e "${RED}❌ $1${NC}"
}

function show_help() {
    cat << EOF
FinFlow-RL 통합 실행 스크립트

사용법:
  $0 [OPTIONS]

옵션:
  --mode MODE       실행 모드 선택
                    - sac: SAC 기본 학습
                    - irt: IRT 기본 학습
                    - experiments: 비교 실험 (Baseline/IRT Single/IRT Multi)
                    - ablation: Ablation study (5개 configuration)
                    - all: 전체 실험 (experiments + ablation)
                    - quick: 빠른 테스트 (10 episodes)

  --episodes N      Episode 수 설정 (기본값: 200)
  --output DIR      출력 디렉토리 (기본값: logs)
  --help            이 도움말 출력

예시:
  $0                                    # Interactive menu
  $0 --mode sac                         # SAC 기본 학습
  $0 --mode experiments --episodes 100  # 비교 실험 (100 episodes)
  $0 --mode quick                       # 빠른 테스트

EOF
}

# ============================================================================
# 실행 함수들
# ============================================================================

function train_sac() {
    local episodes=${1:-$EPISODES}
    local output_dir="${OUTPUT_BASE}/sac_$(date +%Y%m%d_%H%M%S)"

    print_header "SAC 기본 학습"

    print_info "설정:"
    echo "  Model: SAC (Baseline)"
    echo "  Episodes: $episodes"
    echo "  Output: $output_dir"
    echo ""

    python scripts/train.py \
        --model sac \
        --episodes "$episodes" \
        --output "$output_dir"

    print_success "SAC 학습 완료"
    print_info "결과 위치: $output_dir"
}

function train_irt() {
    local episodes=${1:-$EPISODES}
    local output_dir="${OUTPUT_BASE}/irt_$(date +%Y%m%d_%H%M%S)"

    print_header "IRT 학습"

    print_info "설정:"
    echo "  Model: SAC + IRT Policy"
    echo "  Reward: Multi-Objective (Turnover + Diversity + Drawdown)"
    echo "  Episodes: $episodes"
    echo "  Output: $output_dir"
    echo ""

    python scripts/train_irt.py \
        --mode both \
        --episodes "$episodes" \
        --output "$output_dir"

    print_success "IRT 학습 완료"
    print_info "결과 위치: $output_dir"
}

function run_experiments() {
    local episodes=${1:-$EPISODES}
    local output_dir="${OUTPUT_BASE}/comparison_$(date +%Y%m%d_%H%M%S)"

    print_header "비교 실험 (Baseline vs IRT)"

    print_info "설정:"
    echo "  실험 개수: 3"
    echo "  Episodes per experiment: $episodes"
    echo "  Output: $output_dir"
    echo ""
    print_info "실험 목록:"
    echo "  1. Baseline SAC (원본 reward)"
    echo "  2. IRT Single-Objective (공정 비교)"
    echo "  3. IRT Multi-Objective (full system)"
    echo ""

    # Experiment 1: Baseline SAC
    print_header "Experiment 1/3: Baseline SAC"
    python scripts/train.py \
        --model sac \
        --episodes "$episodes" \
        --output "$output_dir/1_baseline_sac"
    print_success "Experiment 1 완료"
    echo ""

    # Experiment 2: IRT Single-Objective
    print_header "Experiment 2/3: IRT Single-Objective"
    python scripts/train_irt.py \
        --mode both \
        --episodes "$episodes" \
        --no-multiobjective \
        --output "$output_dir/2_irt_single"
    print_success "Experiment 2 완료"
    echo ""

    # Experiment 3: IRT Multi-Objective
    print_header "Experiment 3/3: IRT Multi-Objective"
    python scripts/train_irt.py \
        --mode both \
        --episodes "$episodes" \
        --output "$output_dir/3_irt_multi"
    print_success "Experiment 3 완료"
    echo ""

    # 결과 요약
    print_header "비교 실험 완료!"
    print_success "모든 실험이 성공적으로 완료되었습니다"
    echo ""
    print_info "결과 위치: $output_dir"
    echo ""
    echo "다음 단계:"
    echo "  1. 로그 확인: ls -R $output_dir"
    echo "  2. Tensorboard: tensorboard --logdir $output_dir"
    echo "  3. 시각화: 각 실험 폴더의 evaluation_plots/ 참고"
    echo "  4. JSON 결과: 각 실험 폴더의 evaluation_*.json 참고"
    echo ""
}

function run_ablation() {
    local episodes=${1:-$EPISODES}
    local output_dir="${OUTPUT_BASE}/ablation_$(date +%Y%m%d_%H%M%S)"

    print_header "Ablation Study: 보상 구성요소 분석"

    print_info "설정:"
    echo "  Configuration 개수: 5"
    echo "  Episodes per config: $episodes"
    echo "  Output: $output_dir"
    echo ""
    print_info "Configuration 목록:"
    echo "  1. Base only (ln(V_t/V_{t-1}))"
    echo "  2. Base + Turnover"
    echo "  3. Base + Diversity"
    echo "  4. Base + Drawdown"
    echo "  5. Full (Base + All)"
    echo ""

    # Config 1: Base only
    print_header "Config 1/5: Base only"
    python scripts/train_irt.py \
        --mode both \
        --episodes "$episodes" \
        --no-turnover \
        --no-diversity \
        --no-drawdown \
        --output "$output_dir/1_base_only"
    print_success "Config 1 완료"
    echo ""

    # Config 2: Base + Turnover
    print_header "Config 2/5: Base + Turnover"
    python scripts/train_irt.py \
        --mode both \
        --episodes "$episodes" \
        --no-diversity \
        --no-drawdown \
        --output "$output_dir/2_base_turnover"
    print_success "Config 2 완료"
    echo ""

    # Config 3: Base + Diversity
    print_header "Config 3/5: Base + Diversity"
    python scripts/train_irt.py \
        --mode both \
        --episodes "$episodes" \
        --no-turnover \
        --no-drawdown \
        --output "$output_dir/3_base_diversity"
    print_success "Config 3 완료"
    echo ""

    # Config 4: Base + Drawdown
    print_header "Config 4/5: Base + Drawdown"
    python scripts/train_irt.py \
        --mode both \
        --episodes "$episodes" \
        --no-turnover \
        --no-diversity \
        --output "$output_dir/4_base_drawdown"
    print_success "Config 4 완료"
    echo ""

    # Config 5: Full
    print_header "Config 5/5: Full (Base + All)"
    python scripts/train_irt.py \
        --mode both \
        --episodes "$episodes" \
        --output "$output_dir/5_full"
    print_success "Config 5 완료"
    echo ""

    # 결과 분석
    print_header "결과 분석 중..."

    python - <<EOF
import json
from pathlib import Path

results_dir = Path("$output_dir")

configs = [
    "1_base_only",
    "2_base_turnover",
    "3_base_diversity",
    "4_base_drawdown",
    "5_full"
]

print("=" * 80)
print("Ablation Study Results")
print("=" * 80)
print()
print(f"{'Configuration':<25} {'Sharpe':>10} {'Turnover':>10} {'Entropy':>10}")
print("-" * 80)

for config in configs:
    config_dir = results_dir / config / "irt"

    # 최신 실험 폴더 찾기
    subdirs = sorted([d for d in config_dir.glob("*") if d.is_dir()])
    if not subdirs:
        print(f"{config:<25} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
        continue

    latest = subdirs[-1]

    # evaluation_insights.json 읽기
    json_file = latest / "evaluation_insights.json"
    if not json_file.exists():
        json_file = latest / "evaluation_results.json"

    if json_file.exists():
        with open(json_file) as f:
            data = json.load(f)

        # Sharpe ratio
        sharpe = data.get("summary", {}).get("sharpe_ratio",
                 data.get("metrics", {}).get("sharpe_ratio", 0))

        # Turnover
        turnover = data.get("risk_metrics", {}).get("avg_turnover",
                   data.get("metrics", {}).get("avg_turnover", 0))

        # Entropy
        entropy = data.get("prototype_analysis", {}).get("avg_entropy",
                  data.get("metrics", {}).get("portfolio_entropy", 0))

        print(f"{config:<25} {sharpe:>10.3f} {turnover*100:>9.1f}% {entropy:>10.3f}")
    else:
        print(f"{config:<25} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

print()
print("=" * 80)
print()
EOF

    print_success "Ablation Study 완료"
    echo ""
    print_info "결과 위치: $output_dir"
    echo ""
    echo "논문 작성 가이드:"
    echo "  1. 위 표를 논문 Table에 삽입"
    echo "  2. Config 1 → 5로의 성능 향상 분석"
    echo "  3. 각 component의 독립적 기여도 vs 결합 효과 논의"
    echo "  4. Full model이 개별 component 합보다 높은 이유 설명 (시너지 효과)"
    echo ""
}

function run_all() {
    local episodes=${1:-$EPISODES}

    print_header "전체 실험 실행"
    print_info "비교 실험 + Ablation Study를 순차적으로 실행합니다"
    echo ""

    # 비교 실험
    run_experiments "$episodes"

    echo ""
    print_info "비교 실험 완료. Ablation Study 시작..."
    echo ""

    # Ablation study
    run_ablation "$episodes"

    print_header "모든 실험 완료!"
    print_success "비교 실험 (3개) + Ablation Study (5개) = 총 8개 실험 완료"
}

function run_quick() {
    print_header "빠른 테스트 모드"
    print_warning "이 모드는 테스트용입니다 (10 episodes)"
    echo ""

    local choice
    echo "테스트할 모드 선택:"
    echo "  1) SAC 기본"
    echo "  2) IRT"
    echo "  3) 비교 실험 (3개)"
    echo ""
    read -p "선택 (1-3): " choice

    case $choice in
        1)
            train_sac $QUICK_EPISODES
            ;;
        2)
            train_irt $QUICK_EPISODES
            ;;
        3)
            run_experiments $QUICK_EPISODES
            ;;
        *)
            print_error "잘못된 선택입니다"
            exit 1
            ;;
    esac
}

# ============================================================================
# Interactive Menu
# ============================================================================

function show_menu() {
    clear
    cat << EOF
===========================================
   FinFlow-RL 실험 자동화 스크립트
===========================================

실행 모드 선택:

  1) SAC 기본 학습
     - Baseline SAC (원본 reward)
     - Episodes: $EPISODES

  2) IRT 기본 학습
     - SAC + IRT Policy
     - Multi-Objective Reward
     - Episodes: $EPISODES

  3) 비교 실험 (3개)
     - Baseline SAC
     - IRT Single-Objective
     - IRT Multi-Objective
     - Episodes per experiment: $EPISODES

  4) Ablation Study (5개)
     - Base only
     - Base + Turnover
     - Base + Diversity
     - Base + Drawdown
     - Full
     - Episodes per config: $EPISODES

  5) 전체 실험
     - 비교 실험 + Ablation Study
     - 총 8개 실험
     - Episodes: $EPISODES

  6) 빠른 테스트
     - 10 episodes로 빠르게 테스트

  7) 설정 변경
     - Episode 수 조정

  0) 종료

===========================================
EOF

    read -p "선택 (0-7): " choice
    echo ""

    case $choice in
        1)
            train_sac
            ;;
        2)
            train_irt
            ;;
        3)
            run_experiments
            ;;
        4)
            run_ablation
            ;;
        5)
            run_all
            ;;
        6)
            run_quick
            ;;
        7)
            read -p "Episode 수 입력 (현재: $EPISODES): " new_episodes
            if [[ "$new_episodes" =~ ^[0-9]+$ ]]; then
                EPISODES=$new_episodes
                print_success "Episode 수가 $EPISODES로 변경되었습니다"
                sleep 1
                show_menu
            else
                print_error "유효한 숫자를 입력하세요"
                sleep 2
                show_menu
            fi
            ;;
        0)
            print_info "종료합니다"
            exit 0
            ;;
        *)
            print_error "잘못된 선택입니다"
            sleep 2
            show_menu
            ;;
    esac
}

# ============================================================================
# CLI Argument Parsing
# ============================================================================

MODE=""
CUSTOM_OUTPUT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --output)
            CUSTOM_OUTPUT="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "알 수 없는 옵션: $1"
            show_help
            exit 1
            ;;
    esac
done

# ============================================================================
# Main Execution
# ============================================================================

# Custom output directory 설정
if [ -n "$CUSTOM_OUTPUT" ]; then
    OUTPUT_BASE="$CUSTOM_OUTPUT"
fi

# CLI mode vs Interactive mode
if [ -z "$MODE" ]; then
    # No arguments: Interactive menu
    show_menu
else
    # CLI mode
    case $MODE in
        sac)
            train_sac
            ;;
        irt)
            train_irt
            ;;
        experiments)
            run_experiments
            ;;
        ablation)
            run_ablation
            ;;
        all)
            run_all
            ;;
        quick)
            run_quick
            ;;
        *)
            print_error "알 수 없는 모드: $MODE"
            show_help
            exit 1
            ;;
    esac
fi

print_success "실행 완료"
