# bipd/tests/run_stability_check.py

"""
BIPD 시스템 안정성 검증 스크립트

로그 분석에서 발견된 문제들(보상-성과 역상관, 상시 위기, 음수 가중치, Q/TD 폭주 등)에 대한
수정사항이 올바르게 적용되었는지 확인하는 검증 프로토콜을 실행합니다.
"""

import sys
import traceback
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_quick_verification():
    """빠른 검증 (기본 기능 테스트)"""
    print("=== BIPD 시스템 빠른 안정성 검증 ===")

    try:
        from tests.test_stability_verification import StabilityVerificationProtocol

        protocol = StabilityVerificationProtocol()

        # 선택적으로 일부 테스트만 실행
        print("1. 보상-성과 정렬 검증...")
        reward_result = protocol._verify_reward_performance_alignment()
        print(f"   결과: {reward_result.get('status', 'UNKNOWN')}")

        print("2. 위기 탐지 비율 검증...")
        crisis_result = protocol._verify_crisis_detection_rates()
        print(f"   결과: {crisis_result.get('status', 'UNKNOWN')}")

        print("3. 음수 가중치 제거 검증...")
        weight_result = protocol._verify_negative_weight_elimination()
        print(f"   결과: {weight_result.get('status', 'UNKNOWN')}")

        print("4. 로깅 정합성 검증...")
        logging_result = protocol._verify_logging_consistency()
        print(f"   결과: {logging_result.get('status', 'UNKNOWN')}")

        # 간단한 종합 평가
        results = [reward_result, crisis_result, weight_result, logging_result]
        passed = sum(1 for r in results if r.get("status") == "PASS")
        total = len(results)

        print(f"\n빠른 검증 결과: {passed}/{total} 통과 ({passed/total:.1%})")

        if passed == total:
            print("✓ 모든 핵심 기능이 정상적으로 작동합니다.")
            return True
        else:
            print("✗ 일부 기능에 문제가 있습니다. 전체 검증을 실행해 주세요.")
            return False

    except Exception as e:
        print(f"빠른 검증 중 오류 발생: {e}")
        traceback.print_exc()
        return False


def run_full_verification():
    """전체 검증 (모든 테스트 포함)"""
    print("=== BIPD 시스템 전체 안정성 검증 ===")

    try:
        from tests.test_stability_verification import StabilityVerificationProtocol

        protocol = StabilityVerificationProtocol()
        results = protocol.run_full_verification()

        # 전체 결과 확인
        passed_count = sum(1 for r in results.values() if r.get("status") == "PASS")
        total_count = len(results)

        if passed_count == total_count:
            print("\n🎉 전체 검증 완료: 모든 안정성 테스트를 통과했습니다!")
            return True
        else:
            print(f"\n⚠️ 전체 검증 완료: {passed_count}/{total_count} 통과")
            print("실패한 테스트를 확인하고 문제를 해결해 주세요.")
            return False

    except Exception as e:
        print(f"전체 검증 중 오류 발생: {e}")
        traceback.print_exc()
        return False


def check_imports():
    """필수 모듈 import 테스트"""
    print("필수 모듈 import 확인 중...")

    required_modules = [
        "numpy",
        "torch",
        "pandas",
        "sklearn",
        "config",
        "core.environment",
        "agents.tcell",
        "agents.bcell",
        "utils.logger",
        "utils.rolling_stats",
    ]

    failed_imports = []

    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            failed_imports.append(module)

    if failed_imports:
        print(f"\n{len(failed_imports)}개 모듈 import 실패. 의존성을 확인해 주세요.")
        return False
    else:
        print("\n모든 필수 모듈이 정상적으로 로드됩니다.")
        return True


def main():
    """메인 실행 함수"""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "quick"  # 기본값

    print(f"BIPD 안정성 검증 도구 (모드: {mode})")
    print("-" * 50)

    # import 확인
    if not check_imports():
        print("import 확인 실패. 프로그램을 종료합니다.")
        sys.exit(1)

    print("-" * 50)

    # 모드별 실행
    if mode == "quick" or mode == "q":
        success = run_quick_verification()
    elif mode == "full" or mode == "f":
        success = run_full_verification()
    elif mode == "import" or mode == "i":
        success = True  # import 확인만
    else:
        print(f"알 수 없는 모드: {mode}")
        print("사용법: python run_stability_check.py [quick|full|import]")
        sys.exit(1)

    print("-" * 50)

    if success:
        print("안정성 검증이 성공적으로 완료되었습니다.")
        sys.exit(0)
    else:
        print("안정성 검증에서 문제가 발견되었습니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()
