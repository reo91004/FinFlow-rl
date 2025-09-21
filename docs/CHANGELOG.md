# Changelog

모든 주요 변경사항이 이 파일에 기록됩니다.

형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.0.0/)를 기반으로 하며,
이 프로젝트는 [Semantic Versioning](https://semver.org/spec/v2.0.0.html)을 따릅니다.

## [Unreleased]

### 예정
- Transformer 기반 시계열 예측 통합
- 실시간 거래 시스템 (Alpaca, IB API)
- 연합 학습 (Federated Learning) 지원
- 멀티 자산 클래스 확장 (채권, 원자재, 암호화폐)
- WebSocket 기반 실시간 대시보드

---

## [2.0.0] - 2025-01-22

### 🎉 Major Release: BIPD (Biologically-Inspired Portfolio Defense) 2.0

#### ✨ Added
- **B-Cell 통합 에이전트**: IQL + Distributional SAC + CQL 통합
- **SafeTensors 지원**: 안전한 모델 직렬화
- **Soft MoE (Mixture of Experts)**: 5개 전문 전략
  - Volatility Expert
  - Correlation Expert
  - Momentum Expert
  - Defensive Expert
  - Growth Expert
- **XAI 시스템**: SHAP 기반 설명 가능한 AI
- **의사결정 카드**: 각 거래 결정의 상세 설명
- **현실적 백테스팅**: 거래 비용, 슬리피지, 세금 모델링
- **ARCHITECTURE.md**: 전체 시스템 아키텍처 문서
- **docs/ 폴더**: 상세 문서 6종
  - API.md
  - TRAINING.md
  - EVALUATION.md
  - XAI.md
  - CONFIGURATION.md
  - CHANGELOG.md

#### 🔄 Changed
- **SAC 통합**: `src/core/sac.py` 제거, B-Cell에 통합
- **백테스트 통합**: `backtest.py` 기능을 `evaluate.py`에 통합
- **오프라인 데이터 재사용**: 캐싱 메커니즘 개선
- **파라미터 최적화**:
  - `alpha_init`: 0.2 → 0.75
  - `cql_alpha`: 0.01 → 5.0-10.0
  - `memory_capacity`: 500 → 50000
- **로깅 시스템**: FinFlowLogger로 통합

#### 🐛 Fixed
- 오프라인 데이터 매번 재수집 문제 해결
- 평가 결과 잘못된 디렉토리 저장 문제 수정
- OfflineDataset 클래스 이름 충돌 해결
- LiveTradingSystem 누락 참조 제거
- Memory Cell 저장 누락 해결

#### 🗑️ Removed
- `src/core/sac.py`: B-Cell에 통합
- `dashboard.py`: 별도 대시보드 제거
- `live_trading.py`: 향후 재구현 예정
- 중복 네트워크 정의 제거

---

## [1.5.0] - 2024-12-15

### Added
- T-Cell 위기 감지 시스템
- Memory Cell k-NN 경험 재활용
- Isolation Forest 기반 이상치 탐지
- CVaR 제약 강화
- 안정성 모니터링 시스템

### Changed
- IQL expectile 파라미터 조정 (0.5 → 0.7)
- 버퍼 크기 증가 (10000 → 50000)

### Fixed
- Q값 폭발 문제 해결
- 엔트로피 급락 방지

---

## [1.0.0] - 2024-10-01

### 🎉 Initial Release

#### Added
- IQL 오프라인 사전학습
- Distributional SAC 온라인 학습
- 기본 포트폴리오 환경
- yfinance 데이터 로더
- 기본 평가 메트릭
- TensorBoard 연동

#### Features
- 30개 자산 동시 관리
- T+1 결제 시뮬레이션
- Differential Sharpe 최적화
- 기본 거래 비용 모델

---

## [0.9.0] - 2024-08-15 (Beta)

### Added
- 프로토타입 구현
- 기본 SAC 알고리즘
- 단순 환경 구현

### Known Issues
- 학습 불안정
- 메모리 누수
- 제한된 자산 수 (10개)

---

## 버전 명명 규칙

- **Major (X.0.0)**: 호환성이 깨지는 주요 변경
- **Minor (0.X.0)**: 새로운 기능 추가
- **Patch (0.0.X)**: 버그 수정

## 마이그레이션 가이드

### 1.x → 2.0 마이그레이션

#### 코드 변경 필요

```python
# 이전 (1.x)
from src.core.sac import DistributionalSAC
agent = DistributionalSAC(state_dim, action_dim)

# 새로운 (2.0)
from src.agents.b_cell import BCell
agent = BCell(state_dim, action_dim, config)
```

#### 설정 파일 변경

```yaml
# 이전 (1.x)
sac:
  alpha: 0.2
  cql_weight: 0.01

# 새로운 (2.0)
bcell:
  alpha_init: 0.75
  cql_alpha_start: 5.0
```

#### 체크포인트 호환성

```python
# 이전 체크포인트 변환
from src.utils.migration import convert_checkpoint_v1_to_v2

old_checkpoint = torch.load('checkpoint_v1.pt')
new_checkpoint = convert_checkpoint_v1_to_v2(old_checkpoint)
torch.save(new_checkpoint, 'checkpoint_v2.safetensors')
```

---

## 기여자

### 주요 기여자
- FinFlow Team - 초기 구현 및 아키텍처
- Contributors - 버그 수정 및 기능 개선

### 기여 방법
1. Fork 저장소
2. Feature 브랜치 생성 (`git checkout -b feature/AmazingFeature`)
3. 변경사항 커밋 (`git commit -m 'Add: 새로운 기능'`)
4. 브랜치 푸시 (`git push origin feature/AmazingFeature`)
5. Pull Request 생성

---

## 로드맵

### 2025 Q1
- [ ] Transformer 통합 (Attention 메커니즘)
- [ ] 실시간 거래 시스템
- [ ] 웹 대시보드

### 2025 Q2
- [ ] 멀티 자산 클래스
- [ ] 연합 학습
- [ ] 모바일 앱

### 2025 Q3
- [ ] 클라우드 배포
- [ ] API 서비스
- [ ] 엔터프라이즈 기능

### 2025 Q4
- [ ] 자동 파라미터 튜닝
- [ ] 고급 리스크 모델
- [ ] 규제 준수 모듈

---

## 지원

### 버그 리포트
[GitHub Issues](https://github.com/yourusername/FinFlow-rl/issues)

### 질문 및 토론
[GitHub Discussions](https://github.com/yourusername/FinFlow-rl/discussions)

### 보안 이슈
security@finflow.ai로 비공개 보고

---

## 라이센스

MIT License - 자세한 내용은 [LICENSE](../LICENSE) 파일 참조

---

*Last Updated: 2025-01-22*
*Version: 2.0.0 (BIPD)*