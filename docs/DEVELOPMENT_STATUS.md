# 개발 상태 — v0.1.0

## 이번 개발에서 확정한 경계

- 알고리즘 입력: 16 kHz mono `float32`
- 분석 단위: 128 samples, 8 ms
- 분석 채널: 22개
- 선택 전략: 동일 분석 결과를 받는 `ace`와 `msa`
- 기본 활성 채널: 8개
- 출력: 프레임 Electrodogram, 선택 근거 로그, F0 로그, runtime, 음향 보코더
- 안전 경계: 실제 전류·RF·임플란트 출력 없음

## 구현 완료

1. 설정·데이터 모델·파이프라인 인터페이스
2. 상태 유지형 causal 필터뱅크
3. 48 ms 과거 문맥 자기상관 F0 추정
4. 에너지 상위 K개 ACE형 기준선
5. 에너지·배음·연속성·중복 기반 MSA-Select
6. 결정론적 noise-band vocoder
7. CLI, CSV/NPY/JSON/WAV 결과 저장
8. 합성 멜로디 end-to-end 실행과 테스트

## 검증된 실행 결과

내장 2.08초 합성 멜로디, 260프레임을 현재 클라우드 CPU에서 실행했을 때 두 전략 모두 8 ms deadline miss가 0회였다. MSA-Select의 채널 교체율이 ACE형 기준선보다 낮게 측정됐지만, 이는 구현 건전성 확인용 합성 데이터 한 건의 결과일 뿐 성능 우월성의 증거로 사용하면 안 된다.

## 다음 개발 우선순위

### P0 — 기준선 신뢰성

- Nucleus Toolbox와 입력·필터·압축 조건 정렬
- 프레임별 채널 에너지와 top-K 선택 회귀 비교
- 기준 결과를 `tests/reference_data/`에 고정

### P1 — 연구 가설 완성

- 80–300 Hz 조건부 시간적 F0 강화
- 프레임 내부 oscillator 위상 연속성
- `PulseEvent(timestamp_us, electrode, level)` 출력
- MSA-Select ablation: E, E+H, E+H+C, E+H+C−R

### P2 — 평가 자동화

- 단선율 데이터셋 manifest
- 반음 오차·멜로디 윤곽·배음 보존·음성 보호 지표
- ACE/MSA/시간 강화 조건 일괄 실행
- bootstrap 신뢰구간과 통계 비교

### P3 — 실시간 프로토타입

- WAV replay 실시간 스케줄러
- 오디오 callback과 비동기 로그 큐 분리
- p95/p99 deadline 계측
- Raspberry Pi 5 ARM 환경 벤치마크

## 주의할 기술 부채

- 현재 로그 간격 필터뱅크는 구조 검증용이며 임상 ACE 계수와 동일하지 않다.
- 현재 Electrodogram 값은 정규화 포락선이지 환자별 T/C level이나 전류 단위가 아니다.
- 현재 vocoder는 개발 확인용이며 청취실험 전 loudness matching과 조건 통제가 필요하다.
- 합성 멜로디 결과만으로 음악 지각 개선을 주장할 수 없다.

