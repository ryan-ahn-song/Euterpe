# MeloBridge-CAPE + CI Research Platform v0.3.0

| 실행기 | 목적 | 출력 |
|---|---|---|
| `hearing-assist` | MeloBridge-CAPE 음악 특화 보청 보조 연구 프로토타입 | 마이크/WAV를 처리한 실제 음향 파형과 결정 로그 |
| `music-ci` | 기존 인공와우 음악 부호화 연구 시뮬레이터 | Electrodogram, 채널 로그, 보코더 |

## 실제 음향·AI 경로

```text
마이크 또는 WAV
→ 48 kHz / 256-sample streaming
→ 청력도 또는 전문가 수동 이득
→ 8대역 WDRC
→ 음악 단서 10종 실시간 분석
→ CAPE 후보 처리별 개인 지각 효과·불확실성 예측
→ 왜곡·자연스러움·불확실성 제약 정책
→ 선택 DSP 또는 Reference 자동 우회
→ 디지털 피크 리미터·startup ramp·fail-safe
→ 유선 이어폰/헤드폰 또는 처리 WAV
```

오프라인과 실시간 경로는 동일한 DSP 코어를 사용한다. 기본 설정의 알고리즘 지연은 5.33 ms이며, 실제 장치 지연은 드라이버와 하드웨어 버퍼가 추가된다.

## 설치와 첫 실행

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[live]'

python -m unittest discover -s tests -v

hearing-assist profile-check \
  --profile configs/hearing_assist_safe.yaml

hearing-assist demo \
  --profile configs/hearing_assist_example_mild_music.yaml \
  --output-dir outputs/demo

# 포함된 합성 모델로 CAPE 전체 경로 확인
hearing-assist demo \
  --profile configs/hearing_assist_cape_synthetic_demo.yaml \
  --output-dir outputs/cape-demo

hearing-assist devices
```

실시간 실행은 반드시 물리 출력 볼륨을 최저로 내리고 이득 0 dB의 안전 프로필부터 시작한다.

```bash
hearing-assist live \
  --profile configs/hearing_assist_safe.yaml \
  --input-device 1 \
  --output-device 3 \
  --seconds 10 \
  --accept-uncalibrated-risk
```

Windows, Linux, Raspberry Pi 설정과 단계별 점검은 `docs/QUICKSTART_KO.md`에 정리되어 있다.

## 프로필

- `configs/hearing_assist_safe.yaml`: 0 dB 이득, 음악 강화 꺼짐, -12 dBFS ceiling
- `configs/hearing_assist_example_mild_music.yaml`: 알고리즘 시연용 가상 청력도
- `configs/hearing_assist_cape_synthetic_demo.yaml`: CAPE 전체 경로 확인용 합성 모델
- `configs/hearing_assist_audiologist_template.yaml`: 측정값과 검증 이득 입력용 빈 양식

합성 CAPE 모델은 코드 작동을 보여주는 데모이며 사람의 청취 개선 근거가 아니다. 예제의 자동 이득 계산도 임상 처방식이 아니다. 개별 적용은 정확한 청력검사값, 출력 장치 SPL 교정, 커플러 또는 실이측 검증이 필요하다.

## CAPE 모델 학습

학습 데이터 한 행은 `(청취자 문항 응답, 음악 구간 특징, 처리와 강도) → 피치·음색·선율·자연스러움·명료도 결과`인 JSONL이다.

```bash
# 데이터 파이프라인만 검증하는 합성 데이터. 연구 결과로 사용하면 안 된다.
PYTHONPATH=src python scripts/generate_cape_synthetic_data.py \
  --output outputs/cape-synthetic.jsonl

hearing-assist cape-train \
  --dataset outputs/cape-synthetic.jsonl \
  --output outputs/cape-model.npz \
  --data-status synthetic

hearing-assist cape-evaluate \
  --dataset outputs/cape-synthetic.jsonl \
  --model outputs/cape-model.npz

hearing-assist cape-inspect --model outputs/cape-model.npz
```

모델 구조, 데이터 계약, 후보 DSP와 실제 사람 대상 검증 설계는 `docs/CAPE_ARCHITECTURE.md`에 정리되어 있다.

## 테스트와 벤치마크

```bash
python -m unittest discover -s tests -v

PYTHONPATH=src python scripts/benchmark_hearing_assist.py \
  --profile configs/hearing_assist_example_mild_music.yaml \
  --seconds 10
```

자동 테스트는 다음을 확인한다.

- 0 dB 프로필의 sample-aligned passthrough
- 디지털 ceiling 초과 방지
- 작은 입력에 큰 입력보다 더 높은 이득 적용
- NaN 입력 시 음소거 fail-safe
- 프로필 범위 검증
- WAV→처리 WAV→감사 로그 end-to-end 실행
- CAPE JSONL 왕복, 모델 학습·저장·복원 시 예측 일치
- 제약 기반 자동 우회와 실시간 DSP 처리 선택
- 기존 `music-ci` 회귀 테스트

## 중요한 경계

이 프로젝트는 **실행 가능한 비임상 연구 프로토타입**이지만 승인된 보청기나 의료기기는 아니다. 디지털 -9 dBFS는 이어폰의 실제 dB SPL을 나타내지 않으며, 현재 버전에는 적응형 피드백 제거와 공식 처방식, 실이측, 하드웨어 단일고장 검증이 없다.

사용 전 `docs/SAFETY_AND_LIMITATIONS.md`를 확인한다. CAPE 구조는 `docs/CAPE_ARCHITECTURE.md`, DSP 구조는 `docs/HEARING_ASSIST_ARCHITECTURE.md`, 기존 인공와우 연구 엔진 상태는 `docs/DEVELOPMENT_STATUS.md`에 있다.
