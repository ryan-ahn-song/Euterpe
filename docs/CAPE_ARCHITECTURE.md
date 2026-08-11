# MeloBridge-CAPE 구현 구조

## 1. 구현 목표

CAPE는 음악 파형을 직접 생성하는 모델이 아니다. 사용자 (u), 음악 구간 (x), 후보 처리 (a)가 주어졌을 때 피치·음색·선율·자연스러움·명료도 결과를 예측하고, Reference 처리와의 차이를 계산한다.

\[
\hat{\tau}_{u,x,a}=\hat{Y}_{u,x,a}-\hat{Y}_{u,x,0}
\]

이 구현에서 `reference`는 WDRC까지만 적용된 기준 경로다. 예측 효과가 작거나 불확실성·왜곡·자연스러움 저하가 한계를 넘으면 추가 처리를 하지 않는다.

## 2. 코드 경계

| 파일 | 역할 |
|---|---|
| `src/hearing_assist/cape.py` | 데이터 계약, Listener Encoder, 효과 예측 앙상블, 정책 |
| `src/hearing_assist/dsp.py` | 음악 특징 추출, 후보 DSP, 전환·crossfade, 리미터 |
| `src/hearing_assist/synthetic.py` | 소프트웨어 검증용 합성 결과식 |
| `src/hearing_assist/engine.py` | WAV 처리와 CAPE 감사 로그 |
| `src/hearing_assist/cli.py` | 학습·평가·검사·오프라인·실시간 명령 |
| `data/cape/listener_example.json` | 문항 단위 Listener 입력 예제 |
| `models/cape_synthetic_demo.npz` | 합성 결과만 학습한 실행 확인용 모델 |

## 3. Listener Encoder

한 사용자를 평균 점수 몇 개로만 표현하지 않는다. 각 검사 문항은 다음 값을 가진다.

- 과제: `pitch`, `timbre`, `melody`, `naturalness`
- 난이도, 배음성, 다성도, 마스킹 정도
- 정답 또는 정규화된 평정, 응답 확신도, 반응시간

각 문항을 11차원 벡터로 만든 후 고정 비선형 projection을 통과시키고, 문항 순서와 개수에 무관하도록 mean/max pooling한다. 여기에 8점 청력도를 붙여 56차원 청취자 표현을 만든다. 이는 DeepSets의 순열 불변 집계 원리를 작은 CPU 모델에 맞게 단순화한 것이다.

중요한 점은 `pitch_score=0.4` 하나가 아니라, 어떤 난이도·배음성·마스킹 조건에서 성공하거나 실패했는지를 입력에 남긴다는 것이다.

## 4. Music Cue Encoder

현재 512점 분석 프레임에서 다음 10개 값을 0~1 범위로 계산한다.

1. F0 신뢰도 기반 harmonicity
2. spectral entropy
3. spectral flatness
4. positive spectral flux
5. crest factor 기반 transient ratio
6. dynamic range proxy
7. 150~2,000 Hz 에너지 비율
8. 4,000 Hz 이상 에너지 비율
9. F0 confidence
10. 로그 주파수로 정규화한 F0

이 값들은 해석 가능한 저비용 특징이다. `polyphony`, 실제 보컬 비율, 주선율/반주 분리값을 직접 측정하는 것은 아니며, 현재 버전에서는 entropy·flatness·대역 에너지로 일부 상황만 근사한다.

## 5. Treatment와 결과

### 후보 처리

| 이름 | 구현 |
|---|---|
| `reference` | 추가 음악 처리 없이 WDRC 유지 |
| `harmonic_cue` | F0 신뢰도가 임계값 이상일 때 최대 20개 배음 근처 제한 이득 |
| `timbre_preserve` | WDRC가 만든 주파수별 이득 편차를 에너지 가중 평균 쪽으로 제한 |
| `melody_relief` | 150~5,000 Hz의 국소 spectral peak에 제한 이득을 주는 선율 proxy |
| `transient_preserve` | 압축 gain reduction의 attack을 느리게 해 onset을 보존 |
| `minimal_processing` | WDRC 이득을 최대 35% 줄여 입력에 가까워지는 자연스러움 우선 후보 |

`melody_relief`는 음원 분리나 실제 보컬 추출이 아니다. 현재 구현은 tonal salience proxy이므로 연구 발표에서도 이 경계를 밝혀야 한다.

### 모델 출력

각 결과는 0~1 범위다.

\[
\hat{Y}=[P,T,M,N,C]
\]

- (P): 피치 과제 성공 확률 또는 정규화 점수
- (T): 음색 과제 결과
- (M): 선율 추적 결과
- (N): 자연스러움
- (C): 명료도

## 6. 효과 예측 모델

구현 모델은 bootstrap random-feature ensemble이다.

1. Listener, Music, Treatment 벡터 결합
2. 각 앙상블 멤버의 비선형 ReLU random projection
3. bootstrap 표본에서 ridge regression head 학습
4. 멤버 평균으로 결과 예측
5. 멤버 간 예측 편차와 학습 분포 이탈 거리로 불확실성 계산

기본값은 7개 멤버와 멤버당 96개 비선형 특징이다. 포함된 합성 데모 모델은 실시간 여유를 위해 5개 멤버와 64개 특징을 사용한다. 후보와 Reference를 하나의 벡터 연산으로 묶어 매 결정마다 Listener Encoding을 반복하지 않는다.

이 구조를 선택한 이유는 대규모 신경망보다 적은 데이터에서 학습 가능하고, CPU 실시간 추론과 불확실성 기반 abstention을 동시에 구현하기 쉽기 때문이다. 신경망 레이어 자체가 연구 독창성이라는 의미는 아니다. 연구 기여는 개인·구간·처리 조합의 결과 차이를 직접 학습하는 문제 정의에 있다.

## 7. 처리 선택과 자동 우회

청취자 검사에서 계산한 취약도 가중치 (w_u)를 이용해 다음 값을 최대화한다.

\[
a^*=\arg\max_a\left[w_u^T\hat{\tau}_{u,x,a}-\lambda D(x,a)-\beta U(u,x,a)\right]
\]

- (D): 처리 종류·강도·현재 음악 특징으로 계산한 보수적 왜곡 추정치
- (U): 앙상블 분산과 분포 이탈을 합친 불확실성

다음 조건이면 `reference`로 우회한다.

- 학습 데이터에 없던 처리 또는 학습 강도 범위 밖
- 불확실성이 `max_uncertainty` 초과
- 왜곡 추정치가 `max_distortion` 초과
- 자연스러움 예상 감소가 `max_naturalness_drop` 초과
- 제약을 통과한 최선의 순효과가 `min_effect` 미만

결정은 기본 16블록마다 수행하며, 기본 300 ms hold와 200 ms crossfade로 빠른 모드 진동과 클릭을 줄인다. 하드 디지털 리미터와 비정상값 fail-safe는 모델 밖에 남는다.

## 8. 데이터 계약

JSONL 한 줄은 다음 구조다.

```json
{
  "listener": {
    "listener_id": "P001",
    "audiogram_db_hl": [20, 25, 30, 35, 40, 40, 35, 30],
    "responses": [
      {
        "task": "pitch",
        "difficulty": 0.6,
        "harmonicity": 0.7,
        "polyphony": 0.3,
        "masking": 0.4,
        "correct": 1,
        "confidence": 0.8,
        "reaction_time_ms": 1250
      }
    ],
    "outcome_weights": {}
  },
  "music": {
    "harmonicity": 0.8,
    "spectral_entropy": 0.4,
    "spectral_flatness": 0.2,
    "spectral_flux": 0.3,
    "transient_ratio": 0.2,
    "dynamic_range": 0.5,
    "low_mid_energy_ratio": 0.7,
    "high_energy_ratio": 0.1,
    "f0_confidence": 0.85,
    "normalized_f0": 0.45
  },
  "treatment": {"name": "harmonic_cue", "strength": 0.65},
  "outcome": {"pitch": 0.75, "timbre": 0.62, "melody": 0.58, "naturalness": 0.70, "clarity": 0.68}
}
```

모든 결과를 단순 선호 질문에서 만들면 기존 선호도 피팅과 구별되지 않는다. 피치·음색·선율 결과는 실제 수행 과제에서, 자연스러움과 명료도는 별도 평정에서 얻어야 한다.

## 9. 학습과 검증

```bash
hearing-assist cape-train \
  --dataset observations.jsonl \
  --output cape.npz \
  --data-status human-research

hearing-assist cape-evaluate \
  --dataset heldout.jsonl \
  --model cape.npz
```

청취자가 3명 이상이면 CLI는 기본적으로 청취자 단위 holdout을 사용한다. 같은 청취자의 관측값이 train과 validation에 동시에 들어가 생기는 과대평가를 줄이기 위해서다. 실제 연구에서는 별도 test listener를 고정하고 다음을 추가 평가해야 한다.

- 각 결과 차원의 MAE·Brier score와 calibration
- 도움이 되는 처리/해로운 처리 판별
- 사용자별 oracle 대비 regret
- Reference 대비 과제 정확도 변화
- 자연스러움 비열등성
- HAAQI와 spectral distortion
- 처리 활성/우회 비율, p95/p99 시간, deadline miss

## 10. 합성 모델의 정확한 의미

`models/cape_synthetic_demo.npz`는 사람이 만든 결과식에서 생성한 1,024개 관측으로 파이프라인을 확인한 모델이다. 청취자 단위 validation MAE가 기록되어 있지만, 이는 합성 식을 다시 근사한 수치일 뿐 다음을 입증하지 않는다.

- 난청인의 음악지각 개선
- 청각기관의 생리적 시뮬레이션
- 임상 안전성 또는 처방 적합성
- 실제 데이터에서의 일반화

모델 내부에는 `data_status: synthetic`가 저장되어 있으며 `cape-inspect`로 확인할 수 있다.

CAPE Listener 파일의 청력도와 활성 보청 프로필의 청력도는 정확히 같아야 한다. 서로 다른 사용자의 모델·DSP 프로필이 실수로 결합되면 처리기 초기화 단계에서 거부한다.
