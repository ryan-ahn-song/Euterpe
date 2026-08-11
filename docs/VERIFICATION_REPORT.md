# v0.3.0 검증 보고서

검증일: 2026-08-11

## 자동 테스트

```text
18 tests passed
```

확인 항목:

1. 세 개의 보청 보조 프로필 스키마와 값 범위
2. 24 dB를 초과하는 최대 이득 설정 거부
3. 0 dB 프로필의 지연 보정 후 sample-exact passthrough
4. -12 dBFS 리미터 ceiling 불변식
5. WDRC가 작은 입력에 큰 입력보다 더 높은 이득을 적용
6. NaN 입력 블록의 즉시 음소거와 fail-safe 계수
7. WAV 입력부터 처리 WAV·JSON·CSV 로그까지 end-to-end 출력
8. 장치가 없어도 실행되는 mono→stereo 실시간 콜백 통합 테스트
9. CAPE JSONL 스키마 저장·복원
10. CAPE 모델 학습과 저장·복원 전후 예측 일치
11. 정책의 처리 선택과 높은 최소 효과 조건에서 Reference 우회
12. CAPE 선택 DSP가 실시간 처리기에서 작동하면서 디지털 ceiling 유지
13. 기존 인공와우 F0, ACE 선택, 파이프라인, 결과 계약 회귀 테스트

## CAPE 모델 검증

포함 모델은 8명의 합성 청취자, 청취자당 8개 음악 구간, Reference와 15개 후보 처리·강도 조합으로 만든 1,024개 관측을 사용했다. 청취자 2명을 validation으로 완전히 분리했다.

| 지표 | 학습 768건 | 청취자 holdout 256건 |
|---|---:|---:|
| MAE | 0.0167 | 0.0645 |
| RMSE | 0.0217 | 0.0855 |
| Brier/MSE | 0.00047 | 0.00731 |

이는 사람이 작성한 합성 결과식을 모델이 근사할 수 있는지 확인한 소프트웨어 지표다. 실제 청취 수행, 난청인의 개선, 임상 일반화를 보여주지 않는다.

## CAPE end-to-end 성능

내장 2.08초 멜로디와 `cape-synthetic-demo` 프로필을 사용했다.

| 지표 | 결과 |
|---|---:|
| 블록 수 | 391 |
| 평균 처리시간 | 0.42 ms |
| p95 | 1.08 ms |
| p99 | 1.80 ms |
| 최대 | 2.80 ms |
| 블록 deadline | 5.33 ms |
| deadline miss | 0 |
| Reference 블록 | 334 |
| `timbre_preserve` 블록 | 57 |
| fail-safe | 0 |
| 리미터 활성 비율 | 0% |

후보 15개와 Reference를 개별 추론했을 때는 결정 블록이 deadline을 넘었다. Listener Encoding을 한 번만 계산하고 후보 전체를 단일 벡터 연산으로 평가하도록 수정한 뒤 위 결과를 얻었다.

10초 무작위 입력, 1,875블록 CAPE DSP 벤치마크에서도 평균 0.41 ms, p95 1.44 ms, p99 1.64 ms, 최대 2.30 ms, deadline miss 0회를 기록했다. 두 시간 결과 모두 이 클라우드 CPU의 DSP-only 측정이며 오디오 드라이버와 장치 지연은 포함하지 않는다.

## DSP 성능

클라우드 CPU에서 `example-mild-music` 프로필, 48 kHz, 256샘플 블록, 10초 무작위 입력을 처리했다.

| 지표 | 결과 |
|---|---:|
| 블록 수 | 1,875 |
| 선언 알고리즘 지연/블록 deadline | 5.33 ms |
| 평균 DSP 시간 | 0.21 ms |
| p95 | 0.25 ms |
| p99 | 0.28 ms |
| 최대 | 4.39 ms |
| deadline miss | 0 |

이 결과는 DSP 코어만 측정한다. PortAudio, 운영체제, USB 오디오 인터페이스, ADC/DAC와 안전 버퍼 지연은 포함하지 않는다.

## 예제 WAV

`outputs/hearing_assist_smoke/`의 2.08초 음악 입력을 처리했다.

- 입력/출력: 48 kHz mono, 각 99,840 samples
- 비정상값: 없음
- fail-safe: 0회
- deadline miss: 0회
- 처리 출력 최대 피크: 약 -21.05 dBFS
- 리미터 활성 블록 비율: 0%

이 예제는 파일 계약과 신호 경로를 확인하는 smoke test이며 난청인의 청취 개선 증거가 아니다.

## 환경상 확인하지 못한 항목

클라우드 컨테이너에는 PortAudio 라이브러리와 물리 오디오 장치가 없어 실제 마이크·이어폰 스트림은 실행할 수 없었다. 대신 오디오 콜백의 블록 크기, mono→stereo 출력, 예외 음소거와 통계를 자동 테스트했다. 실장 장치에서는 `hearing-assist devices`와 `live --seconds 10`으로 별도 확인해야 한다.
