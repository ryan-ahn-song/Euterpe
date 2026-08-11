# 안전 경계와 적용 범위

## 이 소프트웨어가 하는 것

이 프로젝트는 PC 또는 Raspberry Pi에서 다음 음향 경로를 실제 실행한다.

```text
마이크 → 48 kHz 블록 입력 → 청력도/수동 이득 → WDRC
      → CAPE 효과 예측과 제약 정책 → 후보 DSP/Reference 우회
      → 디지털 리미터 → 유선 음향 출력
```

한 블록의 알고리즘 지연은 기본 설정에서 5.33 ms다. 실제 입출력 지연에는 오디오 인터페이스, 운영체제, 드라이버 버퍼가 추가된다.

## 포함된 안전 제어

- 프로필 값과 이득 범위 검증
- 최대 대역 이득 24 dB의 코드 상한, 예제 기본값 12 dB
- 큰 입력에서 증폭을 줄이는 WDRC
- 기본 -6~-12 dBFS 디지털 피크 리미터
- 시작 시 0에서 서서히 증가하는 출력 램프
- NaN/무한대 또는 DSP 오류 시 해당 블록 음소거
- 미보정 프로필의 실시간 실행에 명시적 경고 동의 요구
- 처리시간, dropout 신호, 리미터와 fail-safe 기록
- CAPE 불확실성·왜곡·자연스러움 한계와 학습 범위 밖 처리 거부
- 모델이 없거나 후보의 순효과가 작을 때 Reference 자동 우회

## 소프트웨어만으로 보장할 수 없는 것

`dBFS`는 디지털 신호 크기이며 `dB SPL`이 아니다. 같은 -9 dBFS도 오디오 인터페이스, 앰프, 이어폰 감도, 이어팁과 귀 결합에 따라 전혀 다른 음압이 된다. 따라서 디지털 리미터만으로 귀에 도달하는 최대 음압을 인증할 수 없다.

현재 버전에는 다음이 없다.

- NAL-NL2/DSL 같은 검증된 처방식의 공식 구현
- 이어폰/리시버별 SPL 교정과 OSPL90 측정
- 실이측(REM) 또는 커플러 기반 검증
- 적응형 음향 피드백 제거
- 빔포밍, 바람 소리 검출, 환경 분류
- 양이 동기화
- 배터리, 발열, 단일고장 안전성 검증
- 의료기기 소프트웨어 생명주기·위험관리·임상 검증
- 실제 난청인 데이터로 학습·검증된 CAPE 모델
- 보컬/주선율을 직접 분리하는 모델과 검증된 polyphony 추정

포함된 `cape_synthetic_demo.npz`는 사람이 작성한 결과식을 학습한 소프트웨어 데모다. 합성 validation 지표나 처리 선택 로그를 사람의 청취 개선 결과로 표현하면 안 된다.

`calibration.output_calibrated: true`는 사용자가 외부 검증 사실을 기록하는 필드일 뿐, 소프트웨어가 자동으로 교정 또는 인증했다는 뜻이 아니다.

## 허용되는 사용 범위

- WAV 기반 알고리즘 개발과 비교
- 이어폰을 귀에서 뺀 상태의 장치 연결 시험
- 낮은 물리 볼륨에서 감독되는 비임상 벤치 시연
- 청각 전문가가 별도 검증하기 전의 연구 프로토타이핑

의료기기, 처방 보청기 대체재, 실제 인공와우 제어기, 아동용 장치 또는 장시간 일상 착용 장치로 사용하면 안 된다.

## 외부 검증이 필요한 이유

FDA의 OTC 보청기 출력 한계도 규정된 음향 커플러에서 측정한 SPL 기준으로 정의된다. ASHA Evidence Maps가 요약한 지침은 처방 목표와 디지털 기능을 확인하는 데 probe-microphone 실이측을 권고한다. WHO-ITU의 안전 청취 체계 역시 장치가 실제 음향 노출을 측정·관리하고 사용자에게 정보를 제공하는 것을 전제로 한다.

- FDA OTC hearing-aid final-rule impact analysis: https://www.fda.gov/media/160971/download
- ASHA, probe-microphone verification summary: https://apps.asha.org/EvidenceMaps/Articles/ArticleSummary/9a7acb2a-5874-4415-bae5-069a9d7f733f
- WHO-ITU safe-listening toolkit: https://www.who.int/publications/i/item/toolkit-for-safe-listening-devices-and-systems
