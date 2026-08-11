# 실시간 보청 보조 프로토타입 실행 가이드

## 1. 준비물

- Python 3.10 이상이 설치된 Windows, macOS, Linux 또는 Raspberry Pi 5
- 지연이 낮은 USB 오디오 인터페이스 또는 내장 오디오 장치
- 마이크 1개
- **유선** 이어폰 또는 헤드폰
- 최초 점검을 함께 할 정상청력 성인 1명

Bluetooth는 코덱·버퍼 지연이 크고 변동하므로 사용하지 않는다. 스피커를 출력으로 사용하면 마이크와 음향 피드백이 생길 수 있으므로 폐쇄형 유선 출력부터 사용한다.

## 2. 설치

### Windows PowerShell

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
```

### Linux / Raspberry Pi OS

PortAudio가 없다면 먼저 설치한다.

```bash
sudo apt update
sudo apt install -y python3-venv libportaudio2 portaudio19-dev
bash scripts/setup_linux.sh
```

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[live]'
python -m unittest discover -s tests -v
```

## 3. 소리 없이 소프트웨어 검증

```bash
hearing-assist profile-check --profile configs/hearing_assist_safe.yaml

hearing-assist demo \
  --profile configs/hearing_assist_example_mild_music.yaml \
  --output-dir outputs/demo
```

`outputs/demo/demo_input.wav`와 `demo_hearing_assist.wav`를 비교할 수 있다. `report/metrics.json`에는 피크, 리미터 작동률, 처리시간이 저장된다.

CAPE 전체 경로는 포함된 합성 모델로 확인한다.

```bash
hearing-assist profile-check \
  --profile configs/hearing_assist_cape_synthetic_demo.yaml

hearing-assist demo \
  --profile configs/hearing_assist_cape_synthetic_demo.yaml \
  --output-dir outputs/cape-demo
```

`report/block_diagnostics.csv`에는 블록별 선택 처리, 강도, 순효과 점수, 불확실성, 왜곡 추정치와 우회 이유가 기록된다. 이 모델은 소프트웨어 시연용 합성 결과만 학습했으므로 청취 개선 근거로 사용할 수 없다.

자신의 PCM WAV 파일을 처리하려면 다음 명령을 사용한다.

```bash
hearing-assist offline input.wav \
  --output outputs/processed.wav \
  --profile configs/hearing_assist_example_mild_music.yaml \
  --report-dir outputs/processed_report
```

## 4. 오디오 장치 선택

```bash
hearing-assist devices
```

표시된 장치 번호 또는 이름을 `--input-device`, `--output-device`에 넣는다. 장치가 48 kHz를 지원하지 않으면 프로필의 `sample_rate`를 장치 지원값으로 바꾸되, 프로필 검증 명령을 다시 실행한다.

## 5. 첫 실시간 점검

1. 이어폰을 귀에서 뺀다.
2. 운영체제와 오디오 인터페이스의 물리 출력 볼륨을 최저로 낮춘다.
3. 이득이 0 dB인 `hearing_assist_safe.yaml`로 10초 점검한다.

```bash
hearing-assist live \
  --profile configs/hearing_assist_safe.yaml \
  --input-device 1 \
  --output-device 3 \
  --seconds 10 \
  --accept-uncalibrated-risk
```

스트림 상태 오류, deadline miss 또는 fail-safe가 0인지 확인한다. 그 뒤에도 출력 볼륨은 아주 조금씩 올린다.

## 6. 개인 설정

`configs/hearing_assist_audiologist_template.yaml`을 복사해 사용한다.

- `hearing_loss_db_hl`: 실제 청력검사 결과
- `manual_gain_db`: 전문가가 정한 대역별 디지털 이득. 지정하면 간이 이득 계산보다 우선함
- `compression`: 압축 시작점, 비율, attack/release
- `limiter.ceiling_dbfs`: 디지털 피크 상한
- `calibration`: 검증한 오디오 장치, 출력 경로, 담당자와 조건

자동 계산은 연구용 보수적 휴리스틱이며 NAL-NL2, DSL 같은 임상 처방식을 구현한 것이 아니다. 실제 피팅에는 정확한 장치 조합으로 실이측 또는 커플러 검증이 필요하다.

## 7. 즉시 중단 조건

다음 상황에서는 출력을 즉시 중단하고 볼륨을 0으로 내린다.

- 불쾌할 정도로 크거나 날카로운 소리
- 지속적인 휘파람성 피드백
- 이명, 귀 통증, 먹먹함 또는 어지럼
- 반복되는 dropout, overflow, fail-safe 또는 deadline miss

이 프로토타입은 응급·의료 판단, 아동 피팅, 일상 상시 착용 또는 상용 배포에 사용하지 않는다.
