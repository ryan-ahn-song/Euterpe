# 보청 보조 소프트웨어 아키텍처

## 실행 경로

```text
PCM WAV ─────────────┐
                    ├─ 256-sample block → HearingAssistProcessor → mono/stereo output
PortAudio microphone ┘
```

오프라인과 실시간 모드는 같은 `HearingAssistProcessor.process_block()`을 호출한다. 이 구조 덕분에 WAV 회귀 테스트에서 검증한 DSP를 오디오 콜백에서 그대로 사용한다.

## DSP 순서

1. 현재 256샘플과 직전 256샘플을 512점 분석 프레임으로 결합
2. sqrt-Hann WOLA와 실수 FFT
3. 청력도 주파수 구간별 입력 레벨 추정
4. 기본 이득과 WDRC 압축량 계산
5. attack/release 기반 이득 평활화
6. 과거 48 ms 기반 F0와 신뢰도 추정, 음악 단서 10종 계산
7. CAPE가 Listener·Music·Treatment 조합의 Reference 대비 효과와 불확실성 예측
8. 제약 정책이 후보 처리 또는 Reference 우회 결정
9. 선택 처리·강도를 hold와 crossfade로 적용
10. 역 FFT와 overlap-add
11. 즉시 attack 디지털 블록 리미터
12. startup ramp와 비정상값 fail-safe

## 이득 계산

수동 이득이 없을 때의 초기 연구 이득은 다음 휴리스틱이다.

\[
G_k^{base}=\operatorname{clip}\left(0.35\max(HL_k-20,0),0,G_{max}\right)
\]

이는 임상 처방식이 아니다. 큰 입력에서는 다음 WDRC 감쇠를 기본 이득에서 뺀다.

\[
G_k(t)=\max\left(0,G_k^{base}-\left(1-\frac{1}{CR}\right)\max(L_k(t)-K,0)\right)
\]

- \(HL_k\): 대역별 청력손실 입력값
- \(G_k^{base}\): 기본 디지털 이득
- \(L_k\): 입력 대역 레벨(dBFS)
- \(K\): 압축 knee
- \(CR\): 압축비

## 시간과 스레드

- 샘플링: 48 kHz
- 블록: 256 samples
- 선언된 알고리즘 지연: 5.33 ms
- FFT: 512 points
- 분석 중 미래 입력: 출력 기준 1블록 look-ahead
- 실시간 콜백: 입력, DSP, 출력만 수행
- 파일 로그: 오프라인 실행에서만 저장

CAPE는 기본 16블록마다 결정하며 후보 전체를 한 번의 벡터 연산으로 평가한다. DSP 블록마다 수행되는 안전 리미터와 달리, AI 추론이 실패하거나 제약을 통과하지 못하면 Reference 경로를 사용한다. 상세 구조는 `CAPE_ARCHITECTURE.md`를 참고한다.

실시간 모드의 PortAudio status event와 DSP deadline miss는 종료 요약에 집계한다. 콜백 예외는 출력 블록을 음소거한다.
