# 006 실험 결과 발표 정리

## 1) 실험 개요

이번 실험은 동일한 LAVAR 백본(잠재변수 학습기)을 유지한 상태에서,
타깃 세트(large vs small)와 공급 예측 헤드의 동작 특성을 비교한 롤링 평가(259 folds)이다.

- 참고 로그: `REPORT.md`
- 시각화: `../output_graph_small.png`, `../output_graph_large.png`

---

## 2) LAVAR 모델 요약 (LAVAR_README 기반)

`LAVAR_README.md` 기준으로, 모델은 2-stage 구조다.

1. **Stage 1 (LAVAR)**
   - 입력 공변량 `x`만 사용해 잠재상태 `z`를 학습
   - 오토인코더 재구성 + 잠재 VAR 동역학 손실로 학습
   - 이 단계가 공통 **latent variable finder** 역할

2. **Stage 2 (Supply head)**
   - Stage 1을 freeze하고, 미래 잠재상태에서 공급 타깃 `y`를 예측하는 헤드 학습
   - 밀도 기반 버킷(dense/sparse/ultra)으로 분리해 헤드를 독립 학습

핵심 메시지:

- **"같은 latent finder + 바꿔 낄 수 있는 헤드"** 구조가 맞다.
- 실제로 실험에서는 타깃 세트(`yl`/`ys`)를 바꾸거나, 헤드 구성(버킷별 학습)을 바꿔 비교 가능하다.

---

## 3) Large vs Small 정량 비교

### Large Head (REPORT.md: line 435+)

- classic mean/std: **177.44 / 1308.29**
- robust median: **74.64**
- p75/p90: **106.99 / 157.16**
- p95/p99: **269.81 / 429.96**
- trimmed mean(5%): **86.35**
- IQR/MAD: **50.62 / 23.55**
- guardrail: **3 / 259**
- outliers: **19 folds**

### Small Head (REPORT.md: line 938+)

- classic mean/std: **148.12 / 888.49**
- robust median: **55.37**
- p75/p90: **79.45 / 122.53**
- p95/p99: **172.33 / 495.87**
- trimmed mean(5%): **62.56**
- IQR/MAD: **39.13 / 18.26**
- guardrail: **8 / 259**
- outliers: **14 folds**

### 해석

- 중심 성능(중앙값, trimmed mean)은 **Small가 더 우수**
- 분산/변동성(IQR, MAD)도 **Small가 더 안정적**
- Large는 outlier 개수가 더 많고 tail risk가 더 큼

---

## 4) 그래프 해석

## Small 결과 그래프

![Small 결과](../output_graph_small.png)

- 다수 타깃에서 관측 추세를 따라가지만, 스파이크 구간에서 과/소추정이 반복됨
- 일부 fold에서 급격한 폭주가 존재하나, 전체 분포는 Large 대비 상대적으로 타이트

## Large 결과 그래프

![Large 결과](../output_graph_large.png)

- 고변동 타깃에서 예측선과 실측선 괴리가 더 크고 빈번함
- 특정 구간에서 편향 누적이 발생하며, 상위 오차 구간이 넓게 나타남

---

## 5) 결론

1. 현재 설정에서 **Small Head가 Large Head보다 안정적**이다.
2. 모델 구조적으로는 **같은 잠재표현 학습기(Stage 1)를 공유**하고,
   Stage 2 헤드/타깃을 바꿔 실험하는 방식이 유효하다.
3. Large 성능 개선의 핵심은 평균 성능보다 **tail risk(극단 오차) 제어**다.

---

## 6) 다음 액션 (권장)

1. Large 전용 `lr_supply` 하향 (예: `1e-3 -> 3e-4` 또는 `1e-4`)
2. 재학습 주기 단축 (`retrain_every_days` 90 -> 30~45)
3. fold 품질 악화 시 naive fallback/블렌딩 추가
4. guardrail 임계값 강화(특히 overshoot 구간)
