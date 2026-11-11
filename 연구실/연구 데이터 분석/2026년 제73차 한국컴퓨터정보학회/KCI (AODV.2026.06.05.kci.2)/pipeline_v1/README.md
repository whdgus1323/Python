# pipeline_v1

원본 노트북 `main_v.2.ipynb`, `main_v.3.ipynb`, `main_v.4.ipynb`, `main_v.5.ipynb`, `main_v.6.ipynb`, `main_v.7.ipynb`는 수정하지 않고 유지한다.

이 폴더는 아래 절차를 따르는 별도 실험 라인이다.

1. 데이터 수집
2. 라벨 생성
3. 누수 없는 분할
4. baseline 평가
5. 모델 평가
6. 결과 저장

## 현재 원칙

- online input은 아래 4개만 사용
  - `local CBR`
  - `neighbor count`
  - `hop count`
  - `isDirectRoute`
- `PDR`, `futureSucceeded`, `futureFailed`, `futureDelay`, `runPdr`는 online input으로 쓰지 않는다
- 위 값들은 오직 offline label 생성용 또는 offline 분석용으로만 사용한다
- train/test는 반드시 `stateKey` 기준으로 나눈다
- 같은 `stateKey`가 train/test에 동시에 들어가면 안 된다

## 추천 실행 순서

1. `python run_pipeline.py --mode baseline`
2. `python run_pipeline.py --mode model`
3. 결과 파일 `outputs/summary_*.csv` 확인

## 현재 baseline 기준

- 가장 최근 확인된 누수 없는 기준점:
  - `include_run_pdr = False`
  - `state_cbr_bin_size = 5`
  - `state_neighbor_bin_size = 5`
  - `min_pair_samples_per_state = 10`
- 이 설정에서 `r2_uniform = -0.126022`

이 값은 "아직 모델 문제가 해결되지 않았다"는 기준점으로만 사용한다.
