# main_v.2 전처리 및 제안 기법 수식 정리

본 문서는 `main_v.2.ipynb`에서 사용한 전처리 과정과 MLP 기반 threshold 예측 기법을 논문에 사용할 수 있도록 수식 형태로 정리한 것이다. 성능 평가 지표(MAE, RMSE, $R^2$, ROC AUC, F1-score 등)는 제외하였다.

## 1. Threshold 후보 필터링

각 실험 샘플 $i$는 실험 설정으로부터 low threshold와 high threshold를 갖는다.

$$
\tau_i^{low},\ \tau_i^{high}
$$

유효한 threshold 조합은 다음 조건을 만족하는 경우로 제한한다.

$$
\tau_i^{low} \in \mathcal{T}_{low}
$$

$$
\tau_i^{high} \in \mathcal{T}_{high}
$$

$$
G_{\min} \leq \tau_i^{high}-\tau_i^{low} \leq G_{\max}
$$

`main_v.2`에서는 다음 범위를 사용한다.

$$
\mathcal{T}_{low}=\{10,11,\ldots,24\}
$$

$$
\mathcal{T}_{high}=\{35,36,\ldots,59\}
$$

$$
G_{\min}=25,\quad G_{\max}=40
$$

## 2. Future Window 기반 성능 점수

각 RREP decision 발생 시각을 $t_i$, 해당 노드를 $v_i$라 한다. decision 시각은 초 단위로 변환한다.

$$
s_i=\lfloor t_i \rfloor
$$

decision 이후 $W$초 동안의 route discovery 결과를 집계한다.

$$
S_i=\sum_{k=0}^{W}S(v_i,s_i+k)
$$

$$
U_i=\sum_{k=0}^{W}U(v_i,s_i+k)
$$

$$
F_i=\sum_{k=0}^{W}F(v_i,s_i+k)
$$

여기서 $S_i$, $U_i$, $F_i$는 각각 future window 내 route discovery started, succeeded, failed 횟수이다.

delay는 0보다 큰 값만 평균한다.

$$
\Omega_i=
\{k\mid 0\leq k\leq W,\ D(v_i,s_i+k)>0\}
$$

$$
\bar{D}_i=
\begin{cases}
\frac{1}{|\Omega_i|}
\sum_{k\in\Omega_i}D(v_i,s_i+k), & |\Omega_i|>0\\
0, & |\Omega_i|=0
\end{cases}
$$

`main_v.2`에서는 다음 값을 사용한다.

$$
W=2
$$

성능 점수는 세 조건 점수의 합으로 정의한다.

$$
\mathrm{Score}_i
=
C_{fail,i}
+
C_{succ,i}
+
C_{delay,i}
$$

failure score는 실패 횟수가 적을수록 높은 값을 갖는다.

$$
C_{fail,i}
=
\frac{1}{1+F_i}
$$

success score는 route discovery 성공률을 기반으로 한다.

$$
C_{succ,i}
=
\begin{cases}
1, & S_i\leq0\\
\mathrm{clip}\left(\frac{U_i}{S_i},0,1\right), & S_i>0
\end{cases}
$$

delay threshold는 양수 delay 값의 중앙값으로 정의한다.

$$
D_{th}
=
Q_{0.5}
\left(
\{\bar{D}_i\mid \bar{D}_i>0\}
\right)
$$

delay score는 다음과 같이 정의한다.

$$
C_{delay,i}
=
\begin{cases}
1, & \bar{D}_i\leq0\\
\mathrm{clip}
\left(
1-
\max
\left(
\frac{\bar{D}_i}{\max(D_{th},\epsilon)}
-1,
0
\right),
0,
1
\right),
& \bar{D}_i>0
\end{cases}
$$

따라서 성능 점수의 범위는 다음과 같다.

$$
0\leq \mathrm{Score}_i\leq3
$$

학습에는 다음 조건을 만족하는 샘플만 사용한다.

$$
\mathrm{Score}_i>0
$$

## 3. 입력 특징 전처리

모델 입력은 local CBR, neighbor count, hop count, direct route 여부로 구성한다.

$$
\mathbf{x}_i=
[x_{c,i},x_{n,i},x_{h,i},x_{d,i}]^T
$$

local CBR은 0부터 100 사이의 percentage 값으로 보고 정규화한다.

$$
x_{c,i}
=
\mathrm{clip}
\left(
\frac{c_i}{100},
0,
1
\right)
$$

neighbor count는 기준값 $N_{norm}$으로 정규화한다.

$$
x_{n,i}
=
\mathrm{clip}
\left(
\frac{n_i}{N_{norm}},
0,
1
\right)
$$

hop count는 기준값 $H_{norm}$으로 정규화한다.

$$
x_{h,i}
=
\mathrm{clip}
\left(
\frac{h_i}{H_{norm}},
0,
1
\right)
$$

direct route 여부는 hop count가 1 이하인지로 정의한다.

$$
x_{d,i}
=
\begin{cases}
1, & h_i\leq1\\
0, & h_i>1
\end{cases}
$$

`main_v.2`에서는 다음 값을 사용한다.

$$
N_{norm}=20,\quad H_{norm}=10
$$

## 4. Threshold Target 변환

모델은 low/high threshold를 직접 예측하지 않고, threshold center와 threshold gap을 예측한다.

threshold gap은 다음과 같다.

$$
g_i=
\tau_i^{high}
-
\tau_i^{low}
$$

threshold center는 다음과 같다.

$$
m_i=
\frac{
\tau_i^{low}
+
\tau_i^{high}
}{2}
$$

center의 허용 범위는 gap을 고려하여 정의한다.

$$
m_i^{min}
=
\tau_{min}
+
\frac{g_i}{2}
$$

$$
m_i^{max}
=
\tau_{max}
-
\frac{g_i}{2}
$$

center를 0부터 1 사이로 정규화한다.

$$
\tilde{m}_i
=
\mathrm{clip}
\left(
\frac{
m_i-m_i^{min}
}{
m_i^{max}-m_i^{min}
},
\epsilon,
1-\epsilon
\right)
$$

정규화된 center에는 logit 변환을 적용한다.

$$
y_{m,i}
=
\mathrm{logit}(\tilde{m}_i)
=
\ln
\left(
\frac{\tilde{m}_i}{1-\tilde{m}_i}
\right)
$$

gap은 최소 gap을 제외한 값으로 변환한다.

$$
g_i'
=
\mathrm{clip}
\left(
g_i-g_{min},
0,
\tau_{max}-\tau_{min}-g_{min}
\right)
$$

gap target에는 inverse softplus를 적용한다.

$$
y_{g,i}
=
\mathrm{softplus}^{-1}(g_i')
=
\ln(e^{g_i'}-1)
$$

따라서 모델의 학습 target은 다음과 같다.

$$
\mathbf{y}_i=
[y_{m,i},y_{g,i}]^T
$$

## 5. 샘플 가중치

학습 데이터의 threshold 분포와 hop count 편향을 완화하기 위해 샘플 가중치를 적용한다.

최종 샘플 가중치는 다음과 같다.

$$
w_i=
w_{freq,i}
w_{hop,i}
w_{gap,i}
w_{high,i}
w_{score,i}
$$

high threshold 빈도 기반 가중치는 다음과 같다.

$$
w_{freq,i}
=
\frac{1}
{\sqrt{f(\tau_i^{high})}}
$$

여기서 $f(\tau_i^{high})$는 해당 high threshold 값의 등장 빈도이다.

hop count 기반 가중치는 다음과 같다.

$$
w_{hop,i}
=
\begin{cases}
3.5, & h_i/H_{norm}\geq0.8\\
2.0, & 0.6\leq h_i/H_{norm}<0.8\\
1.0, & h_i/H_{norm}<0.6
\end{cases}
$$

gap 기반 가중치는 다음과 같다.

$$
w_{gap,i}
=
1+
\frac{
g_i-g_{min}
}{
\tau_{max}-\tau_{min}-g_{min}
}
$$

high threshold 크기 기반 가중치는 다음과 같다.

$$
w_{high,i}
=
1+
\frac{
\tau_i^{high}-\tau_{min}
}{
\tau_{max}-\tau_{min}
}
$$

성능 점수 기반 가중치는 다음과 같다.

$$
w_{score,i}
=
\mathrm{Score}_i^\alpha
$$

`main_v.2`에서는 다음 값을 사용한다.

$$
\alpha=1.0
$$

최종적으로 평균 가중치가 1이 되도록 정규화한다.

$$
\tilde{w}_i
=
\frac{
w_i
}{
\frac{1}{N}
\sum_{j=1}^{N}w_j
}
$$

## 6. MLP 기반 Threshold 예측 모델

전처리된 입력 $\mathbf{x}_i$를 사용하여 threshold center와 gap을 예측하는 MLP를 학습한다.

$$
\hat{\mathbf{y}}_i
=
f_{\boldsymbol{\theta}}(\mathbf{x}_i)
$$

모델은 두 개의 은닉층을 갖는다.

$$
\mathbf{h}_{1,i}
=
\mathrm{ReLU}
\left(
W_1\mathbf{x}_i+\mathbf{b}_1
\right)
$$

$$
\mathbf{h}_{2,i}
=
\mathrm{ReLU}
\left(
W_2\mathbf{h}_{1,i}+\mathbf{b}_2
\right)
$$

$$
\hat{\mathbf{y}}_i
=
W_3\mathbf{h}_{2,i}+\mathbf{b}_3
$$

`main_v.2`의 MLP 구조는 다음과 같다.

$$
4\rightarrow32\rightarrow16\rightarrow2
$$

학습 목적 함수는 샘플 가중치를 적용한 평균 제곱 오차로 정의한다.

$$
\mathcal{L}(\boldsymbol{\theta})
=
\frac{1}{N}
\sum_{i=1}^{N}
\tilde{w}_i
\left\|
\mathbf{y}_i
-
\hat{\mathbf{y}}_i
\right\|_2^2
$$

## 7. 예측값의 Threshold 복원

MLP 출력값은 threshold center와 gap으로 복원된다.

center 출력에는 sigmoid 함수를 적용한다.

$$
\hat{\tilde{m}}_i
=
\sigma(\hat{y}_{m,i})
=
\frac{1}{1+e^{-\hat{y}_{m,i}}}
$$

gap 출력에는 softplus 함수를 적용한다.

$$
\hat{g}_i'
=
\mathrm{softplus}
(\hat{y}_{g,i})
=
\ln(1+e^{\hat{y}_{g,i}})
$$

최종 gap은 다음과 같다.

$$
\hat{g}_i=
g_{min}
+
\hat{g}_i'
$$

복원된 center의 허용 범위는 예측 gap을 이용하여 계산한다.

$$
\hat{m}_i^{min}
=
\tau_{min}
+
\frac{\hat{g}_i}{2}
$$

$$
\hat{m}_i^{max}
=
\tau_{max}
-
\frac{\hat{g}_i}{2}
$$

$$
\hat{m}_i
=
\hat{m}_i^{min}
+
\hat{\tilde{m}}_i
\left(
\hat{m}_i^{max}
-
\hat{m}_i^{min}
\right)
$$

최종 low/high threshold는 다음과 같다.

$$
\hat{\tau}_i^{low}
=
\hat{m}_i
-
\frac{\hat{g}_i}{2}
$$

$$
\hat{\tau}_i^{high}
=
\hat{m}_i
+
\frac{\hat{g}_i}{2}
$$

## 논문용 설명 문장

본 연구에서는 RREP decision 이후의 route discovery 결과를 future window 기반으로 집계하여 성능 점수를 정의하였다. 이후 local CBR, neighbor count, hop count, direct route 여부를 입력 특징으로 구성하고, low/high threshold를 직접 예측하는 대신 threshold center와 gap으로 변환하여 MLP regressor를 학습하였다. 또한 high threshold 분포, hop count, threshold gap, high threshold 크기, 성능 점수를 반영한 샘플 가중치를 적용하여 학습 데이터의 편향을 완화하였다. 학습된 MLP의 출력은 sigmoid 및 softplus 변환을 통해 다시 low/high threshold로 복원된다.

