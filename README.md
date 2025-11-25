# 🧠 LSTM Sequence Sum Prediction

LSTM(Long Short-Term Memory) 모델을 활용하여 간단한 시퀀스 입력값의 합을 예측하는 딥러닝 실습 프로젝트입니다.

## 📌 프로젝트 개요

**목적**: 주어진 정수 시퀀스(예: [1, 2, 3])의 합을 예측하도록 LSTM 모델 학습

### 모델 구조
- **입력**: 길이 3의 정수 시퀀스
- **모델**: LSTM → Dense(1)
- **출력**: 예측된 합 (float)
- **손실 함수**: MSE (Mean Squared Error)
- **평가지표**: MAE (Mean Absolute Error)

## ⚙️ 입력 전처리 핵심

```python
arr = np.array(seq, dtype="float32").reshape(1, seq_len, 1)
pred = model.predict(arr, verbose=0)[0, 0]
```

- 입력 시퀀스를 `(batch, time_step, feature)` 형태로 변환
- 모델의 예측 결과에서 스칼라 값을 추출

## 🧩 주요 학습 포인트

1. **LSTM 입력 형식** - `(batch, sequence_length, feature_dim)` 구조의 이해
2. **시계열 데이터 예측** - RNN 구조의 기본 원리 학습
3. **성능 평가** - MSE/MAE 손실 비교를 통한 모델 성능 해석

## 📚 기술 스택

- **Language**: Python
- **Library**: TensorFlow / Keras, NumPy, Matplotlib

## 🏁 결과 요약

간단한 수열 예측 문제를 통해 LSTM의 시계열 데이터 처리 능력과 모델 입력 형태(3D 텐서) 구조를 실습했습니다.

---

### 🚀 Getting Started

```bash
# 필요한 라이브러리 설치
pip install tensorflow numpy matplotlib

# 모델 학습 및 테스트
python train.py
```

### 📊 Example

```python
# 입력: [1, 2, 3]
# 실제 합: 6
# 예측 결과: 5.98
```
