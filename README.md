🧠 LSTM Sequence Sum Prediction

이 프로젝트는 LSTM(Long Short-Term Memory) 모델을 활용하여
간단한 시퀀스(sequence) 입력값의 합(sum) 을 예측하는 실습입니다.

📌 프로젝트 개요

목적: 주어진 정수 시퀀스(예: [1, 2, 3])의 합을 예측하도록 LSTM 모델을 학습

모델 구조:

입력: 길이 3의 정수 시퀀스

모델: LSTM → Dense(1)

출력: 예측된 합 (float)

손실 함수: MSE (Mean Squared Error)

평가지표: MAE (Mean Absolute Error)

🚀 학습 코드 요약
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


학습 데이터의 20%를 검증(validation)으로 사용

Epoch마다 MSE/MAE 변화를 확인하며 학습 진행

📈 모델 평가
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MSE: {loss:.4f}, Test MAE: {mae:.4f}")


결과 예시
Test MSE: 1.0850, Test MAE: 0.5713

🔍 예시 예측 결과
demo_example([1, 2, 3])
demo_example([4, 0, 9])
demo_example([7, 7, 7])


출력 예시:

입력 시퀀스: [1, 2, 3]
정답(세 수의 합): 6
LSTM 예측값 :  5.98
----------------------------------------
입력 시퀀스: [7, 7, 7]
정답(세 수의 합): 21
LSTM 예측값 :  21.05
----------------------------------------

⚙️ 입력 전처리 핵심
arr = np.array(seq, dtype="float32").reshape(1, seq_len, 1)
pred = model.predict(arr, verbose=0)[0, 0]


입력 시퀀스를 (batch, time_step, feature) 형태로 변환

모델의 예측 결과에서 스칼라 값을 추출

🧩 주요 학습 포인트

LSTM 입력 형식 (batch, sequence_length, feature_dim)의 이해

시계열 데이터 예측을 위한 RNN 구조의 기본 원리

MSE/MAE 손실 비교를 통한 성능 해석

📚 기술 스택

Language: Python

Library: TensorFlow / Keras, NumPy, Matplotlib

🏁 결과 요약

간단한 수열 예측 문제를 통해
LSTM의 시계열 데이터 처리 능력과
모델 입력 형태(3D 텐서) 구조를 실습했습니다.
