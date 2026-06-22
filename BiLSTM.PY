import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(42)
days=np.arange(1000)
data=(
    np.sin(days*0.05)*20+days*0.02+np.random.normal(0,2,1000)
)
df=pd.DataFrame({"Value":data})
# print(df.head())
# plt.figure(figsize=(10,5))
# plt.plot(df["Value"])
# plt.show()
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaled_data=scaler.fit_transform(
    df[["Value"]]
)
def create_sequences(data, window_size):
    X = []
    y = []
    for i in range(len(data)-window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)
X,Y=create_sequences(
    scaled_data,30
)
split=int(len(X)*0.8)
X_train=X[:split]
X_test=X[split:]
Y_train=Y[:split]
Y_test=Y[split:]
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
model=Sequential()
model.add(Bidirectional(LSTM(50,return_sequences=False),input_shape=(30,1)))
model.add(Dense(1))
model.summary()
model.compile(optimizer="adam",
              loss="mse")
history=model.fit(
    X_train,Y_train,epochs=20,batch_size=32,validation_data=(X_test,Y_test)
)
prediction=model.predict(X_test)
prediction=scaler.inverse_transform(prediction)
actual=scaler.inverse_transform(
    Y_test.reshape(-1,1)
)
plt.figure(figsize=(12,6))
plt.plot(
    actual,
    label="Actual"
)
plt.plot(
    prediction,
    label="Predicted"
)
plt.legend()
plt.show()