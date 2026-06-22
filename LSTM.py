import numpy as np
data=np.array([
    10,15,12,18,20,
    25,22,28,30,35
])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data= scaler.fit_transform(
    data.reshape(-1,1)
)
def create_sequence(data,window_size):
    X=[]
    Y=[]
    for i in range(len(data)-window_size):
        X.append(data[i:i+window_size])
        Y.append(data[i+window_size])
    return np.array(X),np.array(Y)
X,Y=create_sequence(data,3)
# print(X.shape)
# print(Y) 
X=X.reshape(
    X.shape[0],
    X.shape[1],
    1
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
model=Sequential()
model.add(LSTM(10,input_shape=(3,1)))
model.add(Dense(1))
model.compile(optimizer="adam",loss="mae")
model.fit(X,Y,epochs=100,verbose=1)
prediction=model.predict(X)
prediction=scaler.inverse_transform(prediction)
print(prediction)
import matplotlib.pyplot as plt
plt.plot(
    scaler.inverse_transform(Y.reshape(-1,1)),
    label="Actual"
)
plt.plot(
    prediction,
    label="Predicted"
)
plt.legend()
plt.show()