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
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import(
    Input,Dense,LayerNormalization,MultiHeadAttention,GlobalAveragePooling1D
)
def transformer_encoder(inputs):
    attention_output=MultiHeadAttention(num_heads=4,key_dim=16)(inputs,inputs)
    x=LayerNormalization()(inputs+attention_output)
    ff=Dense(64,activation="relu")(x)
    ff=Dense(inputs.shape[-1])(ff)
    output=LayerNormalization()(x+ff)
    return output
input_layer=Input(shape=(30,1))
x=transformer_encoder(input_layer)
x=GlobalAveragePooling1D()(x)
output=Dense(1)(x)
model=Model(
    input_layer,outputs=output
)
model.compile(optimizer="adam",loss="mse")
# model.summary()
history = model.fit(
    X_train,
    Y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test,Y_test)
)
predictions=model.predict(X_test)
predictions=scaler.inverse_transform(predictions)
print(predictions)
import matplotlib.pyplot as plt
plt.plot(
    scaler.inverse_transform(Y.reshape(-1,1)),
    label="Actual"
)
plt.plot(
    predictions,
    label="Predicted"
)
plt.legend()
plt.show()
