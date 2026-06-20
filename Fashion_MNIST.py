from tensorflow.keras.datasets import fashion_mnist
(X_train,y_train),(X_test,y_test)=fashion_mnist.load_data()
# print(X_train.shape)
# print(y_train.shape)
X_train=X_train.reshape(-1,28,28,1)
X_test=X_test.reshape(-1,28,28,1)
X_train=X_train/255.0
X_test=X_test/255.0
from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
model=Sequential([
Convolution2D(32,(3,3),activation="relu",input_shape=(28,28,1)),
MaxPooling2D((2,2)),
Convolution2D(64,(3,3),activation="relu"),
MaxPooling2D((2,2)),
Flatten(),
Dense(128,activation="relu"),
Dense(10,activation="softmax")])
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen=ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

history=model.fit(datagen.flow(X_train,y_train,batch_size=32),validation_data=(X_test,y_test),epochs=10)
model.summary()