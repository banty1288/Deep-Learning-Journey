from tensorflow.keras.applications import ResNet50
base_model=ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)
base_model.trainable=False
print(base_model.output_shape)
base_model.summary()
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
base_model.trainable=False
model=Sequential([
    base_model,GlobalAveragePooling2D(),
    Dense(128,activation="relu"),
    Dense(6,activation="softmax")
])
model.summary()
from tensorflow.keras.optimizers import Adam

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True
print("Trainable Layers:")
for layer in base_model.layers[-20:]:
    print(layer.name)
trainable_count = 0
for layer in model.layers:
    print(layer.name, layer.trainable)
    model.summary()
