import tensorflow
import keras
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 32
IMG = 200
IMG_SIZE = (200, 200)
PATIENCE = 5
EPOCHS = 30
NUMOFCLASSES = 10

model = keras.models.Sequential([
    keras.layers.Conv2D(96, (3,3), padding='same', activation='relu', input_shape = (IMG, IMG, 3)),
    keras.layers.MaxPooling2D(2, 2),

    keras.layers.Conv2D(224, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPool2D(2, 2),

    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPool2D(2, 2),

    keras.layers.Conv2D(96, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPool2D(2, 2),

    keras.layers.Flatten(),
    keras.layers.Dropout(0.5), #jedan od nacina da se zaustavi preobucavanje

    keras.layers.Dense(448, activation='relu'),

    keras.layers.Dense(NUMOFCLASSES, activation='softmax')

])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


train_images = "C:/Users/PC/Desktop/archive/trainValidTestBolji/train"
validation_images = "C:/Users/PC/Desktop/archive/trainValidTestBolji/valid"

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale = 1. / 255, 
    rotation_range = 20 ,
    width_shift_range = 0.2 ,
    height_shift_range = 0.2 ,
    shear_range = 0.2 ,
    zoom_range = 0.2 ,
    horizontal_flip = True
)

training_set = train_datagen.flow_from_directory(
    train_images,
    shuffle=True,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode = 'categorical'
)

valid_datagen = keras.preprocessing.image.ImageDataGenerator(rescale= 1. / 255)

validation_set = valid_datagen.flow_from_directory(
    validation_images,
    shuffle=False,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode = 'categorical'
)



stepsPerEpochs = np.ceil (training_set.samples / BATCH_SIZE)
validationSteps = np.ceil (validation_set.samples / BATCH_SIZE)

best_model_file = "C:/Users/PC/Desktop/archive/bestModel.h5"
stop_early = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=PATIENCE) #drugi od nacina za preobucavanje
model_checkpoint = keras.callbacks.ModelCheckpoint(
    best_model_file, 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)

history = model.fit(
    training_set,
    validation_data=validation_set,
    epochs=EPOCHS,
    steps_per_epoch=stepsPerEpochs,
    validation_steps=validationSteps,
    verbose=1,
    callbacks=[stop_early, model_checkpoint]
)



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.subplot(121)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Accuracy')
plt.subplot(122)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss')
plt.show()