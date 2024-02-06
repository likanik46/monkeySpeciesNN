from keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

IMG_SIZE = (200, 200)
BatchSize = 32
best_model_file = "C:/Users/PC/Desktop/archive/bestModel.h5"
testMyImagesFolder = "C:/Users/PC/Desktop/archive/trainValidTestBolji/test"
classes = ["mantled_howler", "patas_monkey", "bald_uakari", "japanese_macaque", "pygmy_marmoset", "white_headed_capuchin", "silvery_marmoset", "common_squirrel_monkey", "black_headed_night_monkey", "nilgiri_langur"]


model = load_model(best_model_file)


test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory(testMyImagesFolder,
                                            shuffle=False,  
                                            target_size=IMG_SIZE,
                                            batch_size=BatchSize,
                                            class_mode='categorical')


num_samples = test_set.samples
num_batches = int(np.ceil(num_samples / BatchSize))

labels = np.array([])
pred = np.array([])

for i in range(num_batches):
    img_batch, lab_batch = next(test_set)
    labels = np.append(labels, np.argmax(lab_batch, axis=1))
    pred = np.append(pred, np.argmax(model.predict(img_batch, verbose=0), axis=1))
    print(f"Processed batch {i+1}/{num_batches}")


accuracy = accuracy_score(labels, pred)
print('Tacnost modela je: ' + str(100 * accuracy) + '%')

# Generate confusion matrix
cm = confusion_matrix(labels, pred, normalize='true')
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)


plt.figure(figsize=(10, 8))
cm_display.plot()
plt.xticks(rotation=45)  
plt.show()
