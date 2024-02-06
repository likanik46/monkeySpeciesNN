from keras.models import load_model
import numpy as np
import keras
import matplotlib.pyplot as plt 
import random
import glob


IMG=200
IMG_SIZE = [IMG,IMG]

NumOfClasses = 10
BatchSize = 32


best_model_file = "C:/Users/PC/Desktop/archive/bestModel.h5"
model = load_model(best_model_file)


testMyImagesFolder = "C:/Users/PC/Desktop/archive/trainValidTestBolji/test"
test_datageb = keras.preprocessing.image.ImageDataGenerator(rescale = 1. /255)
test_set = test_datageb.flow_from_directory(testMyImagesFolder,
                                                shuffle=False, 
                                                target_size = IMG_SIZE,
                                                batch_size = BatchSize,
                                                class_mode = 'categorical')


predictions = model.predict(test_set)
predictionsResults = np.argmax(predictions, axis=1)
print(predictionsResults)


columns = ["Label","Latin Name","Common Name","Train Images","Validation Images"]

import pandas as pd

df = pd.read_csv("C:/Users/PC/Desktop/archive/monkey_labels.txt",names=columns,skiprows=1)

df['Label'] = df['Label'].str.strip()
df['Latin Name'] = df['Latin Name'].replace("\t","")
df['Latin Name'] = df['Latin Name'].str.strip()
df['Common Name'] = df['Common Name'].str.strip()

df= df.set_index("Label")

print(df)



monkeyDic = df["Common Name"]
print(monkeyDic)



def compareResults():
    image_files = glob.glob(testMyImagesFolder + '/*/*.jpg')
    nrows = 5
    ncols = 6
    picnum = nrows * ncols

    fig , ax = plt.subplots(nrows , ncols , figsize=(3*ncols , 3*nrows))
    fig.subplots_adjust(hspace=0.5) 
    correct = 0

    for i in range(picnum) :
        x = random.choice(image_files)
        xi = image_files.index(x)
        img1 = plt.imread(x)

        pred1 = monkeyDic[predictionsResults[xi]]
        pred1 = pred1[:7]

        real1 = monkeyDic[test_set.classes[xi]]
        real1 = real1[:7]

        if (pred1 == real1 ):
            correct = correct + 1

        name = 'predicted : {} \nreal: {}'.format(pred1,real1)
        plt.imshow(img1)
        plt.title(name)

        sp = plt.subplot(nrows,ncols, i+1 )
        sp.axis('off') 

    print(" ======================================================= ")
    print("Ukupno : {} Tacno : {}".format(picnum , correct))

    plt.show()

compareResults()