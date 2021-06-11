from keras.utils.np_utils import to_categorical
import os
import cv2
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import *
import os

dataPath = './dataset'
data_dir_list = os.listdir(dataPath)
# ['happy', 'sadness', 'neutral', 'anger']
print(data_dir_list)

imgDataList = []

for dataset in data_dir_list:
    img_list = os.listdir(dataPath+'/' + dataset)
    print('Loaded the images of dataset-'+'{}\n'.format(dataset))
    # print(img_list)  Btüün image datası
    for img in img_list:
        inputImg = cv2.imread(dataPath + '/' + dataset + '/' + img)
        inputImgResize = cv2.resize(inputImg, (48, 48))
        # fotolar zaten 48x48. Emin olmak için resize yapıyorum
        imgDataList.append(inputImgResize)

imgData = np.array(imgDataList)
imgData = imgData.astype('float32')
imgData = imgData/255  # Normalization

num_classes = 4

# 981  #(shape, dtype, order) arr nin içi o yüzden 0 ı alıyoruz. Kaç tane img datası oldugunu gösteriyor
num_of_samples = imgData.shape[0]
print(num_of_samples)

# 1lerden olusan 981 elemanlık array olustrudum
labels = np.ones((num_of_samples), dtype='int64')
# print(labels)

labels[0:420] = 0  # 421 happy
labels[421:740] = 1  # 320 sadness
labels[741:1055] = 2  # 315 neutral
labels[1056:1271] = 3  # 216 anger
# print(labels)

Y = to_categorical(labels, num_classes)
# Çoklu sınıflandırma ile ilgilendiğimiz için etiketleri kategorik olarak etiketlememiz gerekiyor
# Elimizde bulunan sayıları(yani resimlerdeki sayıları 0 dan 6 ya kadar) encode ederek
# başka formata cevirdik
# 0 => [1,0,0,0,0,0,0,0,0,0]
# 2 => [0,0,1,0,0,0,0,0,0,0]
# 9 => [0,0,0,0,0,0,0,0,0,1]
# Burada bizim yaptığımız encoding, one-hot-encoding olarak geçer
# print(Y)

# Shuffle the dataset
x, y = shuffle(imgData, Y, random_state=2)
# Shuffle işlemi arrlerin içinde indexlerin yerlerini değiştiriyor. Burda imgData içindeki imageların sırasını rastgele değiştirip x e yazdık.
# Aynı şekilde Y içindekileri rastgele değiştirip y ye yazdık.
# print(x)
# print(y)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.05, random_state=2)
# Şimdi burada x in bir kısmını test olarak alıcam ve uyg içinde test olarak onu kullanıcam.
# test_size = datasetimin %95 train %5 validation a ayır demek.
# random_state ise yapılan işlemin belli bir sırada yapılmasını sağlıyor.

input_shape = (48, 48, 3)
# png datayı direkt kullandığımız için 3 kanalı var. digitRecognition da 1 demiştim çünkü ordaki datalar png değil csv içindeydi.

model = Sequential()  # modeli oluşturduk.
# 1
model.add(Conv2D(filters=6, kernel_size=(5, 5),
          input_shape=input_shape, padding='Same', activation='relu'))
# filters = convolution layerdaki filtre sayısı
# kernel_size = filtrenin boyutu.
# SamePadding kullanıyoruz(kenarlara 0 koyarak, input size = output size).
# Activation func. = relu
model.add(MaxPooling2D(pool_size=(2, 2)))

# 2
model.add(Conv2D(filters=16, kernel_size=(5, 5),
          padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 3
model.add(Conv2D(filters=64, kernel_size=(5, 5),
          padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


# Flatten -> matrixi tek sütun haline getirme işlemi. ANN için
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

# Adam Optimizer
# Learning rateimiz normalde  sabittir. Adam Optimizer kullanarak değiştirebiliyoruz.
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

# loss functionı categorical_crossentropy ile buluyoruz. Eğer yanlış predict ederse loss yüksek, doğru predict ederse loss 0.
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
hist = model.fit(x_train, y_train, batch_size=8, epochs=50, verbose=1,
                 validation_data=(x_test, y_test))

# Epoch and Batch Size
# 981 resmimiz var. batch__size = 64 dedim.  981/64 = 15,33 kez batch yaparız.
# Bu da epoch olarak adlandırılır. Her epochta 15,33 kez batch yapıyoruz demek.

# Evaluate model
score = model.evaluate(x_test, y_test)
print('Test accuracy:', score[1])

model.save('main.model')
