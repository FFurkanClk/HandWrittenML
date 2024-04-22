
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

mnist= tf.keras.datasets.mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print('MNIST Dataset Shape:')
print('X_train: ' + str(X_train.shape))
print('Y_train: ' + str(Y_train.shape))
print('X_test:  '  + str(X_test.shape))
print('Y_test:  '  + str(Y_test.shape))

class_names = ['0', '1', '2', '3', '4','5', '6', '7', '8', '9']

#plt.figure()
#plt.imshow(X_train[1])
#plt.colorbar()
#plt.grid(False)
#plt.show()

''' önişleme '''
X_train = X_train / 255.0
X_test = X_test / 255.0



''' Katmanları oluşturma '''
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(128, activation='relu'),     # Relu aktivasonu kullanıyoruz.
    keras.layers.Dense(10)
])

# Modeli hazırlama
model.compile(optimizer='adam',
loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True),
metrics=['accuracy']
)

# Model eğitimi 
model.fit(X_train, Y_train, epochs=10)

# Modelin doğruluğunu ölçüp yazdırma
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose = 2)
print('\nTest Accuracy: ', test_acc)

# Tahmin yaptırma
model=tf.keras.models.load_model("elyazisi.model")
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(X_test)

# print(np.argmax(predictions[0]))

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100*np.max(predictions_array), class_names[true_label]), color = color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color = "#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Verifying predictions of 12th image
i = 10
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], Y_test, X_test)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], Y_test)
plt.show()

# Plotting several images alongwith their predictions
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2*i + 1)
    plot_image(i, predictions[i], Y_test, X_test)
    plt.subplot(num_rows, 2*num_cols, 2*i + 2)
    plot_value_array(i, predictions[i], Y_test)
plt.tight_layout()
plt.show()

#Tekli rakam tahmin ettirmek için ayrı bi kod yazdım dataset dışından veri ekleyip tahmin ettiriyoruz

def predict_digit(image_path):
  
  try:
    # görseli yükleme kodu
    img_new = Image.open(image_path)

    # model 28x28 piksel verileri işleme alıyor datasetin veri boyutu 28x28
    img_array = np.array(img_new)
    resized = cv2.resize(img_array, (28, 28))  # yeni görselin matrisini düzenliyoruz
    gray_scale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    image = cv2.bitwise_not(gray_scale)  # pikseller okunabilsin diye arkaplan rengini değiştiryoruz

    # görsel boyutunu ayarlıyoruz
    image = image.reshape((1, 28, 28))
    image = image / 255.0  # Normalize pixel values

    # tahmin yaptırıyoruz
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)

    return str(predicted_digit)
  except:
    print("Error!")
    return None

image_path = r'C:\Users\furka\Desktop\furkanprojeimg\Project\rakam4.png'
predicted_digit = predict_digit(image_path)

if predicted_digit:
  print("Predicted digit:", predicted_digit)
else:
  print("Prediction Error.")