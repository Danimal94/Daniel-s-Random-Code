import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

img = cv.imread(r"C:\Users\danie\Desktop\Python Practice\pythonProject\Lib\20.jpeg")

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data() #mnist= the stock dataset of handwritten digits

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape = (28,28)))
#28x28 input shape is number of pixels in each image

model.add(tf.keras.layers.Dense(units= 128, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(units= 128, activation = tf.nn.relu))
#connecting neuron layers from one layer to the next, more neurons = more computing power more sophistication more accuracy
#rectified linear unit : The Rectified Linear Unit is the most commonly used activation function in deep learning models.
# The function returns 0 if it receives any negative input, but for any positive value x it returns that value back.

model.add(tf.keras.layers.Dense(units = 10 , activation = tf.nn.softmax))
#creating output neurons. softmax scales percentages of each final neuron to sum to 1(100%) giving the probability score.

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
model.fit(x_train, y_train, epochs = 3)
accuracy, loss = model.evaluate(x_test, y_test)
print(accuracy)
print(loss)

model.save('digits.model')


########################################################################################
def rescaleFrame(frame, scale = 1): #this function works  for images, video and live videeo
    width = 28
    height = 28

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)
########################################################################################

rescaled_image = rescaleFrame(img)
gray = cv.cvtColor(rescaled_image, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)

canny = cv.Canny(blur, 125, 175)
cv.imshow('Blur', canny)

contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

img1 = canny
img1 = np.array([img1])
prediction = model.predict(img1)
print(f' the result is probably {np.argmax(prediction)}')  # argmax gives result with highest probability
plt.imshow(img[0], cmap=plt.cm.binary)
plt.show()

cv.waitKey(0)