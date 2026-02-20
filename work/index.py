import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2


mnist=tf.keras.datasets.mnist


# #LUAM DATELE PENTRU TRAINING SI TESTING
(x_train, y_train),(x_test,y_test)=mnist.load_data();

#NORMALIZARE DATE SA FIE INTRE 0-1

x_train=tf.keras.utils.normalize(x_train,axis=1);
x_test=tf.keras.utils.normalize(x_test,axis=1);

# # #CREARE MODEL
# model=tf.keras.models.Sequential()

# # #ADAUGARE LAYER,FLATTEN layer transforma intr o singura linie grid ul
# model.add(tf.keras.layers.Flatten(input_shape=(28,28)))

# # #DENSE LAYER, BASIC LAYER, EACH NEURON IS CONECTED TO EACH OTHER NEURON
# # #DENSE : permite modelului sa invete relatii compleze intre pixeli

# model.add(tf.keras.layers.Dense(128,activation='relu'));

# model.add(tf.keras.layers.Dense(128,activation='relu'));

# model.add(tf.keras.layers.Dense(10,activation='softmax'));

# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']);

# model.fit(x_train,y_train,epochs=5);

# model.save('handwritten.keras')

model=tf.keras.models.load_model('handwritten.keras');

loss,accuracy=model.evaluate(x_test,y_test);

print(loss);
print(accuracy);

image_number = 1

while True:
    filepath = f"drawndigits/{image_number}.png"
    
    # Verificăm dacă fișierul există
    if not os.path.isfile(filepath):
        break

    # Citim imaginea în alb-negru
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    # Redimensionăm la 28x28 (MNIST size)
    img = cv2.resize(img, (28, 28))

    # Inversăm culorile (cifre negre pe fundal alb → cifre albe pe fundal negru)
    img = np.invert(img)

    # Normalizăm valorile între 0 și 1
    img = img / 255.0

    # Adăugăm batch dimension (modelul așteaptă shape (1,28,28))
    img = np.array([img])

    # Predicția modelului
    prediction = model.predict(img)

    # Afișăm informațiile
    print(f"Processing {filepath}")
    print(f"This digit is probably a {np.argmax(prediction)}")

    # Arătăm imaginea
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()

    # Trecem la următoarea imagine
    image_number += 1
    
  



