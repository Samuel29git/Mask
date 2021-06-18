# importa liberire utili
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

#collega drive per i file
from google.colab import drive
drive.mount('/content/drive')

#path immagini del database 
path_images = '/content/drive/MyDrive/mask_detector/experiements/data'

# inizializazione immagini dal database
print("[INFO] loading images...")
imagePaths = list(paths.list_images(path_images))
data = []
labels = []

# loop per caricamento di ogni immagine presente nella cartella
for imagePath in imagePaths:

	# extract the class label from the filename
	label = imagePath.split(os.path.sep)[-2]

	# immagine convertita 224x224 per funzionare
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	# aggiorna le immagini e gli strati dopo la conversione
	data.append(image)
	labels.append(label)

# converte dati in array NumPy
data = np.array(data, dtype="float32")
labels = np.array(labels)

# perform one-hot encoding sui livelli cioè se '0 1' con mask e senza '1 0' utilizza dei codici binari non un unico carattere
lb = LabelBinarizer() 
labels = lb.fit_transform(labels)  
labels = to_categorical(labels)  

# 80% per addestramento dell'algoritmo
# 20% per test dell'algoritmo
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# costruzione del database aumentato(data augmentation) girando ribaltando oppure sfocando le immagini
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")
  
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
# costruzione testa del modello perchè si sta utilizzando un qualcosa già creato ma con funzioni differenti
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
# posizionamento della testa in 'testa' all'algoritmo per garantire funzionamento
model = Model(inputs=baseModel.input, outputs=headModel)
# loop tutti quanti i livelli e frezzarli ovvero bloccarli in modo che non vengano modificati durante l'esecuzione
for layer in baseModel.layers:
	layer.trainable = False
  
# inizializazion dell'allenamento dell epoche e della grandezza dei lotti(BS) di immagini da calcolare durante le epoche
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# copilazione modello
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# allenamento testa precedentemente creata 
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# raffigurazione della perdità(loss) e dell'accuratezza(accuracy) del programma
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

# predizioni del test set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
# classificazione delle immagini del test set
predIdxs = np.argmax(predIdxs, axis=1)
# abbelimento dati nel print
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))
# salvataggio del modello in modo da non doverlo riavviare ogni volta
print("[INFO] saving mask detector model...")
model.save("model", save_format="h5")
