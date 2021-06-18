# import delle librerie utili
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# prende dimensione dell'immagine e la trasforma in un dato binario
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))
	# attraverso passaggio del dato binario rilevazione faccia
	faceNet.setInput(blob)
	detections = faceNet.forward()
	# inizializazione delle liste delle facce, e delle predizioni fatte dall'algoritmo per il riconoscimeto delle facce
	faces = []
	locs = []
	preds = []
    	# loop per riconoscimento delle facce
	for i in range(0, detections.shape[2]):
		# estrazione probabilità del riconoscimento attraverso parametro confidence
		confidence = detections[0, 0, i, 2]
		# filtra i riconoscimenti in base al parametro confidence
		if confidence > args["confidence"]:
			# elabora cordinate del rettangolino nell'immagine
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			# si fa in modo che il rettangolo non esca dalla finestra del programma 
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            # estrae faccia riconverte con colori RGB e ridimensionamento a 224x224
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			# aggiunge la faccia e il rettangolino alle proprie liste
			faces.append(face)
			locs.append((startX, startY, endX, endY))
            # fai rilevamento solo se faccia presente nel riquadro
	if len(faces) > 0:
		# per renderlo più veloce prende immagini sorgente a lotti(gruppi) e non singole
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	# return di 2 tule per locazione della faccia
	return (locs, preds)
# costuzione analizatore e analizare
ap = argparse.ArgumentParser()
ap.add_argument("-f", "-res10_300x300_ssd_iter_140000.caffemodel", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "-model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
# caricare da locale modello per rilevazione del viso 
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join(["deploy.prototxt"])
weightsPath = os.path.sep.join(["res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
# caricare da locale modello per rilevazione della mascherina
print("[INFO] loading face mask detector model...")
maskNet = load_model("model")
# inizializazione webcam 
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
# loop per i frame della webcam
while True:
	# prendi input webcam massimo a 400pixel 
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	# riconoscimento della mascherina 
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    # una volta trovata faccia continuo loop su di essa per aggiornare la valutazione
	for (box, pred) in zip(locs, preds):
		# unpack dei dati rilevati
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
		# determina colore del rettangolino
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		# percentuale di mask o no
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		# print del rettangolino con %
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        	# mostrare output
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# Si spenge con la q
	if key == ord("q"):
		break
# chiude tutto e 'pulisce'
cv2.destroyAllWindows()
vs.stop()
