from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import os
import imutils
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def load_image(path):
	return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)


train_dir_pos = 'train/pos/'
train_dir_neg = 'train/neg/'

pos_imgs = []
neg_imgs = []

for img_name in os.listdir(train_dir_pos):
	img_path = os.path.join(train_dir_pos, img_name)
	img = load_image(img_path)
	pos_imgs.append(img)

for img_name in os.listdir(train_dir_neg):
	img_path = os.path.join(train_dir_neg, img_name)
	img = load_image(img_path)
	neg_imgs.append(img)


pos_features = []
neg_features = []
labels = []

nbins = 9
cell_size = (8, 8)
block_size = (3, 3)

hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], img.shape[0] // cell_size[0] * cell_size[0]),
						_blockSize=(block_size[1] * cell_size[1], block_size[0] * cell_size[0]),
						_blockStride=(cell_size[1], cell_size[0]),
						_cellSize=(cell_size[1], cell_size[0]),
						_nbins=nbins)

for img in pos_imgs:
	pos_features.append(hog.compute(img))
	labels.append(1)

for img in neg_imgs:
	neg_features.append(hog.compute(img))
	labels.append(0)

pos_features = np.array(pos_features)
neg_features = np.array(neg_features)
x = np.vstack((pos_features, neg_features))
y = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


def reshape_data(input_data):
	nsamples, nx, ny = input_data.shape
	return input_data.reshape((nsamples, nx*ny))


x_train = reshape_data(x_train)
x_test = reshape_data(x_test)

clf_svm = SVC(kernel='linear', probability=True)
clf_svm.fit(x_train, y_train)
y_train_pred = clf_svm.predict(x_train)
y_test_pred = clf_svm.predict(x_test)
print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))

imagePaths = list(paths.list_images("images"))


def sliding_window(image, stepSize, windowSize):
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def pyramid(image, scale=1.5, minSize=(30, 30)):
	yield image

	while True:
		width = int(image.shape[1] / scale)
		image = imutils.resize(image, width=width)

		if image.shape[0] >= minSize[1] or image.shape[1] >= minSize[0]:
			yield image


matches = []
for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=min(400, image.shape[1]))
	downscale = 1.3
	windowWidth = 64
	windowHeight = 128
	scale = 0
	for (i, resized) in enumerate(pyramid(image, scale=downscale)):

		if resized.shape[0] < windowHeight or resized.shape[1] < windowWidth:
			break

		for (x, y, window) in sliding_window(resized, stepSize=6, windowSize=(windowWidth, windowHeight)):

			if window.shape[0] != windowHeight or window.shape[1] != windowWidth:
				continue
			greyscale = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
			features = hog.compute(greyscale)
			fr = features.reshape(1, -1)
			prediction = clf_svm.predict(fr)

			if prediction != 1:
				continue
			else:
				if clf_svm.decision_function(fr) > 0.8:
					matches.append(
						(int(x * (downscale ** scale)), int(y * (downscale ** scale)), clf_svm.decision_function(fr),
						 int(64 * (downscale ** scale)), int(128 * (downscale ** scale))))
		scale += 1
	clone = image.copy()

	found = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in matches])

	sc = [score[0] for (x, y, score, w, h) in matches]
	sc = np.array(sc)
	pick = non_max_suppression(found, probs=sc, overlapThresh=0.3)

	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(clone, (xA, yA), (xB, yB), (255, 255, 0), 2)

	cv2.imshow("Detected", clone)
	cv2.waitKey(0)
