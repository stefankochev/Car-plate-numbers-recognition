import cv2
import numpy as np
import sys


class CarPlateGraphics:

	def __init__(self, name, gui):
		self.image_name = name
		self.gui = gui

	def saveImage(self, image, name):
		path = 'output/{}_{}.png'.format(self.image_name, name)
		cv2.imwrite(path, image)
		self.gui.q.append(path)



	def gaussianBlur(self, image):
		image = cv2.GaussianBlur(
	        image,
	        (5,5),
	        0)

		self.saveImage(image, 'resized')

		return image


	def equalize(self, image):
		image = cv2.equalizeHist(image)
		self.saveImage(image, 'equalized')

		return image


	def cvt(self, image):
		image = cv2.cvtColor(
	        self.make_image_black(
	            cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 8), 
	            cv2.COLOR_BGR2GRAY)
		self.saveImage(image, 'black')

		return image


	def treshold(self, image):
		_, image = cv2.threshold(
			image,
			64, #64
			255, 
			cv2.THRESH_BINARY)
		self.saveImage(image, 'mask')

		return image

	def invert_bitwise(self, image):
		image = cv2.bitwise_not(image)

		self.saveImage(image, 'mask_inverted')

		return image


	def erode(self, image):
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		image = cv2.erode(image, kernel, iterations = 1)

		self.saveImage(image, 'mask_eroded')

		return image


	def make_image_black(self, image, n):
		img = image.reshape((-1,3))
		img = np.float32(img)

		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

		_, label, center=cv2.kmeans(
		    img,
		    n,
		    None, 
		    criteria, 
		    10, 
		    cv2.KMEANS_RANDOM_CENTERS)

		center = np.uint8(center)
		ret = center[label.flatten()]
		ret = ret.reshape((image.shape))

		return ret


	def prepare_image_for_ocr(self, image):

		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		image = cv2.resize(
			image
			, None
			, fx=5.0
			, fy=5.0
			, interpolation=cv2.INTER_CUBIC)

		image = self.gaussianBlur(image)
		image = self.equalize(image)
		image = self.cvt(image)
		image = self.treshold(image)
		image = self.erode(image)

		return image