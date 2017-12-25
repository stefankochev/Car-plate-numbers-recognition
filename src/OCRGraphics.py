import cv2
import numpy as np
import sys


class CarPlateGraphics:

	def __init__(self, name):
		self.image_name = name


	def gaussianBlur(self, image):
		image = cv2.GaussianBlur(
	        image,
	        (5,5),
	        0)
		cv2.imwrite('output/' + self.image_name + '_resized.png', image)

		return image


	def equalize(self, image):
		image = cv2.equalizeHist(image)
		cv2.imwrite('output/' + self.image_name + '_equalized.png', image)

		return image


	def cvt(self, image):
		image = cv2.cvtColor(
	        self.make_image_black(
	            cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 8), 
	            cv2.COLOR_BGR2GRAY)
		cv2.imwrite('output/' + self.image_name + '_black.png', image)

		return image


	def treshold(self, image):
		_, image = cv2.threshold(
			image,
			64, #64
			255, 
			cv2.THRESH_BINARY)
		cv2.imwrite('output/' + self.image_name + '_mask.png', image)

		return image

	def invert_bitwise(self, image):
		image = cv2.bitwise_not(image)
		cv2.imwrite('output/' + self.image_name + '_mask_inverted.png', image)

		return image


	def erode(self, image):
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
		image = cv2.erode(image, kernel, iterations = 1)
		cv2.imwrite('output/' + self.image_name + '_mask_eroded.png', image)

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