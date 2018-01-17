import cv2
import numpy as np
import sys
import CarPlateOCR as cpo
import OCRGraphics as og
import os
import knn



class OCR:

	def __init__(self, image_name, gui):
		self.image_name = image_name
		self.graphics = og.CarPlateGraphics(image_name, gui)
		self.gui = gui

	def is_probably_character(self, w, h, col):
	    area = w*h

	    if (area > 5000 and area < 25000):
	        return True

	    return False

	def is_character_for_sure(self, bb, mean, median, meanW, meanH, medianW, medianH):
		center, (x,y,w,h) = bb

		if(abs(w*h - median) > 10000):
			return False
		
		if(abs(w - meanW) > 13 and abs(h - meanH) > 13):
			return False
		
		if(abs(h - meanH) > 30):
			return False
		
		return True



	def get_bb_mean_and_median(self, bounding_boxes):
		widths = []
		heights = []
		areas = []
		for _, (_,_,w,h) in bounding_boxes:
			widths.append(w)
			heights.append(h)
			areas.append(w*h)

		mean = np.mean(areas)
		median = np.median(areas)
		meanW = np.mean(widths)
		meanH = np.mean(heights)
		medianW = np.median(widths)
		medianH = np.median(heights)

		return mean, median, meanW, meanH, medianW, medianH

	def get_all_characters(self, image):
		img_inverted = self.graphics.invert_bitwise(image)
		contours = cv2.findContours(img_inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

		char_mask = np.zeros_like(image)

		bounding_boxes = []
		for contour in contours:
			x,y,w,h = cv2.boundingRect(contour)
			area = w * h
			center = (x + w/2, y + h/2)
			if self.is_probably_character(w, h, bounding_boxes):
				x, y, w, h = x-4, y-4, w+8, h+8
				bounding_boxes.append((center, (x,y,w,h)))

		mean, median, meanW, meanH, medianW, medianH = self.get_bb_mean_and_median(bounding_boxes)

		bounding_boxes = filter(lambda bb: self.is_character_for_sure(bb, mean, median, meanW, meanH, medianW, medianH) , bounding_boxes)

		for center, (x,y,w,h) in bounding_boxes:
			cv2.rectangle(
				char_mask,
				(x,y),
				(x+w, y+h),
				255,
				-1)

		self.graphics.saveImage(char_mask, 'mask_squares')

		img_clean = cv2.bitwise_not(cv2.bitwise_and(char_mask, char_mask, mask = img_inverted))

		bounding_boxes = sorted(bounding_boxes, key=lambda item: item[0][0])  

		characters = []
		for center, bbox in bounding_boxes:
			x,y,w,h = bbox
			char_image = img_clean[y:y+h, x:x+w]
			characters.append((bbox, char_image))

		return img_clean, characters


	def highlight_characters(self, image, chars):
		output_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
		for bbox, _ in chars:
			x,y,w,h = bbox
			cv2.rectangle(
				output_img,
				(x,y),
				(x+w, y+h),
				255,
				1)

		return output_img


	def process_image(self, path):
		if not os.path.exists("output"):
			os.makedirs("output")

		img = cv2.imread(path)
		img = self.graphics.prepare_image_for_ocr(img)

		clean_img, chars = self.get_all_characters(img)
		output_img = self.highlight_characters(clean_img, chars)
		self.graphics.saveImage(output_img, 'out')

		samples = np.loadtxt('char_samples3.data',np.float32)
		responses = np.loadtxt('char_responses3.data',np.float32)
		responses = responses.reshape((responses.size,1))

		#model = cv2.ml.KNearest_create()
		#model.train(
		#	samples, 
		#cv2.ml.ROW_SAMPLE, 
		#	responses)

		model2 = knn.Knn(samples, responses)

		plate_chars = ""
		for _, char_img in chars:
			try:
				small_img = cv2.resize(char_img,(10,10))
				small_img = small_img.reshape((1,100))
				small_img = np.float32(small_img)
				#retval, results, neigh_resp, dists = model.findNearest(small_img, k = 3)
				result = model2.find_nearest(small_img, k = 3)
				plate_chars += str(chr(int(result)))

			except ValueError as err:
				print(err)

		print("Licence plate: %s" % plate_chars)

		return plate_chars
