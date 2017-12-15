import cv2
import numpy as np
import sys

def make_image_black(image, n):
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


def prepare_image(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img_resized = cv2.resize(
        img_gray
        , None
        , fx=5.0
        , fy=5.0
        , interpolation=cv2.INTER_CUBIC)

    img_resized = cv2.GaussianBlur(
        img_resized,
        (5,5),
        0)
    cv2.imwrite('licence_plate_resized.png', img_resized)

    img_equalized = cv2.equalizeHist(img_resized)
    cv2.imwrite('licence_plate_equalized.png', img_equalized)

    img_black = cv2.cvtColor(
        make_image_black(
            cv2.cvtColor(img_equalized, cv2.COLOR_GRAY2BGR), 8), 
            cv2.COLOR_BGR2GRAY)
    cv2.imwrite('licence_plate_black.png', img_black)

    ret, mask = cv2.threshold(
        img_black, 
        64, 
        255, 
        cv2.THRESH_BINARY)
    cv2.imwrite('licence_plate_mask.png', mask) 

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, kernel, iterations = 1)
    cv2.imwrite('licence_plate_mask_eroded.png', mask)

    return mask


def is_character(w, h, col):
    area = w*h

    if (area > 7000 and area < 25000):
        return True

    return False


def get_all_characters(image):
    img_inverted = cv2.bitwise_not(image)
    cv2.imwrite('licence_plate_mask21.png', img_inverted) #only for debug, delete before release
    
    contours = cv2.findContours(img_inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

    char_mask = np.zeros_like(image)

    bounding_boxes = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        area = w * h
        center = (x + w/2, y + h/2)
        if is_character(w, h, bounding_boxes):
            x, y, w, h = x-4, y-4, w+8, h+8
            bounding_boxes.append((center, (x,y,w,h))) #add filter after for loop and then second loop for creating rectangles
            cv2.rectangle(
                char_mask,
                (x,y),
                (x+w, y+h),
                255,
                -1)

    cv2.imwrite('licence_plate_mask_squares.png', char_mask)

    img_clean = cv2.bitwise_not(cv2.bitwise_and(char_mask, char_mask, mask = img_inverted))

    bounding_boxes = sorted(bounding_boxes, key=lambda item: item[0][0])  

    characters = []
    for center, bbox in bounding_boxes:
        x,y,w,h = bbox
        char_image = img_clean[y:y+h, x:x+w]
        characters.append((bbox, char_image))

    return img_clean, characters


def highlight_characters(image, chars):
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


def process_image(path):
    img = cv2.imread(path)
    img = prepare_image(img)

    clean_img, chars = get_all_characters(img)
    output_img = highlight_characters(clean_img, chars)
    cv2.imwrite('licence_plate_out.png', output_img)

    samples = np.loadtxt('char_samples2.data',np.float32)
    responses = np.loadtxt('char_responses2.data',np.float32)
    responses = responses.reshape((responses.size,1))

    model = cv2.ml.KNearest_create()
    model.train(
        samples, 
        cv2.ml.ROW_SAMPLE, 
        responses)

    plate_chars = ""
    for _, char_img in chars:
        small_img = cv2.resize(char_img,(10,10))
        small_img = small_img.reshape((1,100))
        small_img = np.float32(small_img)
        retval, results, neigh_resp, dists = model.findNearest(small_img, k = 1)
        plate_chars += str(chr((results[0][0])))
    print("Licence plate: %s" % plate_chars)
    return plate_chars
