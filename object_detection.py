import sys
import os
import cv2
import numpy as np
import colorsys
import math
import scipy.ndimage



# Step 1 - converting bgr to rgb to hsl
def convertBGRtoHSL(image):

    
    R = image[:,:,2]
    G = image[:,:,1]
    B = image[:,:,0]

    rgb = np.stack([R, G, B], axis = 2)
    
    R_dash = rgb[:, :, 0] / 255.0
    G_dash = rgb[:, :, 1] / 255.0
    B_dash = rgb[:, :, 2] / 255.0


    c_max = np.maximum(np.maximum(R_dash, G_dash), B_dash)
    c_min = np.minimum(np.minimum(R_dash, G_dash), B_dash)

    L = (c_max + c_min) / 2


    return L



# Step 2 - otsu's threshold method
def otsu(image):

    
    image = cv2.threshold(image, 0.0, 255.0, cv2.THRESH_BINARY, cv2.THRESH_OTSU)[1]

    return image


# Step 3 - filling the holes in a binary image
def fill(img):

    img_output = scipy.ndimage.binary_fill_holes(img).astype(float)

    return img_output



# Step 4 - Median filtering to remove noise from the image
def filter(img):

    img_shape = img.shape

    img_output = img

    for i in range(img_shape[0] - 2):

        for j in range(img_shape[0] - 2):

            window = []

            for k  in range(i - 2, i + 3):

                for l in range(j - 2, j + 3):

                    window.append(img[k][l])
            window.sort()

            img_output[i][j] = window[12]
    
    return img_output



# Step 5 - Sobel operator for edge detection
def sobel_operator(img):

    dx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    dy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    img_output = np.hypot(dx, dy)

    return img_output



# Step 6 - Image thinning
def thinning(img):

   
    thin = np.ones((3, 3), np.float64)

    eroded = cv2.erode(img, thin, iterations = 1)
    
    return eroded



def detection(img):


    resized = np.resize(img, 300)
    ratio = img.shape[0] / float(resized.shape[0])
    
    # step 1
    L = convertBGRtoHSL(img)
    

    # step 2    
    O = otsu(L)
    

    # step 3
    filled_image = fill(O)
    

    # step 4
    image_after_filtering = filter(filled_image)


    # step 5
    sobel_image = sobel_operator(image_after_filtering)
    

    # step 6
    thinned_image = thinning(sobel_image)
    
    # step 7

    
    arr = np.uint8(image_after_filtering)

    _, contours, _ = cv2.findContours(arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]

    shape = ""

    M = cv2.moments(cnt)

    if M["m00"] != 0:
            
        cx = int((M["m10"] / M["m00"]) * ratio)
        cy = int((M["m01"] / M["m00"]) * ratio)

        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)

        


        # Method 1 for shape recognition
        # approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)


        # if(len(approx) == 3):
        #     shape = "Triangle"

        # elif(len(approx) == 4):

        #     x, y, w, h = cv2.boundingRect(approx)

        #     aspect_ratio = w / float(h)

        #     if(aspect_ratio >= 0.95 and aspect_ratio <= 1.05):
        #         shape = "Square"
        #     else:
        #         shape = "Rectangle"
                
        # elif(len(approx) == 5):
        #         shape = "Pentagon"

        # else:
        #         shape = "Circle"





        # Method 2 for shape recognition

        compactness = (perimeter ** 2) / area

        if(compactness <= 14):
            shape = "Circle"
        elif(compactness >=15 and compactness <= 19):
            shape = "Quadrilateral"

        elif(compactness >= 20 and compactness <= 40):
            shape = "Triangle"



        cv2.drawContours(img, [cnt], 0, (0,255,0), 3)

        cv2.putText(img, shape, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        

    cv2.imwrite("circle_result.png", img)
    


img = cv2.imread('1_circle.png')

img_result = detection(img)

