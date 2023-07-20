# IMPORT MODULES
import cv2
import numpy as np
# INPUT THE IMAGE
input_image = cv2.imread("actual_threshold_image.jpg")
# DEFINING FUNCTIONS
def show(caption,image):
    cv2.imshow(caption,image)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()

def convolve_image(filter,image):
    w = image.shape[0] - filter.shape[0] + 1
    h = image.shape[1] - filter.shape[1] + 1
    output_image = np.zeros((w,h))
    for i in range(w):
        for j in range(h):
            patch = image[i:i+filter.shape[0],j:j+filter.shape[1]]
            output_image[i,j] = np.sum(patch*filter)/9
    output_image = output_image.astype(np.uint8)
    return output_image

"""
    GRAYSCALING AN IMAGE
"""
def grayscale(image):
    gray_image = np.zeros((image.shape[0],image.shape[1]))
    for i in range(image.shape[0]):
        ele = []
        for j in range(image.shape[1]):
            b,g,r = image[i,j]
            gray_image[i,j] = int(b/3 + r/3 + g/3)
    np_array = np.array(gray_image)
    output_image = np_array.astype(np.uint8)          # to convert every element to unsigned integer, conversion is done in order to 
    return output_image
#testing out
gray_image = grayscale(input_image)
#show("GRAY IMAGE",gray_image)
"""
    GAUSSIAN SMOOTHING
"""
def gauss_filter(size,sigma):
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    #kernel /= (2 * np.pi * sigma**2)  # Normalize kernel values to sum to 1
    return kernel
gauss_image = convolve_image(gauss_filter(7,1.2),gray_image)
#show("Gauss image",gauss_image)
"""
    SOBEL FILTER
"""
def sobel_filter(input_image):
    filter_hor = np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])
    filter_ver = np.array([[+1, +2, +1], [0, 0, 0], [-1, -2, -1]])
    output_image_x = convolve_image(filter_hor,input_image)
    output_image_y = convolve_image(filter_ver,input_image)
    
    output = np.hypot(output_image_y,output_image_x)
    output = output / output.max() * 255                # normalizing the values
    output = output.astype(np.uint8)
    return output

def scharr_filter(input_image):
    filter_hor = np.array([[+3, 0, -3], [+10, 0, -10], [+3, 0, -3]])
    filter_ver = np.array([[-3, +10, -3], [0, 0, 0], [+3, +10, +3]])
    output_image_x = convolve_image(filter_hor,input_image)
    output_image_y = convolve_image(filter_ver,input_image)
    
    output = np.hypot(output_image_y,output_image_x)
    output = output / output.max() * 255                # normalizing the values
    output = output.astype(np.uint8)
    return output

# scharr_image = scharr_filter(gauss_image)
# show("Scharr Image",scharr_image)
sobel_image = sobel_filter(gauss_image)
show("Sobel Image",sobel_image)

"""
    EDGE THINNING PROCESS
"""
def edge_thinning(input_image):
    thinning_output_image = np.zeros((input_image.shape[0],input_image.shape[1]))
    # now we will extract the pixels in the input image
    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            # finding out if the pixel is a part of the edge
            if input_image[i,j] != 0:
                if((0 < i < input_image.shape[0] and 0 < j < input_image.shape[1] - 1 and input_image[i,j+1] == 0) \
                or (0 < i < input_image.shape[0]-1 and 0 < j < input_image.shape[1]   and input_image[i+1,j] == 0) \
                or (1 < i < input_image.shape[0] and 0 < j < input_image.shape[1]     and input_image[i-1,j] == 0) \
                or (0 < i < input_image.shape[0] and 1 < j < input_image.shape[1]     and input_image[i,j-1] == 0)):
                        thinning_output_image[i,j] = input_image[i,j]
                else:
                    thinning_output_image[i,j] = 0
    thinning_output_image = thinning_output_image/np.max(thinning_output_image) *255   # normalizing shades
    thinning_output_image = thinning_output_image.astype(np.uint8)
    return thinning_output_image

edge_thinning_image = edge_thinning(sobel_image)
print(edge_thinning_image)
show("Edge Thinning image",edge_thinning_image)

"""
    HYSTERESIS THRESHOLDING
"""
def hyst_thresh(image,low_thresh,high_thresh):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] > high_thresh:
                image[i,j] = 255
            elif image[i,j] > low_thresh: 
                if((   0 < i < image.shape[0] and 0 < j < image.shape[1] - 1 and image[i,j+1] > high_thresh) \
                   or ( 0 < i < image.shape[0]-1 and 0 < j < image.shape[1]  and image[i+1,j] > high_thresh) \
                   or ( 1 < i < image.shape[0] and 0 < j < image.shape[1]    and image[i-1,j] > high_thresh) \
                   or ( 0 < i < image.shape[0] and 1 < j < image.shape[1]    and image[i,j-1] > high_thresh)):
                    image[i,j] = 255
            else:
                image[i,j] = 0
hyst_thresh_img = edge_thinning_image
hyst_thresh(hyst_thresh_img,80,200)
show("Hysteresis image",hyst_thresh_img)