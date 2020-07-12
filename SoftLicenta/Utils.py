from scipy.spatial import distance as dist
import numpy as np
import cv2
from scipy import ndimage

# This function is used for seperating the digit from noise in "crop_image"
# The Sudoku board will be chopped into 9x9 small square image,
# each of those image is a "crop_image"
def largest_connected_component(image):

    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[:, -1]

    if(len(sizes) <= 1):
        blank_image = np.zeros(image.shape)
        blank_image.fill(255)
        return blank_image

    max_label = 1
    # Start from component 1 (not 0) because we want to leave out the background
    max_size = sizes[1]     

    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2.fill(255)
    img2[output == max_label] = 0
    return img2

def get_best_shift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    return shiftx, shifty

# Shift the image using what get_best_shift returns
def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted


# Prepare and normalize the image to get ready for digit recognition
def prepare(img_array):
    new_array = img_array.reshape(1, 32, 32, 1)
    #print(new_array)
    new_array = new_array.astype('float32')
    new_array /= 255
    return new_array


#Function to print the grid
def printGrid(grid):
    for row in grid:
        print(row)
