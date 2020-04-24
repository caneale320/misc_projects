import numpy as np
import cv2


def angle_of_skew(image):
    '''
    :param image: black and white image as an array
    :return: the angle by which the image should be rotated to correct skew
    '''
    # find the coordinates of interest (text), which is black so pixel value is 0
    coords_of_interest = np.column_stack(np.where(image == 0)) # change to > 0 if using inverted methodology

    # extract the angle, the last item in the list returned by minAreaRect
    angle = cv2.minAreaRect(coords_of_interest)[-1]

    # adjust the angle for proper rotation results, as minAreaRect only returns between 0 and -90
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -(angle)

    return angle


def rotate_image(mat, angle):
    """
    :param mat: the image to be rotated as an array
    :param angle: the angle by which the image needs to be rotated
    :return: the rotated image
    """

    height, width = mat.shape[:2]
    image_center = (width/2, height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to original) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat
