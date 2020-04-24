import numpy as np
import argparse
import cv2
from PyPDF2 import PdfFileReader
import pdf2image as pdf2
import pandas as pd
from hough_space_func import hori_or_not
import time
import math
from image_rotation import angle_of_skew, rotate_image
from PIL import Image
import os

# construct argument parse object and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument('-in', '--input_path', required=True, help='path to folder containing pdf and/or tiff files')
ap.add_argument('-out', '--output_path', required=True,
                help='path to out folder containing processed pdf and/or tiff files')

# get all variables from the parsed arguments
args = vars(ap.parse_args())
in_path = args['input_path']
out_path = args['output_path']

files = os.listdir(in_path)


def create_images(path, start, stop):
    """
    :param path: location of pdf document or tiff image to convert to images
    :param start: index of starting page using zero based indexing
    :param stop: every page up to and including this page will be converted
    :return: single column df of converted images
    """

    # create pdf object from path
    pages = pdf2.convert_from_path(path, 500, thread_count=4, first_page=start, last_page=stop)

    # create data frame for use of .map and .apply functions
    df = pd.DataFrame(pages, columns=['PIL_images'])
    return df


def convert_to_np(page):
    '''
    :param page: a page from the document being worked on, in PIL form
    :return: the image of the page with pixel values stored as an np array
    '''
    # convert to RGB
    pil_image = page.convert('RGB')
    # convert to np array for open CV
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR because OpenCV is only compatible with BGR format
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def check_np_images(image):
    '''
    :param image: any image stored as an np array
    :return: a plot of the image using pyplot
    '''
    from matplotlib import pyplot as plt

    plt.imshow(image, interpolation='nearest')
    plt.show()


def grey_and_thresh(image):
    '''
    :param image: an image stored in an np array
    :return: the image converted to black and white
    '''
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    '''
    # invert image for white text on black background (optional, can test with and and without)
    # other parameters to change if image is inverted, all noted in respective function definitions
    gray_image = cv2.bitwise_not(gray_image)
    '''
    # threshold image into black and white for more accurate processing
    thresholded = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return thresholded


def chunk(path):
    """
    :param path: path of the pdf document for COR/processing
    :return: a list of the page numbers at the beginning and end of each chunk
    global variable for number of chunks and length of pdf
    """
    global len_pdf
    global num_chunks

    '''
    determines a number of chunks based on processing 15 pages at a time 
    (different amount could be chosen based on available computing power
    '''
    num_chunks = math.ceil(len_pdf / 15)

    # checks if the length of PDF is not one, as len == 1 is a special case
    if len_pdf != 1:
        # creates the chunk list
        chunks = list(range(1, len_pdf, 15))
        # adds the last page of the pdf in the case it is not a multiple of 15 so all pages are processed in every case
        if chunks[-1] != len_pdf:
            chunks.append(len_pdf)
        return chunks
    else:
        # returns a list for single page pdfs or images such that no erorrs will be thrown
        return [1, 1]
    

# main loop  to go through all files in the specified folder
for path in files:
    # store filename for later
    name = path

    # start timing
    total_time_start = time.clock()

    # create full path directory to file, as main path input is a folder
    path = in_path + '\\' + path

    # identify if pdf or image and convert accordingly to create a pdf bytes object as later functions were originally designed for PDF
    if path[-3:] == 'pdf':
        pdf = open(path, 'rb')
    elif path[-3:] != 'pdf':
        import img2pdf

        pdf = img2pdf.convert(path)
        
        file = open("temp.pdf", "wb")
        file.write(pdf)
        path = "temp.pdf"

    # create list of chunk beginning points using function
    chunks_list = chunk(path)

    # initialize master list for manual verification, can be exported as CSV or TXT later
    master_hori = pd.Series()

    # sub loop to go through all chunks of the file if longer than 15 pages
    for i in range(num_chunks):
        pages_df = create_images(path, chunks_list[i], chunks_list[i+1])

        np_images = pages_df['PIL_images'].map(convert_to_np)

        # # check image conversion if desired
        # pages_df['np_images'].map(check_np_images)

        # convert to greyscale and threshold
        bw_images = np_images.map(grey_and_thresh)

        # find angles of skew for images
        angles = bw_images.map(angle_of_skew)

        # rotate images
        df = pd.DataFrame({'bw_images': bw_images, 'angles': angles})
        rotated_images = df.apply(lambda x: rotate_image(x['bw_images'], x['angles']), axis=1)

        # determine if images are horizontal or vertical
        is_horizontal = rotated_images.map(hori_or_not)

        # append to master list for manual verification
        master_hori = master_hori.append(is_horizontal, ignore_index=True)

        # rotate images which are not correctly oriented
        for j in range(len(rotated_images)):
            # True values represent a correct orientation, so rotation is executed on false values
            if not(is_horizontal[j]):
                rotated_images[j] = rotate_image(rotated_images[j], 270) # 270 degres is most common, could also be 90

        # create the path for each output file
        indiv_out_name = out_path + '\\' + name.split('.')[0] + ".pdf"

        # create output list to store each page in the pdf
        pdf_output = []
        for k in range(len(rotated_images)):
            im = Image.fromarray(rotated_images[k])
            pdf_output.append(im)

        # saves the output as a pdf, creates pdf with the first image(page) then adds the rest with append_images
        pdf_output[0].save(indiv_out_name, "PDF", resolution=100.0, save_all=True, append_images=pdf_output[1:])

    # export list of whether pages are oriented correctly to a CSV in the output path
    master_hori.to_csv(out_path)

    total_time_end = time.clock()
    print('File runtime:', total_time_end - total_time_start)

