import numpy as np
import argparse
import cv2
import PyPDF2
from PyPDF2 import PdfFileReader
import pdf2image as pdf2
from PIL import Image
import pandas as pd
from hough_space_func import hori_or_not
import time
import multiprocessing
import math
from multiprocessing import Pool
from image_rotation import angle_of_skew, rotate_image

# # construct argument parse object and parse arguments
# ap = argparse.ArgumentParser()
# ap.add_argument('-p', '--pdf', required=True, help='path to input pdf file')
#
# # get all variables from the parsed arguments
# args = vars(ap.parse_args())
total_time_start = time.clock()


def chunk(path):
    """
    :param path: path to the pdf document for COR/processing
    :return: a list of the page numbers at the beginning and end of each chunk
    global variable for number of chunks and length of pdf
    """
    global len_pdf
    len_pdf = PdfFileReader(open(path, "rb")).getNumPages()
    global num_chunks
    num_chunks = math.ceil(len_pdf/15)
    chunks = list(range(0, len_pdf, 15))
    if chunks[-1] != len_pdf:
        chunks.append(len_pdf)
    return chunks


def create_images(path, start, stop):
    """
    :param path: location of document to convert to images
    :param start: index of starting page using zero based indexing
    :param stop: every page up to but not including this page will be converted
    :return: df of converted images
    """
    pages = pdf2.convert_from_path(path, 500, thread_count=4, first_page=start, last_page=stop)
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
    # Convert RGB to BGR
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
    invert image for white text on black background (optional, can test with and and without)
    gray_image = cv2.bitwise_not(gray_image)
    '''
    # threshold image into black and white
    thresholded = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return thresholded


def parallelize(df):
    """
    :param df: full data frame of input images to be chunked for processing
    :return: a list
    """
    num_chunks = len(pages_df) / (len(pages_df)/20)
    num_cores = multiprocessing.cpu_count()

    split_df = np.array


chunks_list = chunk(r'C:\Users\CALEB_NEALE\Downloads\Eisaman_5H_Completion_Report.pdf')

columns = ['np_images', 'bw_images', 'angles', 'rotated_images', 'is_horizontal']
df = pd.DataFrame(columns=columns)

for i in range(num_chunks):
    pages_df = create_images(r'C:\Users\CALEB_NEALE\Downloads\Eisaman_5H_Completion_Report.pdf',
                             chunks_list[i], chunks_list[(i+1)])

    np_images = pages_df['PIL_images'].map(convert_to_np)

    # check image conversion if desired
    # pages_df['np_images'].map(check_np_images)

    # convert to greyscale and threshold
    bw_images = np_images.map(grey_and_thresh)

    # these functions are not necessary for pdfs which were searchable/properly formatted before upload
    # find angles of skew for images
    angles = bw_images.map(angle_of_skew)

    # rotate images
    df = pd.DataFrame({'bw_images': bw_images, 'angles': angles})
    rotated_images = df.apply(lambda x: rotate_image(x['bw_images'], x['angles']), axis=1)
    print(rotated_images)
    # determine if images are horizontal or vertical
    is_horizontal = bw_images.map(hori_or_not)  # change df column to rotated images if using above

    print(is_horizontal)

total_time_end = time.clock()
print(total_time_end - total_time_start)

# create lsit of all black coordinates
# black_coords = np.column_stack(np.where(thresh > 0))
#
# # load pdf in read mode for images
# pdf_in = open(args['pdf'], 'rb')
#
# # create pypdf2 pdf object
# pdf_reader = PyPDF2.PdfFilereader(pdf_in)
#
# # create pdf_writer object for creation of final pdf later
# pdf_writer = PyPDF2.PdfFilewriter()