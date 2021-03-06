import cv2
import numpy as np


def hori_or_not(img, show=True):
    """
    :param img: b&w np image to classify as vertical or horizontal (readable)
    :param show: whether to create output files showing lines drawn on the image
    :return: bool value whether image is readable, true for readable/horizontal, false for not
    """
    edges = cv2.Canny(img, 50, 150, apertureSize=7)
    minLineLength = 500
    maxLineGap = 20
    lines = cv2.HoughLinesP(edges, 1, np.pi/180,10, minLineLength, maxLineGap)

    # executed only if lines are to be displayed, show=True
    if show:
        # loops through each set of coordinates generated by HoughLinesP and creates a line for each point.
        # Only necessary for visualization
        i=0
        while i < len(lines):
            for x1,y1,x2,y2 in lines[i]:
                cv2.line(img,(x1,y1),(x2,y2),(0,255,0),50)
            i+=1

        # write output
        scale_percent = 25  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        resized_edges = cv2.resize(edges, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow('houghlines.jpg',resized)
        cv2.waitKey(0)
        cv2.imshow('houghlines.jpg',  resized_edges)
        cv2.waitKey(0)
        print("running!")

    # checks if any lines were drawn, returns np.nan if not
    if type(lines) == np.ndarray:
        # calculates initial and final x and y coordinates
        x1 = lines[:,0,0]
        y1 = lines[:,0,1]
        x2 = lines[:,0,2]
        y2 = lines[:,0,3]

        # calculates the change in x and change in y in order to later calculate average slope
        delta_x = x2 - x1
        delta_y = y2 - y1

        # calculates average slope to determine if vertical or horizontal
        avg_slope = np.mean(delta_y)/np.mean(delta_x)

        # uses a threshold to determine if vertical or horizontal
        # threshold (default 2.3) can be adjust for greater or less tolerance
        if -2.3 < avg_slope < 2.3:
            return True
        else:
            return False
    else:
        return np.nan
