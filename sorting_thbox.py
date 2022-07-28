# import the necessary packages
from decimal import DivisionUndefined
import numpy as np
import argparse
import imutils
import cv2

LINES = []
LABELS = []
### fdf
def sort_contours(cnts, method="left-to-right"):
	global LABELS
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes, LABELS) = zip(*sorted(zip(cnts, boundingBoxes, LABELS),
		key=lambda b:b[1][i], reverse=reverse))
	# print(boundingBoxes)
	# print(cnts)
	global LINES
	LINES = divide_line(boundingBoxes, i)

	# print(tuple(zip(cnts, boundingBoxes)))
	print (sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][i], reverse=reverse))
	# print(list(zip(cnts, boundingBoxes)))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

def divide_line(boundingBoxes, i_ex):
    rows = [0 for i in range(len(boundingBoxes))]
    row_max = 0
    for (i, c) in enumerate(boundingBoxes):
        if i == 0: continue
        if not is_sameline(boundingBoxes[i][i_ex], boundingBoxes[i-1][i_ex]):
            row_max += 1
        rows[i] = row_max
    print ('row list:',rows)
    return rows

    
def is_sameline(value1, value2, threshold = 50):
    return abs(value1 - value2) < threshold    

def is_contain_objects(num_o, check_label, check_line, lines_list = LINES, labels_list = LABELS):
    label_obj_inline = []
    lines_list = LINES
    for (i,c) in enumerate(labels_list):
        if lines_list[i] == check_line:
            label_obj_inline += [c]
    
    res = any([check_label]*num_o == label_obj_inline[i:i+num_o] for i in range(len(label_obj_inline) - 1))
    return res


def draw_contour(image, c, i):
	# compute the center of the contour area and draw a circle
	# representing the center
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
 
	# draw the countour number on the image
	cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (255, 255, 255), 2)
 
	# return the image with the contour number drawn on it
	return image


def _box_to_contours_ori(_img_path, _img_label):
    # fl = open(_img_label, 'r')
    # coords = fl.readlines()
    # fl.close()
    
    # img = cv2.imread(_img_path) #args["image"])
    # img_h, img_w = img.shape[0:2]
    # res = ()
    
    # for dt in coords:            
    #     # class x_center y_center width height
    #     dt_split = dt.split(' ')
    #     dt_split_int = list(map(lambda x: float(x), dt_split))
    #     _, tmp_x_cen, tmp_y_cen, tmp_box_w, tmp_box_h = dt_split_int
    #     # get ABCD (numpy array)
    #     tmp_ABCD = get_cornercoords(tmp_x_cen, tmp_y_cen, tmp_box_w, tmp_box_h, img_w, img_h)
    #     # print(tmp_xy1xy2)
    #         # draw.rectangle(tmp_xy1xy2, outline=(0, 0, 0, 255), width=10)
    #     res = res + (tmp_ABCD, )
    
    # # print(res)

    # (cnts, boundingBoxes) = sort_contours(res,"top-to-bottom")   #(method=args["method"])
    # (cnts, boundingBoxes) = sort_contours(res,"left-to-right")   #(method=args["method"])

    # # loop over the (now sorted) contours and draw them
    # for (i, c) in enumerate(cnts):
    #     draw_contour(img, c, i)

    # # show the output image
    # imS = cv2.resize(img, (500, 800))     
    # cv2.imshow("Sorted", imS)

    # # cv2.imshow('bounding', cv2.resize(cv2.drawContours(img, res, -1, (0,255,0), 3), (600, 900))  )

    # cv2.waitKey(0)
    # return res
    pass


def box_to_contours(coords, img_w, img_h):
    global LABELS
    res = ()
    for dt in coords:            
        # class x_center y_center width height
        dt_split = dt.split(' ')
        dt_split_int = list(map(lambda x: float(x), dt_split))
        _, tmp_x_cen, tmp_y_cen, tmp_box_w, tmp_box_h = dt_split_int
        LABELS += [int(_)]
        # get ABCD (numpy array)
        tmp_ABCD = get_cornercoords(tmp_x_cen, tmp_y_cen, tmp_box_w, tmp_box_h, img_w, img_h)
        # print(tmp_xy1xy2)
            # draw.rectangle(tmp_xy1xy2, outline=(0, 0, 0, 255), width=10)
        res = res + (tmp_ABCD, )
    print(LABELS)
    # print(res)
    return res

   
def get_recfromcorners(corners_np):
    """generate x1, x2, y1, y2 from 4 corners (contours opencv format)

    Args:
        corners_np:
            np_array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                ...
            )

    Returns:
        [
            [x1, x2, y1, y2],
            ...
        ]
    """
    # print(corners_np)
    x1 = corners_np[0][0][0]
    y1 = corners_np[0][0][1]
    x2 = corners_np[2][0][0]
    y2 = corners_np[2][0][1]
    return [x1, x2, y1, y2]

    
def get_cornercoords(x_center, y_center, box_width, box_height, image_width, image_height):
    """get corners list (contours opencv format)

    Args:
        x_center (float): abscissa (x axis) of box center
        y_center (float): ordinate (y axis)of box center
        box_width (float):
        box_height (float):
        image_width (float):
        image_height (float):

    Returns:
        np_array([[x1, y1]], [[x2,y1]], [[x2,y2]], [[x1,y2]]): array of corners coordinates in numpy format
    """
    # class x_center y_center width height
    x1 = image_width * (x_center - box_width / 2)
    x2 = image_width * (x_center + box_width / 2)
    
    y1 = image_height * (y_center - box_height / 2)
    y2 = image_height * (y_center + box_height / 2)
    
    corners = [[[x1, y1]], [[x2,y1]], [[x2,y2]], [[x1,y2]]]
    corners_np = np.array(corners, dtype=np.dtype(np.int32))
    
    return corners_np


def get_center_rec(x12y12):
    """get center(x, y) from x1, x2, y1, y2 coordinates

    Args:
        x12y12: [x1, x2, y1, y2]

    Returns:
        [x_aver, y_aver]
    """
    x1, x2, y1, y2 = x12y12
    return [(x1 + x2) / 2, (y1 + y2) / 2]


def draw_sortedbox(_img_path, _img_label):
    global LABELS
    global LINES
    fl = open(_img_label, 'r')
    coords = fl.readlines()
    fl.close()
    
    img = cv2.imread(_img_path) #args["image"])
    img_h, img_w = img.shape[0:2]

    contours = box_to_contours(coords, img_w, img_h)
    print(contours[0].shape)

    (cnts, boundingBoxes) = sort_contours(contours,"top-to-bottom")   #(method=args["method"])
    # (cnts, boundingBoxes) = sort_contours(res,"left-to-right")   #(method=args["method"])
    # print(boundingBoxes)

    for (i, c) in enumerate(cnts):
        # print(i)
        # print(ROWS[i])
        draw_contour(img, c, LINES[i])
    
    centers = []
    for (i, c) in enumerate(cnts):
        centers += [get_center_rec(get_recfromcorners(c))]
    

    # show the output image
    img = cv2.drawContours(img, cnts, -1, (0,255,0), 3)
    
    imS = cv2.resize(img, (500, 800))     
    cv2.imshow("Sorted", imS)

    # cv2.imshow('bounding', cv2.resize(cv2.drawContours(img, res, -1, (0,255,0), 3), (600, 900))  )
    # 
    print(is_contain_objects(2,0,1))
    print(LINES)
    print(LABELS)
    cv2.waitKey(0)


def draw_box(frame, bbox, color=(255,0,0)):
	x1, y1, x2, y2 = bbox
	cv2.rectangle(frame, pt1=(x1, y1), pt2=(x2, y2), color=color, thickness=2)
	return frame

def IOU(box1, box2):
	""" We assume that the box follows the format:
		box1 = [x1,y1,x2,y2], and box2 = [x3,y3,x4,y4],
		where (x1,y1) and (x3,y3) represent the top left coordinate,
		and (x2,y2) and (x4,y4) represent the bottom right coordinate """
	x1, y1, x2, y2 = box1	
	x3, y3, x4, y4 = box2
	x_inter1 = max(x1, x3)
	y_inter1 = max(y1, y3)
	x_inter2 = min(x2, x4)
	y_inter2 = min(y2, y4)
	width_inter = abs(x_inter2 - x_inter1)
	height_inter = abs(y_inter2 - y_inter1)
	area_inter = width_inter * height_inter
	width_box1 = abs(x2 - x1)
	height_box1 = abs(y2 - y1)
	width_box2 = abs(x4 - x3)
	height_box2 = abs(y4 - y3)
	area_box1 = width_box1 * height_box1
	area_box2 = width_box2 * height_box2
	area_union = area_box1 + area_box2 - area_inter
	iou = area_inter / area_union
	return iou


image_path = 'datasets/data_th/images/val/THM_297.jpg'
label_path = 'datasets/data_th/labels/val/THM_297.txt'

clist = draw_sortedbox(image_path, label_path)
# print((t[0]).shape)
# print(clist)


