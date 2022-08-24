from __future__ import print_function

import mysql.connector
from mysql.connector import errorcode
from datetime import date, datetime, timedelta
# import datetime

from pathlib import Path
import torch
import json

# from yolov5.utils.general import (LOGGER, check_requirements, check_suffix, check_version, colorstr, increment_path,
#                            make_divisible, non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh)

# from yolov5.utils.plots import Annotator, colors, save_one_box
from PIL import Image
import numpy as np

DB_NAME = 'yolov5'

cnx = mysql.connector.connect(
  host="localhost",
  user="root",
  password="luan",
  database='yolov5'
)

cursor = cnx.cursor()

# add_output_th = ("INSERT INTO output_th "
#                "(folder_path, image_name, predict_json, predict_date) "
#                "VALUES (%(image_folder_path)s, %(image_name)s, %(predict_folder_path)s, %(predict_json)s, %(predict_date)s)")

image_folder_path = 'datasets/data_th/images/val/'
image_name = 'THM_254.jpg'

predict_folder = 'predict/'
path_weights = 'yolov5/runs/train/exp9/weights/best.pt'

def update_db(_img_name, predict_json_str, _out_folder, results, _folder = '../datasets/data_th/images/val/'):
    cnx = mysql.connector.connect(
        host="localhost",
        user="root",
        password="luan",
        database='yolov5'
    )

    cursor = cnx.cursor()
    
    now = datetime.now()
    formatted_date = now.strftime('%y%m%d_%Hh%Mm%Ss')
    formatted_date_db = now.strftime('%Y-%m-%d %H:%M:%S')
    predict_json_file = ''.join(map(str,image_name.split('.')[:-1])) + '-' + formatted_date + '.json'
    
    
    predict_image = 'out_' + ''.join(map(str,image_name.split('.')[:-1])) + '-' + formatted_date + '.jpg'
    ################################################################################################
    
    # pprint=False
    # show=False
    # save=True
    # render=False
    # labels=True
    # # save_dir= Path('')
    # save_dir= Path(_out_folder)
    # # print('dir: ',save_dir)

    # results.files = [predict_image]

    # crops = []
    # for i, (im, pred) in enumerate(zip(results.imgs, results.pred)):
    #     print(i)
        
    #     s = f'image {i + 1}/{len(results.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
    #     if pred.shape[0]:
    #         for c in pred[:, -1].unique():
    #             n = (pred[:, -1] == c).sum()  # detections per class
    #             s += f"{n} {results.names[int(c)]}{'s' * (n > 1)}, "  # add to string
    #         if show or save or render:
    #             annotator = Annotator(im, example=str(results.names))
    #             for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
    #                 label = f'{results.names[int(cls)]} {conf:.2f}'
    #                 annotator.box_label(box, label if labels else '', color=colors(cls))
    #             im = annotator.im
    #     else:
    #         s += '(no detections)'

    #     im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
    #     if pprint:
    #         print(s.rstrip(', '))
    #     if show:
    #         im.show(results.files[i])  # show
    #     if save:
    #         f = results.files[i]
    #         im.save(save_dir / f)  # save
    #         if i == results.n - 1:
    #             print(f"Saved {results.n} image{'s' * (results.n > 1)} to {colorstr('bold', save_dir)}")
    #     if render:
    #         results.imgs[i] = np.asarray(im)
            
    ################################################################################################
    
    

    _cursor = cursor
    _cnx = cnx
    #   Insert to database
    #       Check is contain image
    query_name_img = "select * from Image where name = %s and folder_path = %s"
    _cursor.execute(query_name_img, (_img_name,_folder,))
    record = _cursor.fetchall()
    
    if record == []:
        cmd_insert_img = ("INSERT INTO yolov5.Image(name, folder_path) "
        # "VALUES({}, {})").format('12', '123123')
        "VALUES(\'{}\', \'{}\')").format(str(_img_name), str(_folder))
        
        # cmd_insert_img = "INSERT INTO yolov5.Image(name, folder_path) VALUES('15555555552', '132323')"
        # print('line 29: ', cmd_insert_img)
        
        _cursor.execute(cmd_insert_img)
        _cnx.commit()
    
    cmd_insert_predict = ("INSERT INTO Predict_result "
               "(image_name, image_folder, predict_folder, predict_image, predict_json_file, predict_json_str, time)"
               "VALUES (%s, %s, %s, %s, %s, %s, %s)")
    tmp_data = (_img_name, _folder, _out_folder, predict_image, predict_json_file, predict_json_str, formatted_date_db)
    
    print(datetime.now())
    _cursor.execute(cmd_insert_predict, tmp_data)
    _cnx.commit()