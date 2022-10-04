# ================================================================
#   Copyright (C) 2020 * Ltd. All rights reserved.
#   File name   : ObjectDetectirTFLITE.py
#   Author      : Ruben Cardenes
#   Created date: 20.Mar.2019
#   Description : Object Detector wrapper for TFLITE models
# ================================================================

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import numpy as np
import cv2
import yaml
import random


def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def read_class_colors(class_colors_file_name):
    with open(class_colors_file_name, 'r') as fc:
        try:
            colors_dict = yaml.safe_load(fc)
        except yaml.YAMLError as exc:
            print(exc)
    colors = [None] * len(colors_dict['main_colors'])
    colors_secondary = [None] * len(colors_dict['secondary_colors'])
    for class_name, v in colors_dict['main_colors'].items():
        colors[colors_dict['main_colors'][class_name]['id']] = tuple(map(int, colors_dict['main_colors'][class_name]['color'].split(",")))
    for class_name, v in colors_dict['secondary_colors'].items():
        colors_secondary[colors_dict['secondary_colors'][class_name]['id']] = tuple(map(int, colors_dict['secondary_colors'][class_name]['color'].split(",")))

    return colors, colors_secondary


def scale_boxes(boxes, w, h):
    out_boxes = []
    for b in boxes:
        b_scale = [b[1]*w, b[0]*h, b[3]*w, b[2]*h]
        b_scale.extend(b[4:])
        out_boxes.append(b_scale)
    return out_boxes


def draw_bbox(image, bboxes, classes=None,
                     show_label=True,
                     colors=[(255, 0, 0), (255, 255, 0)],
                     thickness=2, uncertainty_th=0.4, index_to_hide=-1):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id, object_id] format coordinates.q
    """
    if type(image) is np.ndarray:
        image_h, image_w, _ = image.shape

    bbox_thick = thickness
    if len(colors) < len(classes):
        for i in range(len(classes)-len(colors)):
            colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    for i, bbox in enumerate(bboxes):
        if index_to_hide == i:
            continue
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        if class_ind >= len(classes):
            continue
        if score < uncertainty_th:
            # continue
            bbox_color = colors[-1]
            class_text = 'uncertain'
        else:
            bbox_color = colors[class_ind]
            class_text = classes[class_ind]

        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        # cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
        # cv2.line(image, (x1, y1), (x2, y2), bbox_color, bbox_thick)
        w = coor[2] - coor[0]
        h = coor[3] - coor[1]
        s = min(w,h)
        #  uper line
        cv2.line(image, (coor[0], coor[1]), (coor[0]+s//3, coor[1]), bbox_color, bbox_thick)
        cv2.line(image, (coor[0] + w - s//3, coor[1]), (coor[0] + w, coor[1]), bbox_color, bbox_thick)
        #  left line
        cv2.line(image, (coor[0], coor[1]), (coor[0], coor[1]+s//3), bbox_color, bbox_thick)
        cv2.line(image, (coor[0], coor[1] + h - s//3), (coor[0], coor[1]+h), bbox_color, bbox_thick)
        # lower line
        cv2.line(image, (coor[2], coor[3]), (coor[2] - s//3, coor[3]), bbox_color, bbox_thick)
        cv2.line(image, (coor[2] - w + s//3, coor[3]), (coor[2] - w, coor[3]), bbox_color, bbox_thick)
        # right line
        cv2.line(image, (coor[2], coor[3]), (coor[2], coor[3]-s//3), bbox_color, bbox_thick)
        cv2.line(image, (coor[2], coor[3] - h + s//3), (coor[2], coor[3]-h), bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (class_text, score)
            # t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.putText(image, bbox_mess, (c1[0], c1[1]-9), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale,  bbox_color, bbox_thick//2, lineType=cv2.LINE_AA)

    return image

class ObjectDetectorTFLITE(object):
    """Object Detector with TFLITE model"""

    def __init__(self, cfg):
        self._interpreter = tf.lite.Interpreter(model_path=cfg["MODEL_PB_FILE"])
        self._interpreter.allocate_tensors()
        self.input_details = self._interpreter.get_input_details()
        self.output_details = self._interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        self.classes = read_class_names(cfg["CLASSES"])
        self.num_classes = len(self.classes)
        self.score_th = cfg["SCORE_THRESHOLD"]
        self.iou_threshold = cfg["IOU_THRESHOLD"]

    def __str__(self):
        return 'Object Detector with TFLITE model:\n\
                    input: {}\n\
                    output: {}'.format(self.input_details, self.output_details)

    def predict_image(self, image):
        """Detecting object for an gray image
        Args:
            image: numpy array with shape (image_height, image_width)
        Returns:
            boxes
        Raises:
            ValueError: if dimension of the image is not 2
        """
        # if len(image.shape) != 2:
        #     raise ValueError('dimension of the image must 2')

        org_height, org_width, _ = image.shape
        #print("Original shape ", (org_height, org_width))
        #print("Reshaping to ",(self.input_shape[2], self.input_shape[1]))
        input_data = cv2.resize(image, (self.input_shape[2], self.input_shape[1]))
        input_data = np.reshape(input_data, (1, self.input_shape[2], self.input_shape[1], 3))
        input_data = input_data.astype(np.float32)
        input_data = input_data/255.0
        # fill data and inference
        self._interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self._interpreter.invoke()

        # get output data
        out_boxes = self._interpreter.get_tensor(self.output_details[0]['index'])
        out_classes = self._interpreter.get_tensor(self.output_details[1]['index'])
        out_scores = self._interpreter.get_tensor(self.output_details[2]['index'])


        # Post-processing
        #out_boxes = utils.postprocess_boxes_2(out_boxes, (org_height, org_width),
        #                                      self.input_shape[2], self.score_th)
        #out_boxes = utils.nms(out_boxes, self.iou_threshold, method='nms')
        return out_boxes, out_scores, out_classes


