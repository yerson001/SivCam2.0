# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#   File name   : SSD_Detector.py
#   Author      : Ruben Cardenes
#   Created date: 11.Dec.2019
#   Description : class to encapsulate Object detection with SSD model
# ================================================================

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

def draw_pretty_bbox(image, bboxes, classes=None,
                     show_label=True,
                     colors=[(255, 0, 0), (255, 255, 0)],
                     colors_secondary=[(0, 102, 255), (227, 226, 184), (102, 153, 255), (102, 255, 153), (153, 0, 153), (50, 50, 50)],
                     filter_classes=[True, True, True, True, True, True],
                     thickness=2, uncertainty_th=0.4, index_to_hide=-1, show_id=True):
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
        if len(bbox) > 6:
            object_id = bbox[6]
        else:
            object_id = 0
        class_ind = int(bbox[5])
        if class_ind >= len(classes):
            continue
        #if not filter_classes[class_ind]:
        #    continue
        if score < uncertainty_th:
            # continue
            bbox_color = colors[-1]
            class_text = 'uncertain'
        #if score < 0.6:
        #    bbox_color = colors_secondary[class_ind]
        #    class_text = classes[class_ind]
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
            # Uncomment this line to also show the object id
            if show_id:
                bbox_mess = '%s: %.2f, %d' % (class_text, score, object_id)
            else:
                bbox_mess = '%s: %.2f' % (class_text, score)
            # bbox_mess = '%s: %.2f' % (class_text, score)
            # t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.putText(image, bbox_mess, (c1[0], c1[1]-9), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale,  bbox_color, bbox_thick//2, lineType=cv2.LINE_AA)

    return image


def bboxes_iou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms_disregard_classes(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    best_bboxes = []

    while len(bboxes) > 0:
            max_ind = np.argmax(bboxes[:, 4])
            best_bbox = bboxes[max_ind]
            best_bboxes.append(best_bbox)
            bboxes = np.concatenate([bboxes[: max_ind], bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            bboxes[:, 4] = bboxes[:, 4] * weight
            score_mask = bboxes[:, 4] > 0.
            bboxes = bboxes[score_mask]

    return best_bboxes


class SSDDetector:
    def __init__(self, cfg):
        self.input_size = cfg["INPUT_SIZE"]
        self.classes = read_class_names(cfg["CLASSES"])
        self.num_classes = len(self.classes)
        self.score_threshold = cfg["SCORE_THRESHOLD"]
        self.iou_threshold = cfg["IOU_THRESHOLD"]
        self.detection_graph = self.load_graph(cfg["MODEL_PB_FILE"])
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.prepare_placeholders()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph, config=config)

    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def to_image_coords(self, boxes, height, width, scores, classes):
        """
        The original box coordinate output is normalized, i.e [0, 1].

        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros((len(boxes),6))
        box_coords[:, 0] = boxes[:, 1] * width
        box_coords[:, 1] = boxes[:, 0] * height
        box_coords[:, 2] = boxes[:, 3] * width
        box_coords[:, 3] = boxes[:, 2] * height
        # box_coords[:, 0] = boxes[:, 0] * height
        # box_coords[:, 1] = boxes[:, 1] * width
        # box_coords[:, 2] = boxes[:, 2] * height
        # box_coords[:, 3] = boxes[:, 3] * width
        box_coords[:, 4] = scores
        box_coords[:, 5] = classes
        return box_coords

    def prepare_placeholders(self):
        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

    def set_score_threshold(self, th):
        self.score_threshold = th

    def predict_image(self, image_np):
        # image_np is a numpy array of dimensions H, W, Channels
        # Next line will make the image array as (1, H, W, Channels)
        # We have to swap the B and R channels as this model works best for RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_np = np.expand_dims(image_np, 0)
        (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores,
                                             self.detection_classes],
                                            feed_dict={self.image_tensor: image_np})

        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        # Filter boxes with a confidence score less than `confidence_cutoff`
        boxes, scores, classes = self.filter_boxes(self.score_threshold, boxes, scores, classes)
        # print("num boxes detected ", len(boxes))
        # The current box coordinates are normalized to a range between 0 and 1.
        # This converts the coordinates actual location on the image.
        height, width = image_np.shape[1], image_np.shape[2]
        boxes = self.to_image_coords(boxes, height, width, scores, classes)

        boxes = nms_disregard_classes(boxes, 0.2, method='nms')

        return list(boxes)

    def session_close(self):
        self.sess.close()


if __name__ == "__main__":
    classes = read_class_names("./data/coco.names")
    cfg = {"INPUT_SIZE": 600,
           "CLASSES": "./data/coco.names",
           "SCORE_THRESHOLD": 0.3,
           "IOU_THRESHOLD": 0.1,
           "MODEL_PB_FILE": "./data/ssd_mobilenet_v2_coco_graph.pb"
         }
    SSD = SSDDetector(cfg)
    image_np = cv2.imread('./data/road.jpg')
    boxes = SSD.predict_image(image_np)
    print(boxes)
    for i, b in enumerate(boxes):
        if b[5] > 4:
            boxes[i][5] = 0

    class_color_filename = './data/colors.yaml'
    colors, colors_secondary = read_class_colors(class_color_filename)
    colors1 = list(map(lambda x: (int(x[0] * 1), int(x[1] * 1), int(x[2] * 1)), colors))
    image_np = draw_pretty_bbox(image_np, boxes, show_label=True,
                                      colors=colors1, classes=classes)
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow("result", image_np)
    cv2.waitKey(0)
    SSD.session_close()