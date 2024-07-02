import os
import yaml
from easydict import EasyDict as edict
import numpy as np
import torch
from loguru import logger

class YamlParser(edict):
    """
    This is the YAML parser to open the file and facilitate easy reading and writing within it
    """

    def __init__(self, cfg_dict=None, cfg_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if cfg_file is not None:
            assert(os.path.isfile(cfg_file))
            with open(cfg_file, 'r') as fo:
                yaml_ = yaml.load(fo.read(), Loader=yaml.FullLoader)
                cfg_dict.update(yaml_)

        super(YamlParser, self).__init__(cfg_dict)

    def merge_from_file(self, cfg_file):
        with open(cfg_file, 'r') as fo:
            yaml_ = yaml.load(fo.read(), Loader=yaml.FullLoader)
            self.update(yaml_)

    def merge_from_dict(self, config_dict):
        self.update(config_dict)

class PerClassDecorator:
    def __init__(self, method):
        # Store the method that will be decorated
        self.update = method
        self.nr_classes = 80
        self.per_class_active_tracks = {}
        for i in range(self.nr_classes):
            self.per_class_active_tracks[i] = []

    def __get__(self, instance, owner):
        # This makes PerClassDecorator a non-data descriptor that binds the method to the instance
        def wrapper(*args, **kwargs):
            # Unpack arguments for clarity
            args = list(args)
            dets = args[0]
            im = args[1]
            
            if instance.per_class is True:

                # Initialize an array to store the tracks for each class
                per_class_tracks = []
                
                # same frame count for all classes
                frame_count = instance.frame_count

                for i, cls_id in enumerate(range(self.nr_classes)):
 
                    if dets.size > 0:
                        class_dets = dets[dets[:, 5] == cls_id]
                    else:
                        class_dets = np.empty((0, 6))
                    logger.debug(f"Processing class {int(cls_id)}: {class_dets.shape}")

                    # activate the specific active tracks for this class id
                    instance.active_tracks = self.per_class_active_tracks[cls_id]
                    
                    # reset frame count for every class
                    instance.frame_count = frame_count
                    
                    # Update detections using the decorated method
                    tracks = self.update(instance, class_dets, im)

                    # save the updated active tracks
                    self.per_class_active_tracks[cls_id] = instance.active_tracks

                    if tracks.size > 0:
                        per_class_tracks.append(tracks)
                                        
                # when all active tracks lists have been updated
                instance.per_class_active_tracks = self.per_class_active_tracks
                
                # increase frame count by 1
                instance.frame_count = frame_count + 1

                tracks = np.vstack(per_class_tracks) if per_class_tracks else np.empty((0, 8))
            else:
                # Process all detections at once if per_class is False or detections are empty
                tracks = self.update(instance, dets, im)
            
            return tracks

        return wrapper


def iou_batch(bboxes1, bboxes2) -> np.ndarray:
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1]) +
        (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) -
        wh
    )
    return o

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x_c, y_c, width, height) format to
    (x1, y1, x2, y2) format where (x1, y1) is the top-left corner and (x2, y2)
    is the bottom-right corner.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def tlwh2xyah(x):
    """
    Convert bounding box coordinates from (t, l ,w ,h)
    to (center x, center y, aspect ratio, height)`, where the aspect ratio is `width / height`.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] + (x[..., 2] / 2)
    y[..., 1] = x[..., 1] + (x[..., 3] / 2)
    y[..., 2] = x[..., 2] / x[..., 3]
    y[..., 3] = x[..., 3]
    return y

def xywh2tlwh(x):
    """
    Convert bounding box coordinates from (x c, y c, w, h) format to (t, l, w, h) format where (t, l) is the
    top-left corner and (w, h) is width and height.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2.0  # xc --> t
    y[..., 1] = x[..., 1] - x[..., 3] / 2.0  # yc --> l
    y[..., 2] = x[..., 2]                    # width
    y[..., 3] = x[..., 3]                    # height
    return y

def xyxy2xywh(x):
    print(x)
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
    Returns:
       y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y