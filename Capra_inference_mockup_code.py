import torch
import numpy as np
import cv2
from time import perf_counter
from ultralytics import RTDETR
from pathlib import Path
import supervision as sv
from bytetrack.byte_tracker import BYTETracker
#from strongsort.strongsort import StrongSORT
from bytetrack.byte_utils import YamlParser
import matplotlib.pyplot as plt
import time
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from capra_ros_vision_msgs.msg import BoundingBox, BoundingBoxes

SAVE_VIDEO = True
TRACKER = "bytetrack"

class ObjectDetector(Node):

    def __init__(self):
        super().__init__('RTDETRPublisher')
        self.bridge = CvBridge()
        prev_frames = None
        curr_frames= None
        self.prev_frames = prev_frames
        self.curr_frames = curr_frames

        # initialize piublisher and subscriber
        self.camera_subscriber_ = self.create_subscription(Image, "oak/rgb/image_raw", self.camera_sub, 10)
        self.detection_image_publisher = self.create_publisher(BoundingBoxes,'/bounding_boxes',10)
        self.model = None  # eliminated "used before assignment" warning
        if TRACKER == "bytetrack":
            tracker_cfg = "configs/tracking/bytetrack.yaml"
            cfg = YamlParser()
            cfg.merge_from_file(tracker_cfg)
            print(cfg)

            self.tracker = BYTETracker(
                track_thresh = cfg.track_thresh,
                match_thresh = cfg.match_thresh,
                track_buffer = cfg.track_buffer,
                frame_rate = cfg.frame_rate
             )


    def load_model(self):
        # TODO: load your .pt model here and assign it to self.model
        model = RTDETR('weights/best.pt')
        model.fuse()
        self.model = model
        

    def camera_sub(self, msg):        
        ros_image = msg
        outputs = [None]
        tracker = self.tracker


        if hasattr(tracker, 'model'):
            if hasattr(tracker.model, 'warmup'):
                tracker.model.warmup()
        # Try to do inference and catch `Exception` errors using except
        try:
            # convert ROS image to CV image
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")

            #Start the timer
            start_time = time.time()
            # make prediction
            result = self.predict(cv_image)
            detection_image, _ = self.draw_results(cv_image, result)

            if hasattr(tracker, 'tracker') and hasattr(tracker.tracker, 'camera_update'):
                if prev_frames is not None and curr_frames is not None: #Camera Motion Compensation
                    tracker.tracker.camera_update(prev_frames, curr_frames)

                    new_boxes = []
                    for res in result:
                        box = res.boxes.data.cpu().numpy()
                        outputs[0] = tracker.update(box, detection_image)
                        for i, output in enumerate(outputs[0]):
                            bbox = output[0:4]
                            new_boxes.append(bbox)
                            tracked_id = output[4]
                            print(tracked_id)
                            top_left = (int(bbox[-2] - 100), int(bbox[1]))

                            cv2.putText(detection_image, f"ID : {tracked_id}", top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
                    #box_list = BoundingBoxes()
                    #box_list.bounding_boxes = new_boxes
                    #self.minimal_bbox_pub.publish(box_list)

            # republish frame with bounding box
            detection_msg = self.bridge.cv2_to_imgmsg(detection_image, 'bgr8')
            detection_msg.header = msg.header # ensure msg and prediction has same time stamp
            self.detection_image_publisher.publish(detection_msg)
            #end_time = time.time() #End the timer
            #fps = 1/np.round(end_time - start_time, 2)
        except Exception as e:
            self.get_logger().error(f"Error in camera_sub: {e}")

            #cv2.putText(detection_image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)


    
    def predict(self, frame):
        res = self.model(frame)
        return res
    
    def draw_results(self, frame, results):
        # TODO: draw your bounding boxes here
        xyxys = []
        confs = []
        class_ids = []
        detections = []
        boxes = []
        for result in results:
            class_id = result.boxes.cls.cpu().numpy().astype(int)

            if len(class_id) == 0:
               continue
            
            if len(class_id) > 1:
                class_id = class_id[0]

            if class_id == 0:
                xyxys.append(result.boxes.xyxy.cpu().numpy())
                confs.append(result.boxes.conf.cpu().numpy())
                class_ids.append(result.boxes.cls.cpu().numpy())
                boxes.append(result.boxes)

                detections = sv.Detections(xyxy=result.boxes.xyxy.cpu().numpy(),
                                           confidence=result.boxes.conf.cpu().numpy(),
                                           class_id = result.boxes.cls.cpu().numpy().astype(int))
            
            #print(detections)
            #exit()
            #Format custom labels
            self.labels = [f"{self.CLASS_NAMES_DICT[class_id]}{confidence:0.2f}"
                           for _, _,confidence, class_id, tracker_id,_
            in detections]

        #Annotate and display frame
        frame = self.bbox_annotator.annotate(scene=frame, detections=detections)
        final_frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=self.labels)
        return final_frame, boxes
        #return bounding_box_frame
    
if __name__ == "__main__":
    rclpy.init()
    detector = ObjectDetector()
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()