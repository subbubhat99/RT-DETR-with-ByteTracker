import os
import torch
import numpy as np
import cv2
from time import perf_counter
from pathlib import Path
from ultralytics import RTDETR
import supervision as sv
from byte_tracker import BYTETracker
from byte_utils import YamlParser
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

SAVE_VIDEO = True
TRACKER = "bytetrack"

class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')

        # Declare parameters with default values
        self.declare_parameter('tracker', TRACKER)

        # Retrieve parameter values
        self.tracker = self.get_parameter('tracker').get_parameter_value().string_value

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using device: {self.device}")

        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names

        self.bbox_annotator = sv.BoundingBoxAnnotator(sv.ColorPalette.DEFAULT, thickness=3)
        self.label_annotator = sv.LabelAnnotator(sv.ColorPalette.DEFAULT, text_thickness=3, text_scale=1.5)

        # Load tracker
        if self.tracker == "bytetrack":
            path = os.path.dirname(os.path.abspath(__file__))
            tracker_cfg = f"{path}/../bytetrack.yaml"
            cfg = YamlParser()
            cfg.merge_from_file(tracker_cfg)
            self.tracker = BYTETracker(
                track_thresh=cfg.track_thresh,
                match_thresh=cfg.match_thresh,
                track_buffer=cfg.track_buffer,
                frame_rate=cfg.frame_rate
            )
        else:
            raise ValueError("Unsupported tracker")

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/oak/rgb/image_raw',  # Change this to your image topic
            self.listener_callback,
            10
        )

        self.publisher = self.create_publisher(
            Image,
            'detection_images_pt',  # Topic to publish processed images
            10
        )

        self.subscription  # prevent unused variable warning

        if SAVE_VIDEO:
            self.outputvid = cv2.VideoWriter('result_tracking-1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 8, (1280, 720))

    def load_model(self):
        path = os.path.dirname(os.path.abspath(__file__))
        model = RTDETR(f"{path}/../weights/best.pt")
        # model.fuse()
        return model

    def predict(self, frame):
        res = self.model(frame)
        return res

    def draw_results(self, frame, results):
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
                                           confidence=result.boxes.conf.cpu().numpy().astype(float),
                                           class_id=result.boxes.cls.cpu().numpy().astype(int))

            self.labels = [f"{self.CLASS_NAMES_DICT[class_id]}{confidence:0.2f}"
                           for _, _, confidence, class_id, tracker_id, _
                           in detections]

        frame = self.bbox_annotator.annotate(scene=frame, detections=detections)
    
        return frame, boxes

    def listener_callback(self, msg):
        prev_frames = [None]
        curr_frames = [None]
        try:
            # Convert ROS Image message to OpenCV image
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'Error converting image: {e}')
            return

        start_time = perf_counter()
        results = self.predict(frame)
        frame, _ = self.draw_results(frame, results)

        if hasattr(self.tracker, 'tracker') and hasattr(self.tracker.tracker, 'camera_update'):
            if prev_frames is not None and curr_frames is not None:
                self.tracker.tracker.camera_update(prev_frames, curr_frames)

        for result in results:
            box = result.boxes.data.cpu().numpy()
            outputs = self.tracker.update(box, frame)
            for i, (output) in enumerate(outputs):
                bbox = output[0:4]
                tracked_id = output[4]
                top_left = (int(bbox[-2] - 100), int(bbox[1]))
                cv2.putText(frame, f"ID : {tracked_id}", top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        end_time = perf_counter()
        fps = 1 / np.round(end_time - start_time, 2)
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        # Publish the processed image
        try:
            processed_image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.publisher.publish(processed_image_msg)
        except CvBridgeError as e:
            self.get_logger().error(f'Error converting image: {e}')

        if SAVE_VIDEO:
            self.outputvid.write(frame)
        
        cv2.imwrite("test_inference.png", frame) 

    def destroy_node(self):
        if SAVE_VIDEO:
            self.outputvid.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    detector = ObjectDetector()
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    finally:
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()