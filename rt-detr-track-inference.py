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

SAVE_VIDEO = True
TRACKER = "bytetrack"

class ObjectDetector:

    def __init__(self, capture_ind):
        
        self.capture_ind = capture_ind
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using device: ", self.device)
        
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names

        self.bbox_annotator = sv.BoundingBoxAnnotator(sv.ColorPalette.default(), thickness=3)
        self.label_annotator = sv.LabelAnnotator(sv.ColorPalette.default(), text_thickness=3, text_scale=1.5)

        #Depending on which tracker is used, RE-ID model weights may also be loaded(eg: with StrongSORT)
        #reid_weights = Path("weights/osnet_x0_25_msmt17.pt")

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
 #       else:
  #          tracker_cfg = "rtdetr_pytorch/configs/tracking/strongsort.yaml"
   #         cfg = YamlParser()
    #        cfg.merge_from_file(tracker_cfg)
#
 #           self.tracker = StrongSORT(
  #              reid_weights, 
   #             torch.device(self.device),
    #            False,
     #           max_dist = cfg.strongsort.max_dist, 
      #          max_iou_dist = cfg.strongsort.max_iou_dist,
       #         max_duration = cfg.strongsort.max_duration,
        #        max_unmatched_preds = cfg.strongsort.max_unmatched_preds, 
          #      n_init = cfg.strongsort.n_init, #Number of frames to wait before activating the tracker
         #       nn_budget = cfg.strongsort.nn_budget,
          #      mc_lamda = cfg.strongsort.mc_lambda,
           #     ema_alpha = cfg.strongsort.ema_alpha
            #)

    def load_model(self):

        model = RTDETR("weights/best3.engine")
        model.fuse()

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
    
    def __call__(self):
        
        vid_path = self.vid_path
        cap = cv2.VideoCapture(vid_path)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 478)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 850)

        if SAVE_VIDEO:
            outputvid = cv2.VideoWriter('result_tracking-1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 8, (1280,720))
        #Set up the tracker
        tracker = self.tracker

        if hasattr(tracker, 'model'):
            if hasattr(tracker.model, 'warmup'):
                tracker.model.warmup()

        outputs = [None]
        curr_frames, prev_frames = None, None

        while True:
            start_time = perf_counter()
            ret, frame = cap.read()
            print("ret")
            print(ret)
            #print("frame")
            ##print(frame)
            assert ret
            results = self.predict(frame)

            frame, _ = self.draw_results(frame, results)

            if hasattr(tracker, 'tracker') and hasattr(tracker.tracker, 'camera_update'):
                if prev_frames is not None and curr_frames is not None:
                    tracker.tracker.camera_update(prev_frames, curr_frames)
            #print(len(results))
            for result in results:
                #print("results")
                #print(results)
                box = result.boxes.data.cpu().numpy()
                #print("Coordinates of 1 bounding box: ",box)
                outputs[0] = tracker.update(box, frame)
                #print("outputs ka first element")
                #print(outputs[0])
                for i, (output) in enumerate(outputs[0]):
                    bbox = output[0:4]
                    #print("bbox")
                    #print(bbox)

                    tracked_id = output[4]

                    top_left = (int(bbox[-2] - 100), int(bbox[1]))

                    cv2.putText(frame, f"ID : {tracked_id}", top_left, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

            end_time = perf_counter()
            fps = 1/np.round(end_time - start_time, 2)

            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print("Plotting")
            cv2.imshow('RT-DETR DETECTION',frame)
            #plt.title('RT-DETR DETECTION')

            if SAVE_VIDEO:
                outputvid.write(frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        if SAVE_VIDEO:
            outputvid.release()
        
        cap.release()
        #cv2.destroyAllWindows()

if __name__ == "__main__":

    detector = ObjectDetector(r"C:\\Users\\subra\\Desktop\\DTU\\MSc.Thesis\\Cigarette_Vid-1_Annotations\\obj_train_data\\Cigarette_Vid-1.mp4")
    detector()





