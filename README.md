This project is carried out in collaboration with Capra Robotics ApS and the Technical University of Denmark towards the completion of a Master's thesis project worth 32.5 ECTS towards partial fulfillment of the graduate degree in Autonomous Systems Engineering. 

The project is based on the theme "Simultaneous Object Detection and Multi-Object Tracking of Street Waste using Computer Vision and Deep Learning", seeking to build a novel deep learning framework that is formed as a hybrid of a Real-Time Detection Transformer for object detection, coupled with ByteTracker for incorporating a Tracking-by-Detection functionality into the model, enabling both tasks to be accomplished on the desired targets i.e. Cigarettes and Chewing gums found lying on the streets and other public spaces. 

The resulting model is then incorporated onto the mobile robots operated by Capra, replacing the currently used YOLO-v8n model that only deals with the detection task.
While Capra operates a separate mobile robot for dealing with the cleanup of each of these wastes, namely Chewie and Butty, the model was only integrated and tested through field trials with Butty for the purpose of this thesis.

Model weights for RT-DETR-l pre-trained on the COCO-2017 dataset were obtained from: https://docs.ultralytics.com/models/rtdetr/.

The ByteTracker implementation is referenced and modified from: https://github.com/mikel-brostrom/yolo_tracking/tree/master/boxmot/trackers/bytetrack

