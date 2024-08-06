This project is carried out in collaboration with Capra Robotics ApS and the Technical University of Denmark towards the completion of a Master's thesis project worth 32.5 ECTS towards partial fulfillment of the graduate degree in Autonomous Systems Engineering. 

ABSTRACT

Owing to widespread concern regarding threat to our environment caused by rising levels of global warming and the frequent occurrence of natural disasters around the world, there has been an increased emphasis on tackling the problem of climate change. On an individual level, this also includes how we maintain cleanliness in our surrounding public spaces. As Chewing gums and cigarettes of various kinds are two of the most commonly found waste items lying around the streets of Denmark, this project is aimed at building a Computer Vision algorithm to be integrated on mobile robots, built by Capra Robotics A/S. The resulting algorithm must be capable of detecting, accurately classifying and tracking multiple instances of these two objects. The technique developed builds upon a variant of the widely renown DETR model for End-to-End object detection i.e. Real-Time DETR. This is extended to also perform Multi-Object Tracking using a heuristics-based technique known as ByteTracker. Upon testing and observing performance metrics obtained on the project dataset as well as a custom-designed dataset of videos annotated according to the MOT20 benchmark convention, it is found to be fairly competitive and on-par with other state-of-the-art object detectors and trackers such as YOLO-v8n, DeTR, and TrackFormer, Strong-SORT++ respectively. \\
The result of field tests carried out with this algorithm integrated onto the cigarette-collecting robot Butty with a Tensor-RT Inference optimizer engine, is discussed in the latter half of the report.


Model weights for RT-DETR-l pre-trained on the COCO-2017 dataset were obtained from: https://docs.ultralytics.com/models/rtdetr/.

The ByteTracker implementation references and is modified from: https://github.com/mikel-brostrom/yolo_tracking/tree/master/boxmot/trackers/bytetrack.

