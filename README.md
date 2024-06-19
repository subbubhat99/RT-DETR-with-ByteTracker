This project is carried out in collaboration with Capra Robotics ApS and the Technical University of Denmark towards the completion of a Master's thesis project worth 32.5 ECTS towards partial fulfillment of the graduate degree in Autonomous Systems Engineering. 

ABSTRACT

Owing to widespread concern regarding threat to our environment caused by rising levels of global warming and the frequent occurrence of natural disasters around the world, there has been an increased emphasis on tackling the problem of climate change. On an individual level, this also includes how we maintain cleanliness in our surrounding public spaces. As Chewing gums and cigarettes of various kinds are two of the most commonly found waste items lying around the streets of Denmark, this project is aimed at building a Computer Vision algorithm to be integrated on mobile robots, built by a Danish robotics company i.e. Capra Robotics. The resulting algorithm must be capable of detecting, accurately classifying and tracking multiple instances of these two objects. The technique developed builds upon a variant of the widely renown DETR model for End-to-End object detection i.e. Real-Time DETR. This is extended to also perform Multi-Object Tracking using a heuristics-based technique known as ByteTracker. The resulting hybrid is associated with a novel re-identification functionality handled by RT-DETR itself, replacing conventionally used CNN-based networks added on top of the mainstream framework, such as OS-Net and MobileNet-v2 that were previously needed to deal with such edge-cases. Upon testing and observing performance metrics obtained on the project dataset as well as public benchmarks such as MOT-17 and TACO, it is found to be fairly competitive and on-par with other state-of-the-art object detectors and trackers such as YOLO-v8, DeTR, and TrackFormer, Strong-SORT++ respectively. \\
The result of field tests carried out with this algorithm integrated onto the cigarette-collecting robot Butty, with Tensor-RT integration, is also discussed in the latter half of the report.


Model weights for RT-DETR-l pre-trained on the COCO-2017 dataset were obtained from: https://docs.ultralytics.com/models/rtdetr/.

The ByteTracker implementation is referenced and modified from: https://github.com/mikel-brostrom/yolo_tracking/tree/master/boxmot/trackers/bytetrack

