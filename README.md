This is the official implementation of DAB-DETR(incl. Deformable-DETR) i.e. A Transformer architecture with Dynamic Anchor Boxes and Deformable Attention, that performs both End-to-End Object Detection as well as Multi-Object-Tracking(MOT). This project is being carried out towards the completion of the MSc. thesis carrying 32.5 ECTS, in order to be awarded a graduate degree in Autonomous Systems Engineering from the Technical University of Denmark.

The duration of the thesis project is approximately 5 months and 15 days.  

It has been carried out in collaboration with both the university as well as Capra Robotics A/S, a robotics enterprise based in Aarhus, Denmark. 

The dataset provided by Capra contains 4367 images of Chewing Gums and 3554 images of Cigarette butts, complete with their corresponding label files(.txt files containing data for BBOX detections and corresponding class labels in YOLO format). Further data augmentations were carried out to incorporate numerous special effects into the original images. These include horizontal and vertical flipping, zoom-in/zoom-out, adding rain, snow or fog, altering brightness contrast etc. More advanced transformations such as Mixup or RandAugment were also considered. 

