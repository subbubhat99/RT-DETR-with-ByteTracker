#from collections import defaultdict

import cv2
#import numpy as np

from ultralytics import YOLO

# Load the YOLO-v8 model
model = YOLO('yolov8n.pt')

# Load the video file
vid_path = "C://Users//subbu//OneDrive//Desktop//DTU//MSc.Thesis//Cigarette_Vid-1.mp4"
cap = cv2.VideoCapture(vid_path)

# Store the track history
#track_hist = defaultdict(lambda: [])

#Define a VideoWriter Object
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output3.avi', fourcc, 20.0, (640,480))


#i = 0

while cap.isOpened():
    # Go through each frame of the video
    success, frame = cap.read()

    if success:
        # Run the Byte-Track tracker of YOLO-v8 on the current frame while retaining tracks between frames
        results = model.track(frame, persist=True)

        # Extract the bounding boxes and track IDs from the results
        #print(results[0].boxes)
        #boxes = results[0].boxes.xywh.cpu()
        #track_ids = results[0].boxes.id.int().cpu().tolist()

        # Show the results on the computed frame
        ann_frame = results[0].plot()

        # Plot the tracks
        #for box, track_id in zip(boxes, track_ids):
        #    x,y,w,h = box
        #    track = track_hist[track_id]
        #    track.append(float(x), float(y))
        #    if len(track) > 30:
        #        track.pop(0)

            # Draw the tracklines
            #    points = np.hstack(track).reshape(-l,1,2).astype(np.int32)
            #    cv2.polylines(ann_frame, [points], isClosed=False, color=(230,220,220), thickness=10)


        # Write the annotated frame to an output file
        #out.write(ann_frame)
        cv2.imshow(f"YOLO-Tracking Annotation", ann_frame)
        
        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # End the loop if the video came to an end
        break

# Release the video capture object and close the window
cap.release()
#out.release()
cv2.destroyAllWindows()



