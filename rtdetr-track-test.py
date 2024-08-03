from ultralytics import RTDETR

def main():
    try:

        model = RTDETR("RT-DETR/final_run/weights/rtdetr.pt")
        model.cuda()
        results = model.track(source="Cigarette_Vid-4.mp4", stream=True, show=True, tracker = "bytetrack.yaml")
        #print("Tracking completed successfully")
    #except Exception as e:
    #    print(f"An error has occurred: {e}")
        with open("tracking_results4.txt", "w") as file:
            for frame_idx, frame_results in enumerate(results):
                for frame_result in frame_results:
                    boxes = frame_result.boxes
                    if boxes is not None:
                        for box in boxes:
                            print(f"Box: {box}")  # Debug print to inspect the box object
                            print(f"Box.xywh: {box.xywh}")  # Debug print to inspect the xywh attribute
                            box_xywh_np = box.xywh.cpu().numpy()
                            if box_xywh_np.shape[1] < 4:
                                print(f"Error: box.xywh has less than 4 elements: {box.xywh}")
                                continue  # Skip this box if it doesn't have enough elements
                            track_id = box.id.item() if box.id is not None else 0
                            bbox_x1 = box_xywh_np[0,0].item()
                            bbox_y1 = box_xywh_np[0,1].item()
                            bbox_x2 = box_xywh_np[0,2].item()
                            bbox_y2 = box_xywh_np[0,3].item()
                            conf_score = box.conf.item() if hasattr(box, 'conf') else 1


                    file.write(f"{int(frame_idx+1)},{int(track_id)},{bbox_x1:.2f},{bbox_y1:.2f},{(bbox_x2-bbox_x1):.2f},{(bbox_y2-bbox_y1):.2f},{conf_score:.2f},-1,-1,-1\n")
                file.write("\n")

        print("Tracking completed successfully")
    except Exception as e:
        print(f"An error has occurred: {e}")

if __name__ == "__main__":
    main()
