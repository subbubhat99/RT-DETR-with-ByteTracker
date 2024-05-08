#Program to Augment the Capra Dataset for 3D Object Detection & Classification 
import numpy as np
import cv2
import albumentations as A
import os
import glob
from albumentations.pytorch import ToTensorV2
import random
from tqdm import tqdm

def load_images_from_directory(directory):
    print("Loading images from directory")
    images = []
    image_path = os.listdir(directory)
    for filename in tqdm(image_path):
        if filename.endswith(('.png')):  # Check if the file is an image file
            filepath = os.path.join(directory, filename)
            image = cv2.imread(filepath)  # Load the image using OpenCV
            if image is not None:
                images.append(image)
    return images
def read_yolo_annotations(label_path):
      print("Reading YOLO annotations")
      bboxes = []
      class_labels = []
      num_bboxes = []
      label_path_iter = os.listdir(label_path)
      for filename in tqdm(label_path_iter):
            if filename.endswith('.txt'):
                  # print(f"Processing file: {filename}")  # Print the filename to keep track of the progress
                  file_path = os.path.join(label_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
    
            counter = 0
            img_bb = []
            img_label = []
            for line in lines:
                #try:
                    class_id, x, y, w, h = map(float, line.split())
                #    if any([x < 0, x > 1, y < 0, y > 1, w < 0, w > 1, h < 0, h > 1]):
                      #print(f"Invalid coordinates found in file {filename}:{line}")
                      # Skip the bounding box if it has invalid coordinates
                #      exit(1)
                    img_bb.append([x, y, w, h])
                    img_label.append(int(class_id))
                    counter += 1
                #except ValueError:
                    #print(f"Skipping line: {line}")
            num_bboxes.append(counter)
            bboxes.append(img_bb)
            class_labels.append(img_label)
      return bboxes, class_labels, num_bboxes

def data_aug(input_dir, output_dir, labels_dir, out_labels_dir):
                # load first 10 images from the directory
                imgs = load_images_from_directory(input_dir)
                bbs, class_anns, num_bbs = read_yolo_annotations(labels_dir)
                count = 0
                k = 0
                
                #Make sure that an output directory exists
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)


                
                # Define the list of all transformations
                transforms = [
                        A.HorizontalFlip(p=1.0),  # Apply horizontal flip with probability 0.5
                        A.RandomBrightnessContrast(p=1.0, brightness_limit=0.4, contrast_limit=0.01, brightness_by_max=False),  # Apply Random Brightness/contrast with probability 0.2, keeping brightness and contrast limits at 0.2 each
                        A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.7, alpha_coef=0.1, always_apply=False, p=1.0),
                        A.RandomSnow(snow_point_lower=0.3, snow_point_upper=0.6, brightness_coeff=2.5, always_apply=False, p=1.0),
                        A.RandomRain(brightness_coefficient=0.6, blur_value=1, slant_lower= -10, slant_upper= 10, drop_length= 20, drop_width= 1,rain_type="heavy", drop_color= (200, 200, 200), always_apply=False, p=1.0),
                       # A.RandomCrop(height=50, width=50, always_apply=False, p=0.2),
                        A.Resize(width=256, height=256, always_apply=False, p=1.0),
                        #A.MixUp(p=0.2, always_apply=False, alpha=0.2),

                    ]
                
                current_bb_idx = 0


                for idx, img in tqdm(enumerate(imgs)):
                    #if idx == 181:
                    #    X = 11
                    # Apply a transformation randomly selected from the above list
                    transform = random.choice(transforms)
                    num_bbs_current = num_bbs[idx]

                    # Create the augmentation pipeline
                    aug_ops = A.Compose([transform], bbox_params = A.BboxParams(format='yolo', label_fields = ['class_labels']))
                    
                    try:
                        transformed_img = aug_ops(image=img, bboxes=bbs[idx], class_labels=class_anns[idx])
                    except:
                        print(f"Error in image {idx}")
                        continue

                    #fog_img = A.add_fog(img=img, fog_coef=0.6, alpha_coef=0.4, haze_list=[]) # Add fog to the images
                    #snow_img= A.add_snow(img=img, snow_point=7, brightness_coeff=0.9)  # Add snow to the images
                    #rain_img = A.add_rain(img=img, brightness_coefficient=0.7, blur_value=2, rain_drops=[[2,2],[4,4],[6,6],[8,8]],slant=10, drop_length=20, drop_width=1, drop_color=(200,200,200))  # Add rain to the images
                    #resized_img = A.resize(img=img , width=256, height=256)  # Resize the images to 256x256
                    ##shadow_img = A.add_shadow(img=img, vertices_list=[])  # Add shadow to the images
                    #cropped_img = A.random_crop(img=img, crop_height=50, crop_width=50, h_start=2.5, w_start=2.5)  # Crop the images down by 50 along either dimension
                    #print(count)
                    #count += 1
                    
                    
                    
                    
                    cv2.imwrite(os.path.join(output_dir,f"transformed_Chewie_image_{count}.png"), transformed_img['image'])
                    
                    #cv2.imwrite(os.path.join(output_dir, f"fog_Chewie_image_{count}.png"), fog_img)
                    
                    #cv2.imwrite(os.path.join(output_dir, f"snow_Chewie_image_{count}.png"), snow_img)
                    #cv2.imwrite(os.path.join(output_dir, f"rain_Chewie_image_{count}.png"),rain_img)
                    #cv2.imwrite(os.path.join(output_dir, f"resized_Chewie_image_{count}.png"),resized_img)
                    #cv2.imwrite(os.path.join(output_dir, f"shadow_Chewie_image_{count}.png"),shadow_img)
                    #cv2.imwrite(output_path + f"cropped_Chewie_image_{count}.png",cropped_img[0])
                    
                    
                    #Save the transformed bounding boxes and labels to a text file
                    with open(os.path.join(out_labels_dir, f"transformed_Chewie_labels_{count}.txt"),'w') as f:
                        for bbox, label in zip(transformed_img['bboxes'][current_bb_idx: num_bbs_current+current_bb_idx],transformed_img['class_labels'][current_bb_idx:num_bbs_current+current_bb_idx]):
                            if bbox[1] < 0:
                                print(f"Invalid bounding box found in image {count}")
                            f.write(f"{label} {' '.join(map(str, bbox))}\n")
                            
                    current_bb_idx += num_bbs_current
                    #Increment the counters
                    count += 1
                
                
                

if __name__ == "__main__":
    in_directory = "C://Users//subbu//OneDrive//Desktop//DTU//MSc.Thesis//Chewie//data"
    out_directory = "C://Users//subbu//OneDrive//Desktop//DTU//MSc.Thesis//Capra_Aug//data"
    labels_directory = "C://Users//subbu//OneDrive//Desktop//DTU//MSc.Thesis//Chewie//labels//labels"
    out_labels_directory = "C://Users//subbu//OneDrive//Desktop//DTU//MSc.Thesis//Capra_Aug//labels"
    print("main")
    data_aug(in_directory, out_directory, labels_directory, out_labels_directory)
        


