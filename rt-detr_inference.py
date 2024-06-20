import numpy
import ultralytics
from ultralytics import RTDETR
import torch
from PIL import Image
import torchvision.transforms as transforms
#from torch2trt import torch2trt


def main():
    
    
    model = RTDETR("weights/best3.pt").cuda()
    #Load an image as the dummy input
    img = Image.open("C:\\Users\\subra\\Desktop\\DTU\\MSc.Thesis\\Capra_Aug\\data\\transformed_image_330.png")
    #transformation = transforms.Compose([
     #   transforms.Resize((320,320)),
      #  transforms.ToTensor()
    #])
    #x = transformation(img).unsqueeze(0).cuda()
    #model_trt = torch2trt(model, [x])
    #torch.onnx.export(model, [x],"weights/rtdetr.onnx", verbose=True, opset_version=11, input_names=['input'], output_names=['output'])
    result = model(img)
    path = model.export(format="engine")

if __name__ == '__main__':
    main()
