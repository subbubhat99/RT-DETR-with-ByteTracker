import torch
import torchvision
from ultralytics import RTDETR

def main():
    model = RTDETR("weights/rtdetr_r50vd_6x_coco_from_paddle.pt")
    input_tensor = torch.rand(1,3,320,320).cuda()
    traced_model = torch.jit.trace(model, input_tensor)
    traced_model.save("weights/rtdetr-l.pt")

if __name__ == '__main__':
    main()

