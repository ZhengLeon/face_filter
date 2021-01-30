from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
import torch.nn.functional as F
import torchvision.transforms as transforms

resnet = InceptionResnetV1(pretrained='vggface2').eval()
img_probe = Image.open('/home/zhengly/datasets/member_photo/probe/rj1_90.jpg')
img_gallery_path = "/home/zhengly/datasets/member_photo/gallery/"
img_gallery_list = os.listdir(img_gallery_path)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])

img_probe_cropped = transform(img_probe)
img_probe_embedding = resnet(img_probe_cropped.unsqueeze(0))
result = {}
for img in img_gallery_list:
    img_gal = Image.open(img_gallery_path+'/'+img)
    img_gal_cropped = transform(img_gal)
    img_gal_embedding = resnet(img_gal_cropped.unsqueeze(0))

    distance = F.pairwise_distance(img_probe_embedding, img_gal_embedding, p=2).mean()
    result[img]=distance.item()
print(min(result, key=result.get))
