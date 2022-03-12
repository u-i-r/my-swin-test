import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image

def get_data_list(image_path):
	folder_list = os.listdir(image_path)
	img_list = []
	cls_list = []
	for cls_no in folder_list:
		for img in os.listdir(os.path.join(image_path, cls_no)):
			img_list.append(os.path.join(image_path, cls_no, img))
			cls_list.append(int(cls_no))
	return img_list, cls_list

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	#normalize
	])

def default_loader(path):
	img_pil = Image.open(path)
	#img_pil = img_pil.resize((3, 224,224))
	img_tensor = preprocess(img_pil)
	#print("tensor shape", img_tensor.shape)
	return img_tensor

class MyDataset(Dataset):
	def __init__(self, img_path_list, cls_list, loader = default_loader):
		self.data = img_path_list
		self.label = cls_list
		self.loader = loader

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		img_path = self.data[index]
		img = self.loader(img_path)
		if img.shape[0] == 1:
			#print("pic of 1 channels:", img_path)
			img = img.expand(3, 224, 224)   #单通道黑白图像，扩展为3通道

		target = self.label[index]
		return img, target




if __name__ == "__main__":
	imagenet_path = "data/imagenet"
	make_dataloader(imagenet_path)
