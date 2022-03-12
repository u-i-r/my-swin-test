import os 
from torchvision import transforms
from PIL import Image

path1 = "data/imagenet/train"
path2 = "data/imagenet/val"

path_useless = "data/useless"

pic_paths = []

i = os.listdir(path1)

for j in i:
	k = os.listdir(os.path.join(path1, j))
	for l in k:
		pic_paths.append(os.path.join(path1, j, l))

i = os.listdir(path2)

for j in i:
	k = os.listdir(os.path.join(path2, j))
	for l in k:
		pic_paths.append(os.path.join(path2, j, l))

print("flie num:", len(pic_paths))

op = transforms.Compose([
	transforms.Resize((224,224)),
	transforms.ToTensor()
	])

for f in pic_paths:
	image = Image.open(f)
	tensor = op(image)
	if tensor.shape[0] == 1:
		cmd = "move " + f + " " + path_useless
		os.system(cmd)