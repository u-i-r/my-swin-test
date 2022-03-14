import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import MyDataset, get_data_list
from swin_transformer_pytorch import SwinTransformer
import time

def main():
	train_path = "my-swin-gpu/data/imagenet/train"
	val_path = "my-swin-gpu/data/imagenet/val"
	batch_size = 16
	num_workers = 1
	lr = 0.002
	momentum = 0.9
	device = torch.device("cuda")
	
	train_data, train_cls = get_data_list(train_path)
	train_set = MyDataset(train_data, train_cls)

	train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers)

	val_data, val_cls = get_data_list(val_path)
	val_set = MyDataset(val_data, val_cls)
	val_loader = DataLoader(val_set, batch_size, shuffle=False)

	model = SwinTransformer(
	    hidden_dim=96,
	    layers=(2, 2, 18, 2),
	    heads=(3, 6, 12, 24),
	    channels=3,
	    num_classes=50,
	    head_dim=32,
	    window_size=7,
	    downscaling_factors=(4, 2, 2, 2),
	    relative_pos_embedding=False
	)

	model.to(device)

	'''a = torch.randn(1, 3, 224, 224).to(device)
				logits = model(a)
				print(logits)'''

	optimizer = torch.optim.SGD(model.parameters(), lr, momentum)
	criterion = nn.CrossEntropyLoss()

	loss_list = []

	for epoch in range(50):

		time_start = time.time()

		model.train()

		for idx, (data, target) in enumerate(train_loader):
			data, target = Variable(data), Variable(target)
			data = data.to(device)
			target = target.to(device)
			optimizer.zero_grad()
			output = model(data)
			loss = criterion(output, target)
			loss.backward()
			optimizer.step()
			if idx % 10 == 0:
				loss_list.append(loss.item())
				time_end = time.time()
				time_used = time_end - time_start
				print("Train Epoch:{} {}/{} iters, loss: {:.6f}, use time: {}s".format(
					epoch, idx  * len(data), len(train_loader.dataset), loss.item(), time_used
					))
				time_start = time.time()

		loss_list.append(loss.item())

		model.eval()
		test_loss = 0
		correct = 0

		for data, target in val_loader:
			data, target = Variable(data, volatile=True), Variable(target)
			output = model(data)
			test_loss += criterion(output, target).data[0]
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).cpu().sum()

		val_loss /= len(val_loader.dataset)

		print("\nTest set: Average loss: {:.6f}, Accrucy: {}/{}\n".format(
			test_loss, correct, len(val_loader.dataset)
			))



if __name__ == "__main__":
	main()