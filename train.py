import wrn_mixup_model
import fully_connected
import bridge_network
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from data.datamgr import SimpleDataManager , SetDataManager
from io_utils import model_dict, parse_args, get_resume_file ,get_assigned_file
import configs
import numpy as np
import random
import datetime
import torch.optim.lr_scheduler as lr_scheduler
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


num_classes = 200
image_size = 80

use_gpu = torch.cuda.is_available()

bridge_input_size = 640
bridge_output_size = 640

# class cross_entropy_loss():
# 	def __init__(self):
# 		super(cross_entropy_loss, self).__init__()
# 		self.eps = torch.tensor(np.finfo(float).eps, requires_grad=False)

# 	def loss(self, ypred, ytruth):
# 		# print(self.eps.size())
# 		print(ytruth.size())
# 		print(ypred.size())
# 		cross_entropy = -torch.mean(ytruth * torch.log(ypred + self.eps))
# 		return cross_entropy

class cross_entropy_loss():
	def __init__(self):
		self.eps = torch.tensor(np.finfo(float).eps, requires_grad=False)

	def loss(self, ypred, ytruth):
		ytruth = ytruth.long()
		log_preds = torch.log(ypred + self.eps)
		correct_class_log_probs = log_preds[torch.arange(ytruth.size(0)), ytruth]

		cross_entropy = -torch.mean(correct_class_log_probs)
		return cross_entropy

def train(base_loader, base_loader_test, cnn_model, dnn_model, bridge_model, stop_epoch, dnn_optimizer, dnn_scheduler, bridge_optimizer, bridge_scheduler, params):
	tloss = []
	bloss = []
	itr_ctr = 0
	tot_bloss = 0.
	rbeg = datetime.datetime.now()
	tot_train_batch = len(base_loader)

	for param in cnn_model.parameters():
		param.requires_grad = False

	df = pd.DataFrame(columns=['Epoch', 'Mixup Layers', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy'])
	output_file = os.path.join(params.checkpoint_dir, 'output.csv')


	# dnn_optimizer = optim.Adam(dnn_model.parameters(), lr=0.001)
	# criterion = nn.CrossEntropyLoss()
	criterion = cross_entropy_loss()


	for epoch in range(stop_epoch):
		print('\nEpoch: %d' % epoch)
		cnn_model.eval()
		# cnn_model.train()
		bridge_model.train()
		dnn_model.train()

		mixup_layers = []

		train_loss = 0
		reg_loss = 0
		correct = 0
		correct1 = 0.0
		total = 0
		tot_bloss = 0.

		# for name, param in bridge_model.named_parameters():
		# 	print(name)

		# for name, params in dnn_model.named_parameters():
		# 	print(name)

		# for name, params in cnn_model.named_parameters():
		# 	print(name)

		for batch_idx, (input_var, target_var) in enumerate(base_loader):
			input_var, target_var = input_var.to(device), target_var.to(device)
			input_var, target_var = Variable(input_var), Variable(target_var)
			lam = np.random.beta(params.alpha, params.alpha)
			shuffle = torch.randperm(input_var.shape[0])
			try:
				mixup_layer = random.choice(dnn_model.mixup_layers)
			except IndexError:
				mixup_layer = -1

			mixup_layers.append(mixup_layer)

			# with torch.no_grad():
			out , _ , _ , _  = cnn_model(input_var, target_var, mixup_hidden= True, mixup_alpha = params.alpha , lam = lam)
			# out = bridge_model(out)
			if mixup_layer != -1:
				out, _ = dnn_model([out, shuffle, lam, mixup_layer])
			else:
				out, _ = dnn_model(out)

			output = nn.Softmax(dim=1)(out)
			_, predicted = torch.max(output.data, 1)
			# predicted_classes = torch.argmax(output, dim=1, keepdim=True)


			if lam is not None:
				target = lam * target_var + (1 - lam) * target_var[shuffle]
			else:
				target = target_var
			target = target.to(device)

			loss = criterion.loss(output, target)
			# loss = criterion(out, target)
			tot_bloss += loss
			bloss.append(loss.detach().cpu().data.numpy())

			itr_ctr += 1

			dnn_optimizer.zero_grad()
			bridge_optimizer.zero_grad()
			# tot_bloss.backward(retain_graph=True)
			loss.backward(retain_graph=True)
			# print("Batch Id: ", batch_idx, ", Loss: ", loss.item())
			# for name, param in bridge_model.named_parameters():
			# 	if param.grad is not None:
			# 		print(f"Gradient for {name}: {param.grad.abs().mean().item()}")
			# 	else:
			# 		print(f"No gradient for {name}")

			# for name, param in dnn_model.named_parameters():
			# 	if param.grad is not None:
			# 		print(f"Gradient for {name}: {param.grad.abs().mean().item()}")
			# 	else:
			# 		print(f"No gradient for {name}")
			dnn_optimizer.step()
			# dnn_scheduler.step()
			# bridge_optimizer.step()
			# bridge_scheduler.step()
			# tot_bloss = 0.
			ls = np.mean(bloss)
			tloss.append(ls)
			bloss = []

			total += target_var.size(0)
			correct += (lam * predicted.eq(target_var.data).cpu().sum().float()
						+ (1 - lam) * predicted.eq(target_var[shuffle].data).cpu().sum().float())

			if batch_idx % params.print_batch_freq == 0 or batch_idx == len(base_loader) - 1:
				print('{0}/{1}'.format(batch_idx,len(base_loader)), "Mixup Layer: ", mixup_layer, end=" | ")
				print('Loss: %.9f | Acc: %.3f%%  '
							 % (tot_bloss/(batch_idx+1),100.*correct/total ))

		if not os.path.isdir(params.dnn_checkpoint_dir):
			os.makedirs(params.dnn_checkpoint_dir)

		if not os.path.isdir(params.bridge_checkpoint_dir):
			os.makedirs(params.bridge_checkpoint_dir)

		if not os.path.isdir(params.checkpoint_dir):
			os.makedirs(params.checkpoint_dir)

		if (epoch % params.save_freq == 0) or (epoch==stop_epoch-1):
			print("SAVE FILE EPOCH")
			outfile = os.path.join(params.dnn_checkpoint_dir, f'{epoch}.tar')
			torch.save({'epoch':epoch, 'state':dnn_model.state_dict() }, outfile)

			outfile = os.path.join(params.bridge_checkpoint_dir, f'{epoch}.tar')
			torch.save({'epoch':epoch, 'state':bridge_model.state_dict() }, outfile)

		cnn_model.eval()
		dnn_model.eval()
		bridge_model.eval()
		loss = []
		acc = []

		with torch.no_grad():
			correct_val = 0
			total_val = 0
			for batch_idx, (input_var, target_var) in enumerate(base_loader):
				input_var, target_var = input_var.to(device), target_var.to(device)
				out, _ = cnn_model.forward(input_var)
				# out = bridge_model(out)
				output, _ = dnn_model.forward(out)
				target = target_var.to(device)
				output = nn.Softmax(dim=1)(output)
				loss.append(criterion.loss(output, target).cpu().data)
				# pred = output.data.max(1, keepdim=True)[1]
				_, predicted = torch.max(output.data, 1)
				# print(batch_idx)

				acc.append(predicted.eq(target.data).cpu().float().mean().numpy())
				# correct_val += predicted.eq(target_var.data).cpu().sum().float()
				# total_val += target_var.size(0)

		train_loss = np.mean(loss)
		train_accuracy = np.mean(acc)
		# eval_accuracy = correct_val / total_val
		print("Train Dataset")
		print('Loss: %.9f | Acc: %.3f%%'
							 % (train_loss/(batch_idx+1), 100.*train_accuracy ))

		# torch.cuda.empty_cache()

		with torch.no_grad():
			correct_val = 0
			total_val = 0
			for batch_idx, (input_var, target_var) in enumerate(base_loader_test):
				input_var, target_var = input_var.to(device), target_var.to(device)
				out, _ = cnn_model.forward(input_var)
				# out = bridge_model(out)
				output, _ = dnn_model.forward(out)
				target = target_var.to(device)
				output = nn.Softmax(dim=1)(output)
				loss.append(criterion.loss(output, target).cpu().data)
				# pred = output.data.max(1, keepdim=True)[1]
				_, predicted = torch.max(output.data, 1)
				# print(batch_idx)

				acc.append(predicted.eq(target.data).cpu().float().mean().numpy())
				# correct_val += predicted.eq(target_var.data).cpu().sum().float()
				# total_val += target_var.size(0)

		eval_loss = np.mean(loss)
		eval_accuracy = np.mean(acc)
		# eval_accuracy = correct_val / total_val
		print("Validation Dataset")
		print('Loss: %.9f | Acc: %.3f%%'
							 % (eval_loss/(batch_idx+1), 100.*eval_accuracy ))
		# print(mixup_layers)

		torch.cuda.empty_cache()

		new_row = pd.DataFrame([[epoch, mixup_layers, train_loss/(batch_idx+1), 100.*train_accuracy, eval_loss/(batch_idx+1), 100.*eval_accuracy]], columns=['Epoch', 'Mixup Layers', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy'])

		df = pd.concat([df, new_row], ignore_index=True)

		df.to_csv(output_file)

	return cnn_model, bridge_model, dnn_model


if __name__ == "__main__":
	params = parse_args('train')

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	cnn_model_path = "../checkpoints/train_64_val_16_test_20/30.tar"

	cnn_model = wrn_mixup_model.wrn28_10(num_classes=200).to(device)

	checkpoint = torch.load(cnn_model_path, weights_only=False)

	cnn_model.load_state_dict(checkpoint['state'])

	for name, param in cnn_model.named_parameters():
		param.requires_grad = False

	if not params.without_mixup:
		dnn_model = fully_connected.mixup_model(num_classes=params.num_classes, inputsize=bridge_output_size, bottleneck_size=params.bottleneck).to(device)
	else:
		dnn_model = fully_connected.mixup_model(num_classes=params.num_classes, inputsize=bridge_output_size, bottleneck_size=params.bottleneck, mixup_layers=[]).to(device)

	bridge_sizes = [bridge_input_size, bridge_output_size]

	bridge_model = bridge_network.bridge_network(sizes=bridge_sizes).to(device)

	params.dnn_checkpoint_dir = '%s/checkpoints/dnn/temp/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
	params.bridge_checkpoint_dir = '%s/checkpoints/bridge/temp/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
	params.checkpoint_dir = '%s/checkpoints/temp' %(configs.save_dir)


	milestones = [100, 200, 400, 500]
	dnn_optimizer = optim.Adam(dnn_model.parameters(), lr=0.001)
	dnn_scheduler = lr_scheduler.MultiStepLR(
		dnn_optimizer, milestones=milestones, gamma=0.01)

	bridge_optimizer = optim.Adam(bridge_model.parameters(), lr=0.001)
	bridge_scheduler = lr_scheduler.MultiStepLR(
		bridge_optimizer, milestones=milestones, gamma=0.01)

	base_file = configs.data_dir[params.dataset] + 'base.json'
	val_file = configs.data_dir[params.dataset] + 'val.json'
	start_epoch = params.start_epoch
	stop_epoch = params.stop_epoch

	# for name, param in dnn_model.named_parameters():
	# 	print(f"Parameter: {name}, requires_grad: {param.requires_grad}")

	# for name, param in bridge_model.named_parameters():
	# 	print(f"Parameter: {name}, requires_grad: {param.requires_grad}")

	base_datamgr    = SimpleDataManager(image_size, batch_size = params.batch_size)
	base_loader     = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
	base_datamgr_test    = SimpleDataManager(image_size, batch_size = params.test_batch_size)
	base_loader_test     = base_datamgr_test.get_data_loader( val_file , aug = False )

	train(base_loader, base_loader_test, cnn_model, dnn_model, bridge_model, stop_epoch, dnn_optimizer, dnn_scheduler, bridge_optimizer, bridge_scheduler, params)
