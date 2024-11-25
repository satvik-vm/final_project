import torch.nn as nn

class FCLayer(nn.Module):
	def __init__(self, input_size, output_size):
		super(FCLayer, self).__init__()
		self.fc = nn.Linear(input_size, output_size)
		# self.dropout=nn.Dropout(p=0.3, inplace=False)
		self.bnorm = nn.BatchNorm1d(output_size)
		self.relu = nn.ReLU(inplace=True)
		# self.residual = input_size == output_size

	def forward(self, x):
		x = self.fc(x)
		x = self.bnorm(x)
		x = self.relu(x)
		return x


class BridgeNetwork(nn.Module):
	def __init__(self, sizes):
		super(BridgeNetwork, self).__init__()
		self.sizes = sizes
		self.layers = self._make_layers()

	def _make_layers(self):
		layers = []
		for i in range(len(self.sizes) - 1):
			layers.append(FCLayer(self.sizes[i], self.sizes[i + 1]))
		return nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)

def bridge_network(sizes):
	return BridgeNetwork(sizes)