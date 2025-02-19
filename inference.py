import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# Define the same GCN model architecture used for training.
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Load the Cora dataset.
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Initialize the model with the same parameters as used during training.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(num_features=dataset.num_node_features, hidden_channels=16, num_classes=dataset.num_classes).to(device)
data = data.to(device)

# Load the saved model parameters.
model.load_state_dict(torch.load("gcn_model.pth", map_location=device))
model.eval()
print("Model loaded and set to evaluation mode.")

# Run inference on the test set.
with torch.no_grad():
    out = model(data)
    pred = out.argmax(dim=1)
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
    test_acc = correct / data.test_mask.sum().item()

print(f"Test Accuracy: {test_acc:.4f}")
