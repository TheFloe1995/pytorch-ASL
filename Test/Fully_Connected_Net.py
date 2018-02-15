from torch import nn
import torch.nn.functional as F

class Fully_Connected_Net(nn.Module):
    def __init__(self, in_size=784, out_size = 10):
        super(Fully_Connected_Net, self).__init__()
        self.in_size = in_size

        self.lin1 = nn.Linear(in_size, 100)
        self.lin2 = nn.Linear(100, 100)
        self.lin3 = nn.Linear(100, out_size)

    def forward(self, input):
        x = input.view(input.size()[0], self.in_size)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda
