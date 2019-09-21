## TODO: define the convolutional neural network architecture
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # self.conv1 = nn.Conv2d(1, 32, 5)
        
        # The input image size is 224 x 224, according to the section "Define a data transform" in Notebook 2
        self.conv1 = nn.Conv2d(1, 32, 5) # Image size: 220 (224 - 2x2) x 220
        self.pool1 = nn.MaxPool2d(2, 2) # 110 (220 /2) x 110
        self.drop1 = nn.Dropout2d(p=0.1)

        self.conv2 = nn.Conv2d(32, 64, 3) # 108 (110 - 1x2) x 108
        self.pool2 = nn.MaxPool2d(2, 2) # 54 x 54
        self.drop2 = nn.Dropout2d(p=0.2)

        self.conv3 = nn.Conv2d(64, 128, 2)  # 52 (54 - 1x2) x 52
        self.pool3 = nn.MaxPool2d(2, 2)  # 26 x 26
        self.drop3 = nn.Dropout2d(p=0.3)

        self.conv4 = nn.Conv2d(128, 256, 1)  # 26 x 26
        self.pool4 = nn.MaxPool2d(2, 2) # 13  x 13
        self.drop4 = nn.Dropout2d(p=0.4)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # Qs: Any sense to apply a convolution with size 1? 
        # Example to determine H_input and how to use "view" to reshape the input; 
        # https://towardsdatascience.com/model-summary-in-pytorch-b5a1e4b64d25
        self.fc1 = nn.Linear(256*13*13, 256)
        self.drop5 = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(256, 256)
        self.drop6 = nn.Dropout2d(p=0.6)
        self.fc3 = nn.Linear(256, 136)
               

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        x = F.relu(self.conv1(x))
        x = self.drop1(self.pool1(x))

        x = F.relu(self.conv2(x))
        x = self.drop2(self.pool2(x))

        x = F.relu(self.conv3(x))
        x = self.drop3(self.pool3(x))

        x = F.relu(self.conv4(x))
        x = self.drop4(self.pool4(x))

        x = x.view(-1, self._num_flat_size(x))
        x = F.relu(self.fc1(x))
        x = self.drop5(x)
        x = F.relu(self.fc2(x))
        x = self.drop6(x)
        x = F.tanh(self.fc3(x))
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

    def _num_flat_size(self, x):
        input_dims = x.size()[1:]
        num_input_size = 1
        for d in input_dims:
            num_input_size *= d

        return num_input_size


if __name__ == "__main__":
    net = Net()
    input = torch.randn(1, 1, 224, 224)
    net(input)