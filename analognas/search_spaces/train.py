import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pyvww

f = open("result_train.txt", "a")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    f.write("CUDA")
else: 
    f.write("CPU")
# Dataset 
from torchvision import transforms

transform = transforms.Compose([
    transforms.CenterCrop(100),
    transforms.ToTensor()
])

train_dataset = pyvww.pytorch.VisualWakeWordsClassification(root="/dccstor/vww_dataset/dataset/all2014", 
                    annFile="/dccstor/vww_dataset/dataset/visualwakewords-dataset/annotations/instances_train.json", transform= transform) 
valid_dataset = pyvww.pytorch.VisualWakeWordsClassification(root="/dccstor/vww_dataset/dataset/all2014", annFile="/dccstor/vww_dataset/dataset/visualwakewords-dataset/annotations/instances_val.json", transform= transform)

from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=1)

# Training parameters
SEED = 1
N_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.1
N_CLASSES = 2
WEIGHT_SCALING_OMEGA = 0.6  # Should not be larger than max weight.


from resnet_macro_architecture import Network
from config_space import ConfigSpace

CS = ConfigSpace("VWW") 
config = CS.sample_arch()
network = Network(config).to(device) 


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=LEARNING_RATE, momentum=0.9)


for epoch in range(N_EPOCHS):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        #print(labels) 
        outputs = network(inputs)
        #print(outputs) 
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        f.write(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss}\n')
            

f.write('Finished Training')




