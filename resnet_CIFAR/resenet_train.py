import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from resnet import ResNet

#3 channels and 1 channel
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

batch_size = 128
epoch = 20

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=8,pin_memory=True)

PATH = './res_net_mnist.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = ResNet()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for _ in range(epoch):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs data is a list of [inputs, labels]
        # inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)#forward
        loss = criterion(outputs, labels)#get loss
        loss.backward()#backward
        optimizer.step()#optimize

        # print statistics
        running_loss += loss.item()
    print(f'[{_ + 1}/{epoch},] loss: {running_loss/(i+1) :.4f}')
    torch.save(net.state_dict(), PATH)        

print('Finished Training')
torch.save(net.state_dict(), PATH)