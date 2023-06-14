from dogcat import training_data
from ResNet34 import ResNet
import torch
import torch.nn as nn
import torch.optim as optim


PATH = './res_net_dogcat.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


batch_size = 64
epoch = 1
trainloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size,
                                          shuffle=True, num_workers=8,pin_memory=True)
net = ResNet(2)
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.8)
for _ in range(epoch):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs data is a list of [inputs, labels]
        # inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimiz  e
        outputs = net(inputs)#forward
        loss = criterion(outputs, labels)#get loss
        loss.backward()#backward
        optimizer.step()#optimize

        # print statistics
        running_loss += loss.item()
        print(f'[{_ + 1} / {epoch},{i:4d} ] loss: {loss.item():.4f}')
    print(f'[{_ + 1} / {epoch},] loss: {running_loss/(i+1) :.4f}')
    torch.save(net.state_dict(), PATH)        

print('Finished Training')
torch.save(net.state_dict(), PATH)