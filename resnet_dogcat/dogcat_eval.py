from dogcat import test_data
from ResNet34 import ResNet
import torch


classes = ['cats','dogs']
PATH = './res_net_dogcat.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


batch_size = 64
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          shuffle=True, num_workers=8,pin_memory=True)
net = ResNet(2)
net.to(device)

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
correct = 0 
total = 0
net.load_state_dict(torch.load(PATH))
with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        # images, labels = data
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1
        print(f"testing{i:3d}")

print(f'Accuracy of the network on the  test images: {100 * correct // total} %')
# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
