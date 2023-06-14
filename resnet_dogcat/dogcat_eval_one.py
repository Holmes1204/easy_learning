from dogcat import test_pic
from ResNet34 import ResNet
import matplotlib.pyplot as plt
from numpy import array
import torch


classes = ['cat','dog']
PATH = './res_net_dogcat_e40.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
net = ResNet(2)
net.to(device)
net.load_state_dict(torch.load(PATH))

with torch.no_grad():
    image,image_p = test_pic('./data/test/cats/1.jpg')
    a = torch.zeros([1,*list(image_p.size())])
    a[0]=image_p
    images = a.to(device)
    outputs = net(images)
    _, prediction = torch.max(outputs, 1)
plt.axis("off")
plt.title(classes[prediction])
plt.imshow(array(image))
plt.show()
print(classes[prediction])



