import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device=torch.device("cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')





# functions to show an image



def imshow(img,fig):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    fig.imshow(np.transpose(npimg, (1, 2, 0)))
#    fig.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
#imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


########################################################################
# 2. Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3,5)
        self.pool = nn.MaxPool2d(2, 2)
        self.pool_ = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 16, 5)
        
        self.conv1_ = nn.Conv2d(3, 3,5)
#        self.pool = nn.MaxPool2d(2, 2)
        self.conv2_ = nn.Conv2d(3, 16, 5)
        
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.x_=1
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))#32 #28 14
#        x = self.pool(F.relu(self.conv2(x)))#10 5
        
        y = self.pool_(F.relu(self.conv1_(x)))
        y = self.pool(F.relu(self.conv2_(y)))
        
        x = y.view(-1, 16 * 5 * 5)
#        y = y.view(-1, 8 * 5 * 5)
#        self.x_=torch.cat((x,y),1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#
#class Net(nn.Module):
#    def __init__(self):
#        super(Net, self).__init__()
#        self.conv1 = nn.Conv2d(3, 6,5)
#        self.pool = nn.MaxPool2d(2, 2)
#        self.conv2 = nn.Conv2d(6, 16, 5)
#        self.fc1 = nn.Linear(16 * 5 * 5, 120)
#        self.fc2 = nn.Linear(120, 84)
#        self.fc3 = nn.Linear(84, 10)
#        self.x_=1;
#
#    def forward(self, x):
#        x = self.pool(F.relu(self.conv1(x)))#32 #28 14
#        x = self.pool(F.relu(self.conv2(x)))#10 5
#        self.x_ = x.view(-1, 16 * 5 * 5)
#        x = F.relu(self.fc1(self.x_))
#        x = F.relu(self.fc2(x))
#        x = self.fc3(x)
#        return x



net = Net()
#net.to(device)
########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.

for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]


#        inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

########################################################################
# Let's quickly save our trained model:

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

########################################################################
# See `here <https://pytorch.org/docs/stable/notes/serialization.html>`_
# for more details on saving PyTorch models.
#
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display an image from the test set to get familiar.

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
#imshow(torchvision.utils.make_grid(images))
#imshow(images[0])
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

########################################################################
# Next, let's load back in our saved model (note: saving and re-loading the model
# wasn't necessary here, we only did it to illustrate how to do so):

net = Net()
net.to(device)
#
#
#
net.load_state_dict(torch.load(PATH))
#
#########################################################################
## Okay, now let us see what the neural network thinks these examples above are:
#
#outputs = net(images)
#
#########################################################################
## The outputs are energies for the 10 classes.
## The higher the energy for a class, the more the network
## thinks that the image is of the particular class.
## So, let's get the index of the highest energy:
#_, predicted = torch.max(outputs, 1)
#
#print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                              for j in range(4)))
#
#########################################################################
## The results seem pretty good.
##
## Let us look at how the network performs on the whole dataset.
#
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
#
#########################################################################
## That looks way better than chance, which is 10% accuracy (randomly picking
## a class out of 10 classes).
## Seems like the network learnt something.
##
## Hmmm, what are the classes that performed well, and the classes that did
## not perform well:
#
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
#
#
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
#
#
#
#
#
## Assuming that we are on a CUDA machine, this should print a CUDA device:
#
#print(device)
#
#
#

fig1 = plt.figure(1)
ax1 = fig1.gca()
fig2 = plt.figure(2)
ax2 = fig2.gca()
with torch.no_grad():
    conv=nn.Conv2d(3,3,2)
    conv.weight[:]=1/4.;
    imshow(data[0][3],ax1)
    imshow(conv(data[0])[3],ax2)
    
#    conv=nn.Conv2d(1,3,5)
#    conv.weight[:]=1/9.
#    imshow(data[0][0],ax1)
#    imshow(conv(data[0][:1,2:3])[0],ax2)
    

#imshow()

