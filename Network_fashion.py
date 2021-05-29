
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms


def view_classify(img, ps, version="Fashion"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    img = img.cpu()
    ps = ps.cpu()
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(['T-shirt/top',
                            'Trouser',
                            'Pullover',
                            'Dress',
                            'Coat',
                            'Sandal',
                            'Shirt',
                            'Sneaker',
                            'Bag',
                            'Ankle Boot'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.show()


def setup_model():
    # Model with 784(depending on size of dataset) input neurons,
    # 3 hidden layers(128 neurons, 64 neurons and 32 neurons)
    # and an output layer with 10 neurons
    # each one indicating the probability of a clothing
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

            self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

            self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
            self.drop = nn.Dropout2d(0.25)
            self.fc2 = nn.Linear(in_features=600, out_features=120)
            self.fc3 = nn.Linear(in_features=120, out_features=10)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = self.drop(out)
            out = self.fc2(out)
            out = self.fc3(out)
            return out

    # LogSoftmax as final activation to wrap the probabilities so they add up to 1
    # to get a nice probability distribution
    return Net()


def train_model(model_t, epochs_t, trainloader_t):
    for e in range(epochs_t):
        loss_per_epoch = 0
        for images, labels in trainloader_t:
            # flatten to 1D vector to pass to network
            images, labels = images.cuda(), labels.cuda()
            # images = images.view(images.shape[0], -1)

            optimizer.zero_grad()
            y = model_t(images)
            loss = criterion(y, labels)
            loss.backward()
            optimizer.step()

            loss_per_epoch += loss.item()
        else:
            print("epoch " + str(e) + f" loss: {loss_per_epoch / len(trainloader_t)}")
    return model_t


def predict(model_p, trainloader_p):
    data_iter = iter(trainloader_p)
    images, labels = next(data_iter)
    images, labels = images.cuda(), labels.cuda()
    for i in range(15):
        # img = images[i].view(1, 784)
        img = Variable(images[i])
        with torch.no_grad():
            log_pred = model_p(img)
        pred = torch.exp(log_pred)
        view_classify(img.view(1, 28, 28), pred)


if __name__ == '__main__':
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    # Download and load the training data
    train_set = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    # Download and load the test data
    test_set = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

    model = setup_model().cuda()

    # negative log likelihood loss
    criterion = nn.CrossEntropyLoss()

    # statistic gradient descent
    # learning rate of 0.005
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # 5 training epochs
    epochs = 2

    model = train_model(model, epochs, train_loader)

    # torch.save(model, "model_fashion.pt")

    predict(model, test_loader)
