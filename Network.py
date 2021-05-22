import inline as inline
import matplotlib as matplotlib
import torch
from torch import nn
from torch import optim
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
from matplotlib import pyplot as plt


def view_classify(img, ps):
    ps = ps.data.numpy().squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()


def setup_model():
    # Model with 784(depending on size of dataset) input neurons,
    # 2 hidden layers(128 neurons and 64 neurons)
    # and an output layer with 10 neurons
    # each one indicating the probability of a number 0 - 9
    model_s = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),  # ReLu as linear activation function
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
        nn.LogSoftmax(dim=1))
    # LogSoftmax as final activation to wrap the probabilities so they add up to 1
    # to get a nice probability distribution
    return model_s


def train_model(model_t, epochs_t, trainloader_t):
    for e in range(epochs_t):
        loss_per_epoch = 0
        for images, labels in trainloader_t:
            # flatten to 1D vector to pass to network
            images = images.view(images.shape[0], -1)

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
    img = images[1].view(1, 784)
    with torch.no_grad():
        log_pred = model(img)
    pred = torch.exp(log_pred)
    view_classify(img.view(1, 28, 28), pred)


if __name__ == '__main__':
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    # Download and load the training data
    train_set = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    model = setup_model()

    # negative log likelihood loss
    criterion = nn.NLLLoss()

    # statistic gradient descent
    # learning rate of 0.005
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # 5 training epochs
    epochs = 2

    model = train_model(model, epochs, train_loader)

    predict(model, train_loader)
