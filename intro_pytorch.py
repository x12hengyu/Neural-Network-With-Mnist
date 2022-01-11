import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_set = datasets.MNIST('./data', train=True, download=True,
                       transform=custom_transform)

    test_set = datasets.MNIST('./data', train=False,    
                       transform=custom_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 50)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = 50)

    if not training:
        return test_loader
    return train_loader

    


def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=28*28, out_features=128, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=64, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=10, bias=True)
    )
    return model




def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model = model.train()

    for epoch in range(T):
        correct = 0
        total = 0
        loss_total = 0
        running_loss = 0.0
        for inputs, labels in train_loader:
            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            opt.step()

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            loss_total += 1
            running_loss += loss.item()

        print('Train Epoch: {}   Accuracy: {}/{}({:.2f}%) Loss: {:.3f}'.format(epoch, correct, total, 100*correct/total, running_loss/loss_total))

    


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model = model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for data, labels in test_loader:
            # forward + backward + optimize
            output = model(data)
            loss = criterion(output, labels)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()
        
        if show_loss:
            print('Average loss: {:.4f}'.format(running_loss/total))
        print('Accuracy: {:.2f}%'.format(100*correct/total))
    


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1

    RETURNS:
        None
    """
    prob = []

    with torch.no_grad():
        logits = model(test_images[index])
    prob = F.softmax(logits, dim=1)
    
    prob = np.array(prob)[0]
    index_prob = prob.argsort()[::-1][:3]
    class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    for i in index_prob:
        print('{}: {:.2f}%'.format(class_names[i], 100*prob[i]))


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
    
    train_loader = get_data_loader()
    test_loader = get_data_loader(False)
    model = build_model()
    train_model(model, train_loader, criterion, T = 5)
    evaluate_model(model, test_loader, criterion, show_loss = True)
    test_images, _ = iter(test_loader).next()
    predict_label(model, test_images, 1)

