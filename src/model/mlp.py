"""Script for training and testing a Multi-Layer Perceptron (MLP) model."""

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn
from torch import optim
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

# ----------------- set seed ------------------
torch.manual_seed(42)
np.random.seed(42)


def hidden_blocks(input_size, output_size, activation_function):
    """
    Function to create a hidden block in the neural network.

    Parameters:
    - input_size (int): Number of input features.
    - output_size (int): Number of output features.
    - activation_function (torch.nn.Module): Activation function to be applied.

    Returns:
    - torch.nn.Sequential: Sequential container for the hidden block.
    """
    return nn.Sequential(
        nn.Linear(input_size, output_size),
        activation_function,
    )


class MLP(nn.Module):
    """
    Multi-Layer Perceptron model.

    Parameters:
    - input_size (int): Number of input features.
    - hidden_units (int): Number of units in the hidden layers.
    - num_classes (int): Number of output classes.
    - activation_function (torch.nn.Module): Activation function to be applied.

    Attributes:
    - architecture (torch.nn.Sequential): Sequential container for the MLP architecture.
    """

    def __init__(self, input_size=75, hidden_units=512, num_classes=10,     
                 activation_function=nn.LeakyReLU()):
        super().__init__()
        self.architecture = nn.Sequential(
            hidden_blocks(input_size, hidden_units, activation_function),
            hidden_blocks(hidden_units, hidden_units, activation_function),
            hidden_blocks(hidden_units, hidden_units, activation_function),
            nn.Linear(hidden_units, num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the MLP model.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        return self.architecture(x)


def main():
    """Main function to train and test the MLP model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    le = LabelEncoder()
    data = pd.read_csv('dataset_v8.50.csv')

    X = data.drop(['video_name', 'video_frame', 'skill_id'], axis=1)
    y = data['skill_id']

    # encode the labels
    y = le.fit_transform(y)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # splitting and grouping by video_name
    train_idx, test_idx = next(gss.split(X, y, groups=data['video_name']))

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_test, y_test = X.iloc[test_idx], y[test_idx]

    # Set the hyperparameters
    input_size = len(data.columns) - 3  # exclude 'id_video', 'frame', 'skill_id'
    hidden_units = 512
    num_classes = len(data['skill_id'].unique())
    lr = 0.0001
    n_epochs = 500
    batch_size = 512
    model = MLP(input_size, hidden_units, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr, weight_decay=1e-4)

    train_dataset = TensorDataset(torch.FloatTensor(X_train.values).to(device),
                                   torch.LongTensor(y_train).to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    loss_list = []

    print("Now training the model...")
    for epoch in range(n_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_loader):

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

            if (i + 1) % int(batch_size/4) == 0:
                acc = 100 * correct / total
                # print the loss and the accuracy for batch_size = 512
                loss_list.append(running_loss / batch_size)
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] '
                      f'Loss: {running_loss / batch_size:.3f} Accuracy: {acc:.3f}%')

                running_loss = 0.0
                correct = 0
                total = 0

    # Test the model
    print("Now testing the model...")
    correct = 0
    total = 0

    test_dataset = TensorDataset(torch.FloatTensor(X_test.values).to(device),
                                  torch.LongTensor(y_test).to(device))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(test_loader):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss = loss.item()
            acc = 100 * correct / total

        print(f'Accuracy of the network on the test set: {acc:.3f}%')

    np.save('classes.npy', le.classes_)
    torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    main()
