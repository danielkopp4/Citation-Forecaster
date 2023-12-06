import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def train_neural_network(X_train, y_train, X_test, y_test, input_size, hidden_size1, hidden_size2, output_size, num_epochs=100, batch_size=32, learning_rate=0.001):
    """
    Trains and evaluates a neural network model using PyTorch.

    Args:
        X_train (numpy.ndarray): Input features for training.
        y_train (numpy.ndarray): Target values for training.
        X_test (numpy.ndarray): Input features for testing.
        y_test (numpy.ndarray): Target values for testing.
        input_size (int): Number of input features.
        hidden_size1 (int): Number of neurons in the first hidden layer.
        hidden_size2 (int): Number of neurons in the second hidden layer.
        output_size (int): Number of output neurons.
        num_epochs (int): Number of training epochs (default: 100).
        batch_size (int): Batch size for training and testing (default: 32).
        learning_rate (float): Learning rate for optimizer (default: 0.001).
    """
    # Convert the data to PyTorch tensors and create DataLoader objects
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Define the neural network model
    class Net(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size1)
            self.fc2 = nn.Linear(hidden_size1, hidden_size2)
            self.fc3 = nn.Linear(hidden_size2, output_size)

        def forward(self, x):
            x = nn.functional.relu(self.fc1(x))
            x = nn.functional.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Create an instance of the neural network model
    model = Net(input_size, hidden_size1, hidden_size2, output_size)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("Epoch %d, Loss: %.3f" % (epoch + 1, running_loss / len(train_loader)))

    # Evaluate the model on the test set
    with torch.no_grad():
        total_loss = 0.0
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
        average_test_loss = total_loss / len(test_loader)
        print("Test Loss: %.3f" % average_test_loss)

# Load the Boston Housing dataset and normalize the data
boston = load_boston()
scaler = StandardScaler()
X = scaler.fit_transform(boston.data)
y = boston.target.reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model parameters
input_size = X_train.shape[1]
hidden_size1 = 64
hidden_size2 = 32
output_size = 1

# Train and evaluate the neural network
train_neural_network(X_train, y_train, X_test, y_test, input_size, hidden_size1, hidden_size2, output_size)
