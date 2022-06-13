from src.MLP import MLP
from src.utils import generate_data, f1, f2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import torch


if __name__ == "__main__":
    """
    QUESTAO 1:
    """
    # Letra A)
    # features, labels = generate_data(1000, -4*np.pi, 4*np.pi, f1)

    # Letra B)
    features, labels = generate_data(1000, -1 , 1, f2)

    # Splitting data into a train-valid sets
    X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.3, random_state=88, shuffle=True)

    # Inint MLP model with input_size=2 and output_size=1
    mlp = MLP(2, 1)

    # Defining training parameters
    loss_f = torch.nn.MSELoss()
    opt = torch.optim.SGD(mlp.parameters(), lr=0.1)
    EPOCHS = 150

    # Fit MLP on data
    train_loss, valid_loss = mlp.fit(X_train, y_train, X_valid, y_valid, loss_f, opt, batch_size=8, epochs=EPOCHS)

    # ploting and testing
    plt.plot(range(EPOCHS), train_loss, c='r', label="Train Loss")
    plt.plot(range(EPOCHS), valid_loss, c='black', label="Valid Loss")
    plt.legend(loc="upper left")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()
    print("mlp(0, 0): ", mlp(torch.Tensor([0.0, 0.0])))
    print("f1(0, 0): ", f2(np.array([0.0, 0.0])))

