from src.MLP import MLP
from src.utils import generate_data, f1, f2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import torch

def classe_1(n_samples, theta_min, theta_max):
    thetas = np.random.uniform(low=theta_min, high=theta_max, size=(n_samples,))
    f_x = lambda theta: (theta/4) * np.cos(theta)
    f_y = lambda theta: (theta/4) * np.sin(theta)
    x = [f_x(theta) for theta in thetas]
    y = [f_y(theta) for theta in thetas]
    sin_x = [np.sin(x_) for x_ in x]
    sin_y = [np.sin(y_) for y_ in y]

    return np.vstack(zip(x, y, sin_x, sin_y)), np.array([(1, 0)] * n_samples)


def classe_2(n_samples, theta_min, theta_max):
    thetas = np.random.uniform(low=theta_min, high=theta_max, size=(n_samples,))
    f_x = lambda theta: (theta/4 + 0.8) * np.cos(theta)
    f_y = lambda theta: (theta/4 + 0.8) * np.sin(theta)
    x = [f_x(theta) for theta in thetas]
    y = [f_y(theta) for theta in thetas]
    sin_x = [np.sin(x_) for x_ in x]
    sin_y = [np.sin(y_) for y_ in y]

    return np.vstack(zip(x, y, sin_x, sin_y)), np.array([(0, 1)] * n_samples)


if __name__ == "__main__":
    """
    QUESTAO 2:
    
    Classificar dois padroes de pontos no espaco R2

    input_data: 
        x: (-10, 10)
        y: (-10, 10)

    classe 1:
        (x, y) = ( (theta/4) * cos(theta), (theta/4) * sin(theta) )

    classe 2:
        (x, y) = ( ((theta/4) + 0.8) * cos(theta), ((theta/4) + 0.8) * sin(theta) )
    """
    # Letra A)
    # features, labels = generate_data(1000, -4*np.pi, 4*np.pi, f1)
    data_1, labels_1 = classe_1(5000, 0, 20)
    data_2, labels_2 = classe_2(5000, 0, 20)
    data = np.vstack((data_1, data_2))
    labels = np.vstack((labels_1, labels_2))

    x_train, x_valid, y_train, y_valid = train_test_split(data, labels, test_size=0.3, random_state=42, shuffle=True)

    # Inint MLP model with input_size=2 and output_size=1
    mlp = MLP(4, 2)

    # Defining training parameters
    loss_f = torch.nn.CrossEntropyLoss()
    opt = torch.optim.SGD(mlp.parameters(), lr=0.005)
    EPOCHS = 150

    # Fit MLP on data
    train_loss, valid_loss = mlp.fit(x_train, y_train, x_valid, y_valid, loss_f, opt, batch_size=16, epochs=EPOCHS)

    # ploting and testing
    plt.plot(range(EPOCHS), train_loss, c='r', label="Train Loss")
    plt.plot(range(EPOCHS), valid_loss, c='black', label="Valid Loss")
    plt.legend(loc="upper left")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()