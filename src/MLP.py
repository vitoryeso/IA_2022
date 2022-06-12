import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class MLP(torch.nn.Module):
    def __init__(self, input_shape, output_shape):
        """
        MLP constructor. Using a static architecture, but a flexible
        input/output shape.
        """
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layer_1 = torch.nn.Linear(input_shape, 8)
        self.layer_2 = torch.nn.Linear(8, 16)
        self.layer_3 = torch.nn.Linear(16, 8)
        self.layer_4 = torch.nn.Linear(8, output_shape)

    def forward(self, x):
        """
        Forward functional. Using to __call__ too
        """
        x = torch.nn.functional.tanh(self.layer_1(x))
        x = torch.nn.functional.tanh(self.layer_2(x))
        x = torch.nn.functional.tanh(self.layer_3(x))

        x = self.layer_4(x)
        #return torch.nn.functional.softmax(x, dim=1)
        return x

    def fit(self, X, Yb, X_valid, Yb_valid, loss_f, opt, batch_size=4, epochs=5):
        assert X.shape[1] == self.input_shape
        epoch_train_loss = []
        epoch_valid_loss = []
        for epoch in range(epochs):
            train_loss = []
            for i in range(X.shape[0] // batch_size):

                opt.zero_grad()
                x = torch.Tensor(X[i:i+batch_size])
                yb = torch.Tensor(Yb[i:i+batch_size])
                out = self.forward(x)
                loss = loss_f(out, yb)
                if i == 0 and epoch == 0:
                    print("first loss: ", loss.item())

                loss.backward()
                opt.step()
                train_loss.append(loss.item())

            valid_loss = []
            for i in range(X_valid.shape[0] // batch_size):
                with torch.no_grad():
                    x = torch.Tensor(X_valid[i: i+batch_size])
                    yb = torch.Tensor(Yb_valid[i: i+batch_size])
                    out = self.forward(x)
                    loss = loss_f(out, yb)
                    valid_loss.append(loss.item())
            mean_train_loss = torch.Tensor(train_loss).mean()
            mean_valid_loss = torch.Tensor(valid_loss).mean()
            print(f"EPOCH {epoch}\n\tTRAIN LOSS: {mean_train_loss}\n\tVALID LOSS: {mean_valid_loss}\n")
            epoch_train_loss.append(mean_train_loss)
            epoch_valid_loss.append(mean_valid_loss)

        return epoch_train_loss, epoch_valid_loss


def sinc(x):
    if x == 0.0: return 1; 
    else:
        return np.sin(np.pi*x)/(np.pi*x)

def f1(X):
    return sinc(X[0]) * sinc(X[1])

def generate_data(n, low, high):
    samples = []
    labels = []
    for i in range(n):
        if i < 100:
            samples.append(np.array([0.0, 0.0]))
            labels.append(f1(samples[-1]))
        else:
            samples.append(np.random.uniform(low, high, (2,)))
            labels.append(f1(samples[-1]))

    X = np.vstack(samples)
    y = np.vstack(labels)
    return X, y

if __name__ == "__main__":
    mlp = MLP(2, 1)
    loss_f = torch.nn.MSELoss()
    opt = torch.optim.SGD(mlp.parameters(), lr=0.4)

    features, labels = generate_data(1000, -4*np.pi, 4*np.pi)
    X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.3, random_state=88, shuffle=True)

    print("features shape: ", features.shape)
    print("labels shape: ", labels.shape)
    print(mlp)

    EPOCHS = 150
    train_loss, valid_loss = mlp.fit(X_train, y_train, X_valid, y_valid, loss_f, opt, batch_size=8, epochs=EPOCHS)
    plt.plot(range(EPOCHS), train_loss, c='r', label="Train Loss")
    plt.plot(range(EPOCHS), valid_loss, c='black', label="Valid Loss")
    plt.legend(loc="upper left")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()
    print("mlp(0, 0): ", mlp(torch.Tensor([0.0, 0.0])))
    print("f1(0, 0): ", f1(np.array([0.0, 0.0])))



