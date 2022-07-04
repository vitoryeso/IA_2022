import torch
import numpy as np
from sklearn.metrics import accuracy_score

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
        self.bn_1 = torch.nn.BatchNorm1d(8)
        self.layer_2 = torch.nn.Linear(8,64)
        self.bn_2 = torch.nn.BatchNorm1d(64)

        self.layer_3 = torch.nn.Linear(64,8)
        self.bn_3 = torch.nn.BatchNorm1d(8)
        self.layer_4 = torch.nn.Linear(8, output_shape)

    def forward(self, x):
        """
        Forward functional. Using to __call__ too
        """
        x = torch.tanh(self.layer_1(x))
        x = self.bn_1(x)
        x = torch.nn.functional.relu(self.layer_2(x))
        x = self.bn_2(x)
        x = torch.nn.functional.relu(self.layer_3(x))
        x = self.bn_3(x)

        x = self.layer_4(x)
        #return torch.nn.functional.softmax(x, dim=1)
        return x

    def infer(self, X):
        assert X.shape[1] == self.input_shape
        outs = []
        with torch.no_grad():
            for i in range(X.shape[0]):
                x = torch.Tensor(X[i]).unsqueeze(0)
                out = torch.nn.functional.softmax(self.forward(x))
                outs.append(out)
        return torch.vstack(outs)

    def fit(self, X, Yb, X_valid, Yb_valid, loss_f, opt, batch_size=4, epochs=5):
        assert X.shape[1] == self.input_shape
        epoch_train_loss = []
        epoch_valid_loss = []
        for epoch in range(epochs):
            train_loss = []
            self.train()
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

            self.eval()
            outs = []
            for i in range(X_valid.shape[0]):
                with torch.no_grad():
                    x = torch.Tensor(X_valid[i]).unsqueeze(0)
                    yb = torch.Tensor(Yb_valid[i]).unsqueeze(0)
                    out = self.forward(x)
                    outs.append(torch.nn.functional.softmax(out, dim=1))
                    loss = loss_f(out, yb)
                    valid_loss.append(loss.item())
            predictions = torch.vstack(outs).data.numpy()
            acc = accuracy_score(np.argmax(Yb_valid, axis=1),
                                 np.argmax(predictions, axis=1))
            
            mean_train_loss = torch.Tensor(train_loss).mean()
            mean_valid_loss = torch.Tensor(valid_loss).mean()
            print(f"EPOCH {epoch + 1}\n\tTRAIN LOSS: {mean_train_loss}\n\tVALID LOSS: {mean_valid_loss}\n\tVALID ACCURACY: {acc}\n")
            epoch_train_loss.append(mean_train_loss)
            epoch_valid_loss.append(mean_valid_loss)

        return epoch_train_loss, epoch_valid_loss
