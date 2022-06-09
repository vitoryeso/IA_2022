import torch

class MLP(torch.nn.Module):
    def __init__(self, input_shape, output_shape):
        """
        MLP constructor. Using a static architecture, but a flexible
        input/output shape.
        """
        super().__init__()

        self.layer_1 = torch.nn.Linear(input_shape, 16)
        self.layer_2 = torch.nn.Linear(16, 32)
        self.layer_3 = torch.nn.Linear(32, 8)
        self.layer_4 = torch.nn.Linear(8, output_shape)

    def forward(self, x):
        """
        Forward functional. Using to __call__ too
        """
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.relu(self.layer_2(x))
        x = torch.nn.functional.relu(self.layer_3(x))

        x = self.layer_4(x)
        return torch.nn.functional.softmax(x)


if __name__ == "__main__":
    mlp = MLP(8, 2)
    print(mlp)

