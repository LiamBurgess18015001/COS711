from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(14, 1024),
            nn.Sigmoid(),
            # nn.Linear(2048, 1028),
            # nn.PReLU(),
            # nn.Linear(1028, 512),
            # nn.PReLU(),
            nn.Linear(1024, 512),
            nn.Sigmoid(),
            nn.Linear(512, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, 32),
            nn.Sigmoid(),
            nn.Linear(32, 16),
            nn.Sigmoid(),
            nn.Linear(16, 1),
            nn.Identity(),
        )

    # def __init__(self):
    #     super().__init__()
    #     self.linear_relu_stack = nn.Sequential(
    #         nn.Linear(15, 1),
    #         nn.Identity(),
    #         nn.Linear(1, 1)
    #     )
    #
    def forward(self, x):
        return self.linear_relu_stack(x)
