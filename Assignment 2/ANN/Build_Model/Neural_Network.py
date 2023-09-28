from torch import nn


class NeuralNetwork(nn.Module):
    # def __init__(self):
    #     super().__init__()
    #     # self.flatten = nn.Flatten()
    #     self.linear_relu_stack = nn.Sequential(
    #         nn.Linear(16, 256),
    #         # nn.Sigmoid(),
    #         # nn.Linear(4096, 2048),
    #         # nn.Sigmoid(),
    #         # nn.Linear(2048, 1024),
    #         # nn.Identity(),
    #         # nn.Linear(1024, 256),
    #         nn.Identity(),
    #         nn.Linear(256, 128),
    #         nn.Identity(),
    #         nn.Linear(128, 64),
    #         nn.Identity(),
    #         nn.Linear(64, 32),
    #         nn.Identity(),
    #         nn.Linear(32, 16),
    #         nn.Identity(),
    #         nn.Linear(16, 1),
    #         nn.Identity()
    #     )

    def __init__(self):
        super().__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 1028),
            nn.ReLU(),
            nn.Linear(1028, 1028),
            nn.ReLU(),
            nn.Linear(1028, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Identity(),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)
