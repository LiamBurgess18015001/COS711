from torch import nn


def build_model(model_type: str, activation, size):
    if activation == "sigmoid":
        _activation = nn.Sigmoid
    elif activation == "tanh":
        _activation = nn.Tanh
    else:
        _activation = nn.PReLU

    if model_type == "pyramid":
        if size == "large":
            return nn.Sequential(
                nn.Linear(16, 2048),
                _activation(),
                nn.Linear(2048, 1028),
                _activation(),
                nn.Linear(1028, 512),
                _activation(),
                nn.Linear(512, 256),
                _activation(),
                nn.Linear(256, 128),
                _activation(),
                nn.Linear(128, 32),
                _activation(),
                nn.Linear(32, 16),
                _activation(),
                nn.Linear(16, 1),
                nn.Identity()
            )
        else:
            return nn.Sequential(
                nn.Linear(16, 512),
                _activation(),
                nn.Linear(512, 256),
                _activation(),
                nn.Linear(256, 128),
                _activation(),
                nn.Linear(128, 32),
                _activation(),
                nn.Linear(32, 16),
                _activation(),
                nn.Linear(16, 1),
                nn.Identity()
            )

    elif model_type == "perceptron":
        return nn.Sequential(
            nn.Linear(16, 1),
            nn.Identity()
        )

    elif model_type == "flat-1024":
        if size == "large":
            return nn.Sequential(
                nn.Linear(16, 1024),
                _activation(),
                nn.Linear(1024, 1024),
                _activation(),
                nn.Linear(1024, 1024),
                _activation(),
                nn.Linear(1024, 1024),
                _activation(),
                nn.Linear(1024, 1024),
                _activation(),
                nn.Linear(1024, 1024),
                _activation(),
                nn.Linear(1024, 1024),
                _activation(),
                nn.Linear(1024, 1024),
                _activation(),
                nn.Linear(1024, 1),
                nn.Identity()
            )

        elif size == "medium":
            return nn.Sequential(
                nn.Linear(16, 1024),
                _activation(),
                nn.Linear(1024, 1024),
                _activation(),
                nn.Linear(1024, 1024),
                _activation(),
                nn.Linear(1024, 1024),
                _activation(),
                nn.Linear(1024, 1),
                nn.Identity()
            )

        elif size == "small":
            return nn.Sequential(
                nn.Linear(16, 1024),
                _activation(),
                nn.Linear(1024, 1024),
                _activation(),
                nn.Linear(1024, 1),
                nn.Identity()
            )

    elif model_type == "flat-256":
        if size == "large":
            return nn.Sequential(
                nn.Linear(16, 256),
                _activation(),
                nn.Linear(256, 256),
                _activation(),
                nn.Linear(256, 256),
                _activation(),
                nn.Linear(256, 256),
                _activation(),
                nn.Linear(256, 256),
                _activation(),
                nn.Linear(256, 256),
                _activation(),
                nn.Linear(256, 256),
                _activation(),
                nn.Linear(256, 256),
                _activation(),
                nn.Linear(256, 1),
                nn.Identity()
            )

        elif size == "medium":
            return nn.Sequential(
                nn.Linear(16, 256),
                _activation(),
                nn.Linear(256, 256),
                _activation(),
                nn.Linear(256, 256),
                _activation(),
                nn.Linear(256, 256),
                _activation(),
                nn.Linear(256, 1),
                nn.Identity()
            )

        elif size == "small":
            return nn.Sequential(
                nn.Linear(16, 256),
                _activation(),
                nn.Linear(256, 256),
                _activation(),
                nn.Linear(256, 1),
                nn.Identity()
            )

    raise Exception("bad config")


class NeuralNetwork(nn.Module):
    def __init__(self, model):
        super().__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = model

    def forward(self, x):
        return self.linear_relu_stack(x)
