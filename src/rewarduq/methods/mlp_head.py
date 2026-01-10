import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """A simple multi-layer perceptron (MLP) reward head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: type[nn.Module],
        last_layer_bias: bool = False,
    ):
        """
        Args:
            input_dim (int):
                The dimension of the input features.

            hidden_dim (int):
                The dimension of the hidden layers.

            num_layers (int):
                The number of layers in the MLP.

            activation (type[nn.Module]):
                The activation function class to use (e.g., nn.ReLU, nn.Tanh).

            last_layer_bias (bool):
                Whether to include bias in the last layer. Defaults to False.
        """
        super().__init__()

        if num_layers == 1:
            self.mlp = nn.Linear(input_dim, 1, bias=last_layer_bias)
            return

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())
        layers.append(nn.Linear(hidden_dim, 1, bias=last_layer_bias))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            x (torch.Tensor of shape (batch_size, input_dim)):
                The input tensor.

        Returns:
            out (torch.Tensor of shape (batch_size, output_dim)):
                The output tensor.
        """
        return self.mlp(x)
