from torch import optim

from network import CNN


class Optimizer:
    def __init__(self, params: CNN.parameters, lr: float) -> None:
        self.params     = list(params)
        self.lr         = lr

        # Set defualt values.
        self.optimizer  = None

    def select(self, epoch: int) -> None:
        """select public function
        
        Select the optimizer to use based on epoch number.

        """

        schedule = [
            0.001,
            0.0009,
            0.0008,
            0.0005,
            0.0001
        ]

        lr = schedule[epoch]

        self.optimizer = optim.Adam(params=self.params, lr=lr)

    def step(self) -> None:
        # Make sure the optimizer is not None.
        assert self.optimizer is not None

        self.optimizer.step()

    def zero_grad(self) -> None:
        # Make sure the optimizer is not None.
        assert self.optimizer is not None

        self.optimizer.zero_grad()