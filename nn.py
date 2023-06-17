from tensor import Tensor
import numpy as np

class Layer:
  def __init__(self):
    self.params = []

  def get_param(self):
    return self.params

class Linear(Layer):
  def __init__(self, in_features: int, out_features: int):
    super().__init__()
    w = np.random.randn(in_features, out_features) * np.sqrt(2./in_features)
    self.weight = Tensor(w, autograd=True)
    self.bias = Tensor(np.zeros(out_features), autograd=True)

  def forward(self, x: Tensor) -> Tensor:
    return x.matmul(self.weight) + self.bias.expand(0, len(x.data))


if __name__ == "__main__":
  from torch import nn
  x = Tensor(np.array([[0,0],[0,1],[1,0],[1,1]]), autograd=True)
  y = Tensor(np.array([[0],[1],[0],[1]]), autograd=True)

  model = Linear(2, 3)
  pred = model.forward(x)
  nn.Linear(2, 3)
  print(pred)



