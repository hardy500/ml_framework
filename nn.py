from tensor import Tensor
import numpy as np

class Layer:
  def __init__(self):
    self.params = []

  def parameters(self):
    return self.params

class Linear(Layer):
  def __init__(self, in_features: int, out_features: int):
    super().__init__()
    w = np.random.randn(in_features, out_features) * np.sqrt(2./in_features)
    self.weight = Tensor(w, autograd=True)
    self.bias = Tensor(np.zeros(out_features), autograd=True)

    self.params.append(self.weight)
    self.params.append(self.bias)

  def forward(self, input: Tensor) -> Tensor:
    return input.matmul(self.weight) + self.bias.expand(0, len(x.data))

class Sequential(Layer):
  def __init__(self, layers: list=[]):
    super().__init__()
    self.layers = layers

  def add(self, layer: Layer):
    self.layers.append(layer)

  def forward(self, input: Tensor):
    for layer in self.layers:
      input = layer.forward(input)
    return input

  def parameters(self):
    parameters = []
    for layer in self.layers:
      parameters += layer.parameters()
    return parameters

if __name__ == "__main__":
  from optim import SGD

  np.random.seed(0)

  x = Tensor(np.array([[0,0],[0,1],[1,0],[1,1]]), autograd=True)
  y = Tensor(np.array([[0],[1],[0],[1]]), autograd=True)

  model = Sequential([
    Linear(2,3),
    Linear(3,1)
  ])

  optim = SGD(parameters=model.parameters(), alpha=0.05)

  for i in range(10):
    pred = model.forward(x)
    loss = ((pred - y) * (pred - y)).sum(0)
    loss.backward(Tensor(np.ones_like(loss.data)))
    optim.step()
    print(loss)




