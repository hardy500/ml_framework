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
    return input.matmul(self.weight) + self.bias.expand(0, len(input.data))

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

class MSELoss(Layer):
  def __init__(self):
    super().__init__()

  def __call__(self, pred: Tensor, y: Tensor) -> Tensor:
    return ((pred - y) * (pred - y)).sum(0)

class CrossEntropyLoss:
  def __init__(self):
    super().__init__()

  def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
    return pred.cross_entropy(target)

class Sigmoid(Layer):
  def __init__(self):
    super().__init__()

  def forward(self, input: Tensor) -> Tensor:
    return input.sigmoid()

class Tanh(Layer):
  def __init__(self):
    super().__init__()

  def forward(self, input: Tensor) -> Tensor:
    return input.tanh()

class Embedding(Layer):
  def __init__(self, vocab_size: int, dim: int):
    super().__init__()

    self.vocab_size = vocab_size
    self.dim = dim

    weight = (np.random.rand(vocab_size, dim) - 0.5)/dim
    self.weight = Tensor(weight, autograd=True)
    self.params.append(self.weight)

  def forward(self, input: Tensor) -> Tensor:
    return self.weight.index_select(input)

if __name__ == "__main__":
  from optim import SGD
  np.random.seed(0)

  # data indices
  data = Tensor(np.array([1,2,1,2]), autograd=True)

  # target indices
  target = Tensor(np.array([0,1,0,1]), autograd=True)

  model = Sequential([Embedding(3,3), Tanh(), Linear(3,4)])
  criterion = CrossEntropyLoss()

  optim = SGD(parameters=model.parameters(), alpha=0.1)

  for i in range(10):

      # Predict
      pred = model.forward(data)

      # Compare
      loss = criterion(pred, target)

      # Learn
      loss.backward(Tensor(np.ones_like(loss.data)))
      optim.step()
      print(loss)