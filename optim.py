from tensor import Tensor

class SGD:
  def __init__(self, params: list[Tensor], alpha: float=0.1):
    self.params = params
    self.alpha = alpha

  # NOTE: dont need it?
  #def zero(self):
  #  for p in self.params:
  #    p.grad.data *= 0

  def step(self, zero=True):
    for p in self.params:
      p.data -= p.grad.data * self.alpha
      if zero:
        p.grad.data *= 0