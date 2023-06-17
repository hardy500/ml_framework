from __future__ import annotations
import numpy as np

class Tensor:
  def __init__(self,
               data: list,
               autograd: bool=False,
               creators: list[Tensor, Tensor] | None=None,
               creation_op: str | None=None,
               id: int | None=None):

    self.data = np.array(data)
    self.creators = creators
    self.creation_op = creation_op
    self.grad = None

    if (id is None):
      self.id = np.random.randint(0, 100_000)
    else:
      self.id = id

    self.autograd = autograd
    self.children = {}

    if creators:
      for creator in creators:
        # keeps track of number of children
        if (self.id not in creator.children):
          creator.children[self.id] = 1
        else:
          creator.children[self.id] += 1

  def all_children_grads_accounted_for(self) -> bool:
    # check whether a tensor has received the correct number
    # of gradients from each child
    for _, count in self.children.items():
      if (count != 0):
        return False
    return True

  def backward(self,
               grad: Tensor | None=None,
               grad_origin: Tensor | None=None):

    if (self.autograd):
      if (grad_origin):
        if (self.children[grad_origin.id] == 0):
          raise Exception("cannot backprop more than once")
        else:
          self.children[grad_origin.id] -= 1

      if (self.grad is None):
        self.grad = grad
      else:
        self.grad += grad

      # grad dont have grads of their own
      assert grad.autograd == False

      if (self.creators
          and (self.all_children_grads_accounted_for()
                or grad_origin is None)):

        # begin actual backprop
        if (self.creation_op == "add"):
          self.creators[0].backward(grad, self)
          self.creators[1].backward(grad, self)

        if (self.creation_op == "sub"):
          self.creators[0].backward(Tensor(self.grad.data), self)
          self.creators[1].backward(Tensor(self.grad.__neg__().data), self)

        if (self.creation_op == "mul"):
          new = self.grad * self.creators[1]
          self.creators[0].backward(new, self)
          new = self.grad * self.creators[0]
          self.creators[1].backward(new, self)

        # NOTE: Where does this come from?
        if (self.creation_op == "matmul"):
          c0 = self.creators[0] # usually an activation
          c1 = self.creators[1] # usually a weight matrix
          new = self.grad.matmul(c1.transpose())
          c0.backward(new)
          new = self.grad.transpose().matmul(c0).transpose()
          c1.backward(new)

        if (self.creation_op == "transpose"):
          self.creators[0].backward(self.grad.transpose())

        if ("sum" in self.creation_op):
          dim = int(self.creation_op.split("_")[1])
          self.creators[0].backward(self.grad.sum(dim))

        if (self.creation_op == "neg"):
          self.creators[0].backward(self.grad.__neg__())

        if ("expand" in self.creation_op):
          dim = int(self.creation_op.split("_")[1])
          self.creators[0].backward(self.grad.sum(dim))

  def sum(self, dim: int=0) -> Tensor:
    if (self.autograd):
      return Tensor(np.sum(self.data, dim),
                    autograd=True,
                    creators=[self],
                    creation_op="sum_"+str(dim))
    return Tensor(np.sum(self.data, dim))

  def transpose(self) -> Tensor:
    if (self.autograd):
      return Tensor(self.data.T,
                    autograd=True,
                    creators=[self],
                    creation_op="transpose")
    return Tensor(self.data.T)

  def matmul(self, other: Tensor) -> Tensor:
    if (self.autograd):
      return Tensor(np.dot(self.data, other.data),
                    autograd=True,
                    creators=[self, other],
                    creation_op="matmul")
    return Tensor(np.dot(self.data, other.data))

  def expand(self, dim: int, copies: int) -> Tensor:
    trans = list(range(len(self.data.shape)))
    trans.insert(dim, len(self.data.shape))
    new_shape = list(self.data.shape) + [copies]
    new_data = self.data.repeat(copies).reshape(new_shape)
    new_data = new_data.transpose(trans)

    if (self.autograd):
      return Tensor(new_data,
                    autograd=True,
                    creators=[self],
                    creation_op="expand_" + str(dim))
    return Tensor(new_data)

  def __add__(self, other: Tensor) -> Tensor:
    if (self.autograd and other.autograd):
      return Tensor(self.data + other.data,
                    autograd=True,
                    creators=[self, other],
                    creation_op="add")

    return Tensor(self.data + other.data)

  def __sub__(self, other: Tensor) -> Tensor:
    if (self.autograd and other.autograd):
      return Tensor(self.data - other.data,
                    autograd=True,
                    creators=[self, other],
                    creation_op="sub")

    return Tensor(self.data - other.data)

  def __mul__(self, other: Tensor) -> Tensor:
    if (self.autograd and other.autograd):
      return Tensor(self.data * other.data,
                    autograd=True,
                    creators=[self, other],
                    creation_op="mul")

    return Tensor(self.data * other.data)

  def __neg__(self) -> Tensor:
    if (self.autograd):
      return Tensor(self.data * (-1),
                    autograd=True,
                    creators=[self],
                    creation_op="neg")

    return Tensor(self.data * (-1))

  def __repr__(self) -> str:
    return str(self.data.__repr__())

  def __str__(self) -> str:
    return str(self.data.__str__())

np.random.seed(0)

data = Tensor(np.array([[0,0],[0,1],[1,0],[1,1]]), autograd=True) # (4, 2)
target = Tensor(np.array([[0],[1],[0],[1]]), autograd=True)       # (4, 1)

w = list()
w.append(Tensor(np.random.rand(2,3), autograd=True))
w.append(Tensor(np.random.rand(3,1), autograd=True))

for i in range(10):

    # Predict
    pred = data.matmul(w[0]).matmul(w[1])

    # Compare
    loss = ((pred - target)*(pred - target)).sum(0)

    # Learn
    loss.backward(Tensor(np.ones_like(loss.data)))

    for w_ in w:
        w_.data -= w_.grad.data * 0.1
        w_.grad.data *= 0

    print(loss)