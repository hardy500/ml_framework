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

  def __add__(self, other: Tensor) -> Tensor:
    if (self.autograd and other.autograd):
      return Tensor(self.data + other.data,
                    autograd=True,
                    creators=[self, other],
                    creation_op="add")

    return Tensor(self.data + other.data)

  def __repr__(self) -> str:
    return str(self.data.__repr__())

  def __str__(self) -> str:
    return str(self.data.__str__())

a = Tensor([1,2,3,4,5], autograd=True)
b = Tensor([2,2,2,2,2], autograd=True)
c = Tensor([5,4,3,2,1], autograd=True)

d = a + b
e = b + c
f = d + e

f.backward(Tensor(np.array([1,1,1,1,1])))
print(b.grad.data == np.array([2,2,2,2,2]))