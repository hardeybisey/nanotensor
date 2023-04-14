import numpy as np


class Tensor:
    """
    A class representing a tensor with a single numerical value.

    Parameters:
    -----------
    data : int or float
        The numerical value of the tensor.
    _downstream : tuple of Tensors, optional
        Tensors that depend on this tensor. Defaults to an empty tuple.
    _op : str, optional
        The operation performed on this tensor. Defaults to an empty string.
    dtype : type, optional
        The data type of the tensor. Defaults to None.
    grad : float
        The gradient of the tensor with respect to the loss.
    """
    def __init__(self, data, _downstream=tuple(), _op='', dtype=None):
        """
        Initialize a Tensor object.
        Parameters:
            data (int or float): The numerical value of the tensor.
            _downstream (tuple, optional): A tuple of downstream tensors.
            _op (str, optional): The operation that produced this tensor.
            dtype (type, optional): The desired data type of the tensor.
        Raises:
            AssertionError: If `data` is not of type `int` or `float`.
        """
        assert isinstance(data, (int, float)), "data must be of type array, int or float"
        self.data = float(data) if not dtype else dtype(data)
        self._downstream = _downstream
        self._backward = lambda: None
        self._op = _op
        self.grad = 0.0

    def __add__(self, other):
        """
        Return the sum of this tensor and another tensor or numerical value.
        Parameters:
            other (Tensor or int or float): The other tensor or numerical value to add.
        Returns:
            Tensor: A new tensor representing the sum of this tensor and `other`.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(data=(self.data + other.data), _downstream=(self, other), _op='+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        """
        Return the product of this tensor and another tensor or numerical value.
        Parameters:
            other (Tensor or int or float): The other tensor or numerical value to multiply.

        Returns:
            Tensor: A new tensor representing the product of this tensor and `other`.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(data=(self.data * other.data), _downstream=(self, other), _op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        """
        Return this tensor raised to the power of another numerical value.

        Parameters:
            other (int or float): The numerical value to raise this tensor to the power of.

        Returns:
            Tensor: A new tensor representing this tensor raised to the power of `other`.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(data=(self.data ** other.data), _downstream=(self,), _op='pow')

        def _backward():
            self.grad += other.data * (self.data ** (other.data - 1)) * out.grad
            other.grad += self.data * (other.data ** (self.data - 1)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        """
        Returns:
            Tensor: A new tensor representing the ReLU of this tensor  .
        """
        out = Tensor(data=(0 if self.data < 0 else self.data), _downstream=(self,), _op='relu')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        """
        Returns:
            Tensor: A new tensor representing the hyperbolic tangent of this tensor.
        """
        t = (np.exp(2 * self.data) - 1) / (np.exp(2 * self.data) + 1)
        out = Tensor(data=t, _downstream=(self,), _op='tanh')

        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        """
        Returns:
            Tensor: A new tensor representing the sigmoid function of this tensor.
        """
        s = 1 / (1 + np.exp(-self.data))
        out = Tensor(data=s, _downstream=(self,), _op='sigmoid')

        def _backward():
            self.grad = out.data * (1 - out.data)

        out._backward = _backward
        return out

    def exp(self):
        """
        Returns:
            Tensor: A new tensor representing the exponential function of this tensor.
        """
        x = np.exp(self.data)
        out = Tensor(data=x, _downstream=(self,), _op='exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        """
        Returns:
            Tensor: A new tensor representing the log of this tensor.
        """
        x = np.log(self.data)
        out = Tensor(data=x, _downstream=(self,), _op='log')
        def _backward():
            self.grad += out.data *  1/self.data
        out._backward = _backward
        return out

    def backward(self):
        """
        Returns:
            calculated the gradients of all downstream element of a given tensor.
        """
        nodes = self.__allnodes()
        self.grad = 1
        for node in nodes:
            node._backward()

    def __allnodes(self):
        """
        Returns:
            list: A list of downstream elements that produced the current tensor 
             in a chronological order.
        """
        nodes = []
        def get_downstream(object, visited=set()):
            if object not in visited:
                visited.add(object)
                nodes.append(object)
                for children in object._downstream:
                    get_downstream(children)
        get_downstream(self)
        return nodes

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return self + (-other)

    def __rtruediv__(self, other):
        return self * (other ** -1)

    def __repr__(self):
        """
        Return a string representation of the Tensor object.
        """
        return f"Tensor(data= ({str(self.data)}), grad=({str(self.grad)}))"
