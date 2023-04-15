import numpy as np


class Tensor:
    
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
        x = self.data
        t = (np.exp(2*x) - 1) / (np.exp(2 * x) + 1)
        out = Tensor(data=t, _downstream=(self,), _op='tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        """
        Returns:
            Tensor: A new tensor representing the exponential function of this tensor.
        """
        x = self.data
        out = Tensor(data= np.exp(x), _downstream=(self,), _op='exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        """
        Returns:
            Tensor: A new tensor representing the sigmoid function of this tensor.
        """
        x = self.data
        s = 1 / (1 + np.exp(-x))
        out = Tensor(data=s, _downstream=(self,), _op='sigmoid')
        
        def _backward():
            self.grad = out.data * (1 - out.data)

        out._backward = _backward
        return out
    
    def log(self, eps=1e-15):
        """
        Returns:
            ## Only used for the log loss function
            Tensor: A new tensor representing the log of this tensor.
        """
        # x = Tensor(eps) if self < 0 else self - eps if self == 1 else self
        x = self.data
        out = Tensor(data= np.log(x), _downstream=(self,), _op='log')
        
        def _backward():
            self.grad += out.data *  1/self.data
        out._backward = _backward
        return out

    def backward(self):
        nodes = []
        def get_downstream(object, visited=set()):
            if str(object) not in visited:
                visited.add(str(object))
                nodes.append(object)
                for children in object._downstream:
                    get_downstream(children)
        get_downstream(self)
        self.grad = 1
        for node in nodes:
            node._backward()

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

    def __rsub__(self, other):
        return self + (-other)

    def __eq__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self.data == other.data

    def __ge__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self.data >= other.data
    
    def __le__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self.data <= other.data
    
    def __lt__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self.data < other.data

    def __gt__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self.data > other.data
       
    def __repr__(self):
        """
        Return a string representation of the Tensor object.
        """
        return f"Tensor (data={str(self.data)})"