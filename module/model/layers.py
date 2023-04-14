from nanotensor.module.engine.tensor import Tensor
import numpy as np


class ZeroGrad:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


class Neuron(ZeroGrad):
    def __init__(self, inx, activation):
        """"
        inx: number of input to the neuron
        activation : None [relu, tanh, sigmoid]
        """
        self.weight = [Tensor(np.random.normal()) for _ in range(inx)]
        self.bias = Tensor(np.random.normal())
        self.activation = activation

    def __call__(self, x):
        out = sum((x * w for x, w in zip(x, self.weight))) + self.bias
        if self.activation:
            activation = self.activation.lower()
            if activation == 'relu':
                out = out.relu()
            elif activation == 'tanh':
                out = out.tanh()
            elif activation == 'sigmoid':
                out = out.sigmoid()
        return out

    def parameters(self):
        parameters = self.weight + [self.bias]
        return parameters


class Layer(ZeroGrad):
    layerdict = {'relu': 'ReLU', 'tanh': 'Tanh', 'sigmoid': 'Sigmoid'}

    def __init__(self, size=tuple(), activation=None):
        """"
        inx: number of input to the neuron
        out: number of neurons in the layer
        activation : None [relu, tanh, sigmoid]
        """
        self.activation = activation
        self.shape = size
        self.neurons = [Neuron(size[0], activation=activation) for _ in range(size[1])]
        self.totalparam = len(self.parameters())

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) < 2 else out

    def parameters(self):
        param = [p for neuron in self.neurons for p in neuron.parameters()]
        return param

    # def softmax(self, x):
    #     # HOO HOO, had to come up with a way to implement an operaion that supports from broadcasting
    #     """
    #     Compute the softmax of vector x in a numerically stable way.
    #     """
    #     _max = max((i.data for i in x))
    #     exp_x = np.exp(x - np.array(_max))
    #     exp_x = exp_x/ np.sum(exp_x, axis=0)
    #     return exp_x 
    
    def __repr__(self):
        return f"{'Linear' if not self.activation else Layer.layerdict[self.activation.lower()]}Layer       Shape: {self.shape})      Params: {self.totalparam}"


class Sequential(ZeroGrad):
    def __init__(self, layers):
        self.layers = layers
        self.totalparams = len(self.parameters())

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        param = [p for layer in self.layers for p in layer.parameters()]
        return param

    def summary(self):
        summary = self.__repr__()
        print(summary)

    def __repr__(self):
        layout = "Layer(type)         Shape              Parameters"
        formatter = "\n---------------------------------------------------\n"
        bottom = f"Trainable Parameters                          {self.totalparams}"
        for layer in self.layers:
            layout += formatter + str(layer)
        summary = layout + formatter + bottom
        return summary
