import math
import random
import activation_functions


class Value:
    def __init__(self, data, _parents=(), _op=''):
        self.data = data
        self._prev = set(_parents)
        self.grad = 0
        self._op = _op
        self.__update_parent_grad__ = lambda: None
        self.label = ''

    def __repr__(self):
        return f"Value(data={self.data})"

    def _backward(self):
        self.__update_parent_grad__()

    def backprop(self):
        self.grad = 1
        visited_node = set()
        visit_order = []

        def topology_sort(current_node):
            for node in current_node._prev:
                if node not in visited_node:
                    visited_node.add(node)
                    topology_sort(node)
            visit_order.append(current_node)

        topology_sort(self)
        visit_order.reverse()
        for node in visit_order:
            node._backward()

    def __add__(self, other):
        if type(other) is not Value:
            other = Value(other)

        output = Value(self.data + other.data, (self, other), '+')

        def __update_parent_grad__():
            self.grad += output.grad
            other.grad += output.grad

        output.__update_parent_grad__ = __update_parent_grad__
        return output

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if type(other) is not Value:
            other = Value(other)
        return self + -1 * other

    def __rsub__(self, other):
        if type(other) is not Value:
            other = Value(other)
        return other - self

    def __mul__(self, other):
        if type(other) is not Value:
            other = Value(other)
        output = Value(self.data * other.data, (self, other), '*')

        def __update_parent_grad__():
            self_grad_update = other.data * output.grad
            other_grad_update = self.data * output.grad
            self.grad += self_grad_update
            other.grad += other_grad_update

        output.__update_parent_grad__ = __update_parent_grad__
        return output

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if type(other) is not Value:
            return self / Value(other)
        return self * other ** -1

    def __rtruediv__(self, other):
        if type(other) is not Value:
            return Value(other) / self
        return other.__truediv__(self)

    def __pow__(self, power, modulo=None):
        if type(power) is not Value:
            return self ** Value(power)

        output = Value(self.data ** power.data, (self, power), '**')

        def __update_parent_grad__():
            self_grad_update = (power.data * self.data ** (power.data - 1)) * output.grad
            other_grad_update = ((self.data ** power.data) * math.log(abs(self.data)) * (
                        self.data / abs(self.data))) * output.grad

            self.grad += self_grad_update
            power.grad += other_grad_update

        output.__update_parent_grad__ = __update_parent_grad__
        return output

    def __rpow__(self, other):
        if type(other) is not Value:
            other = Value(other)
        return other ** self

    def __lt__(self, other):
        if type(other) is not Value:
            other = Value(other)
        return self.data < other.data

    def __gt__(self, other):
        if type(other) is not Value:
            other = Value(other)
        return self.data > other.data

    def __neg__(self):
        return self * -1

class Neuron:
    def __init__(self, input_size, has_bias=True):
        self.input_size = input_size
        self.weights = []
        for num in range(input_size):
            self.weights.append(Value(random.uniform(-1, 1)))
        if has_bias:
            self.bias = Value(random.uniform(-1, 1))

    def forward(self, inputs):
        out = 0
        for idx in range(len(self.weights)):
            out += self.weights[idx] * inputs[idx]
        out += self.bias
        return out

    def get_parameters(self):
        parameters = []
        parameters += self.weights
        if self.bias is not None:
            parameters += [self.bias]
        return parameters


class LinearLayer:
    def __init__(self, input_size, layer_size, activation_function=None):
        self.neurons = []
        for num in range(layer_size):
            self.neurons.append(Neuron(input_size))
        self.activation_function = activation_function

    def forward(self, inputs):
        output_list = []
        for neuron in self.neurons:
            output = neuron.forward(inputs)
            if self.activation_function is not None:
                output = self.activation_function(output)
            output_list.append(output)

        return output_list[0] if len(output_list) == 1 else output_list

    def get_parameters(self):
        parameters = []
        for neuron in self.neurons:
            parameters += neuron.get_parameters()
        return parameters


class MLP:

    def __init__(self, input_size, layer_config, activation_function=None):
        sz = [input_size] + layer_config
        self.layers = [LinearLayer(sz[i], sz[i + 1], activation_function) for i in range(len(layer_config))]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def get_parameters(self):
        parameters = []
        for layer in self.layers:
            parameters += layer.get_parameters()
        return parameters


def mse(target, output):
    return (target - output)**2


# Fun exercise :- Try removing tanh and train model :)
model = MLP(3, [4, 4, 1], activation_functions.tanh)
print(len(model.get_parameters()))

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # desired targets

min_loss = None
early_stopping = 5
last_best_gap = 0
itr_count = 0
max_iter = 30000

while itr_count < max_iter:

    # forward pass
    pred = []
    for x in xs:
        pred.append(model.forward(x))

    loss = Value(0)
    for idx in range(len(xs)):
        loss = loss + mse(ys[idx], pred[idx])

    # backpropagate loss and update weights
    for p in model.get_parameters():
        p.grad = 0.0
    loss.backprop()

    for p in model.get_parameters():
        if p.grad > 1:
            p.grad = 1
        p.data -= p.grad * 0.1

    # compare loss with min loss and stop if loss does not improve
    if min_loss is None or min_loss > loss:
        min_loss = loss
        last_best_gap = 0

    else:
        last_best_gap += 1
        if last_best_gap > early_stopping:
            break

    itr_count += 1
    if itr_count % 100 == 0:
        print(itr_count, " :- ", loss.data)
