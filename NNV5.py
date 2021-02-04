import MathSource as MS


def layer_activation(layer, inputs, bias):
    output = MS.Matrix.multiply(layer, inputs)
    output.add(bias)
    output.map(MS.sigmoid)
    return output

def gradient_calculation(inputs, errors, learning_rate):
    gradients = inputs.map(MS.sigmoid_deriv)
    gradients = MS.Matrix.multiply(gradients, errors)
    gradients = MS.Matrix.multiply(gradients, learning_rate)
    return gradients

def applying_adjustments(inputs, gradients, layer, bias):
    hidden_T = MS.Matrix.transpose(inputs)
    weights_deltas = MS.Matrix.multiply(gradients, hidden_T)
    layer.add(weights_deltas)
    bias.add(gradients)


class Layer():

    def __init__(self, weights, bias, result=MS.Matrix):
        self.weights = weights
        self.bias = bias
        self.result = result


class NeuralNetwork():

    def __init__(self, nodes):
        self.nodes = nodes
        self.layers_list = []
        for i in range(len(self.nodes) - 1):
            weights = MS.Matrix(self.nodes[i + 1], self.nodes[i])
            bias = MS.Matrix(self.nodes[i + 1], 1)
            weights.randomize()
            bias.randomize()
            self.layers_list.append(Layer(weights, bias))

        self.learning_rate = 1


    def feedforward(self, input_array):

        inputs = MS.fromArray(input_array)

        data = inputs
        for layers in self.layers_list:
            results = layer_activation(layers.weights, data, layers.bias)
            layers.result = results
            data = results

        return data.toArray()


    def train(self, input_array, target_array):

        # Forward Propagation
        inputs = MS.fromArray(input_array)

        data = inputs
        for layers in self.layers_list:
            results = layer_activation(layers.weights, data, layers.bias)
            layers.result = results
            data = results

        # Error calculation
        targets = MS.fromArray(target_array)
        error = MS.subtract(targets, self.layers_list[-1].result)
        index = list(range(len(self.layers_list)))
        index.reverse()
        first = True
        # Back Propagation
        for i in index:
            if first:
                gradients = gradient_calculation(self.layers_list[i].result, error, self.learning_rate)
                applying_adjustments(self.layers_list[i - 1].result, gradients, self.layers_list[i].weights,
                                     self.layers_list[i].bias)
                first = False
            else:
                weights_T = MS.Matrix.transpose(self.layers_list[i + 1].weights)
                error = MS.Matrix.multiply(weights_T, error)
                gradients = gradient_calculation(self.layers_list[i].result, error, self.learning_rate)
                if i:
                    applying_adjustments(self.layers_list[i - 1].result, gradients, self.layers_list[i].weights,
                                         self.layers_list[i].bias)
                else:
                    applying_adjustments(inputs, gradients, self.layers_list[i].weights,
                                         self.layers_list[i].bias)
