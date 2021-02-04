import numpy as np


class Matrix():

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.matrix = []

        for i in range(self.rows):
            self.column = []
            for j in range(self.cols):
                self.column.append(0)
            self.matrix.append(self.column)


    def multiply(self, n):
        if type(n) == Matrix:
            result = Matrix(self.rows, n.cols)
            for i in range(result.rows):
                for j in range(result.cols):
                    sum = 0
                    for k in range(self.cols):
                        sum += self.matrix[i][k] * n.matrix[k][j]
                    result.matrix[i][j] = sum
            return result
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.matrix[i][j] *= n
            return self


    def add(self, n):
        if type(n) == Matrix:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.matrix[i][j] += n.matrix[i][j]
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.matrix[i][j] += n


    def transpose(self):
        result = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result.matrix[j][i] = self.matrix[i][j]
        return result


    def toArray(self):
        arr = []
        for i in range(self.rows):
            for j in range(self.cols):
                arr.append(self.matrix[i][j])
        return arr


    def randomize(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self.matrix[i][j] += float(2 * np.random.random(1) - 1)


    def map(self, func):
        for i in range(self.rows):
            for j in range(self.cols):
                val = self.matrix[i][j]
                self.matrix[i][j] = func(val)
        return self


def fromArray(arr):
    m = Matrix(len(arr), 1)
    for i in range(len(arr)):
        m.matrix[i][0] = arr[i]
    return m

def subtract(a, b):
    result = Matrix(a.rows, a.cols)
    for i in range(result.rows):
        for j in range(result.cols):
            result.matrix[i][j] = a.matrix[i][j] - b.matrix[i][j]
    return result

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)
