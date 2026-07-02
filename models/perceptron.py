class Perceptron:
    def __init__(self):
        self.w1 = 1
        self.w2 = 1
        self.b = 0

    def predict(self, x1, x2):
        z = self.w1 * x1 + self.w2 * x2 + self.b

        if z < 0:
            return 0
        else:
            return 1

    def train(self, dataset, lr=0.01, epochs=1000):
        for _ in range(epochs):
            for x1, x2, y in dataset:
                prediction = self.predict(x1, x2)
                error = prediction - y

                self.w1 -= error * lr * x1
                self.w2 -= error * lr * x2
                self.b -= error * lr