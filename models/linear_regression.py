class LinearRegression:
    def __init__(self):
        self.w = 0
        self.b = 0

    def predict(self, x):
        return self.w * x + self.b

    def train(self, dataset, lr=0.01, epochs=1000):
        for _ in range(epochs):
            for x, y in dataset:
                prediction = self.predict(x)
                error = prediction - y

                self.w -= error * lr * x
                self.b -= error * lr