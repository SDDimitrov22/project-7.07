import math


class LogisticRegression:
    def __init__(self, n_features, threshold=0.5):
        self.n_features = n_features
        self.threshold = threshold

        self.w = [0.0] * n_features
        self.b = 0.0

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def predict_proba(self, x):
        z = 0

        for i in range(self.n_features):
            z += self.w[i] * x[i]

        z += self.b

        return self.sigmoid(z)

    def predict(self, x):
        probability = self.predict_proba(x)

        if probability >= self.threshold:
            return 1
        else:
            return 0

    def train(self, dataset_x, dataset_y, epochs=1000, learning_rate=0.1):
        for _ in range(epochs):
            for x, y_actual in zip(dataset_x, dataset_y):
                probability = self.predict_proba(x)

                # Правилен отговор минус предсказана вероятност
                error = y_actual - probability

                for i in range(self.n_features):
                    self.w[i] += error * learning_rate * x[i]

                self.b += error * learning_rate