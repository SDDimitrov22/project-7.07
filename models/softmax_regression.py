import math
import random


class SoftmaxRegression:
    def __init__(self, n_classes, n_inputs):
        self.n_classes = n_classes
        self.n_inputs = n_inputs

        self.weights = []

        for _ in range(n_classes):
            row = []

            for _ in range(n_inputs):
                row.append(random.uniform(-0.1, 0.1))

            self.weights.append(row)

        self.bias = [0.0] * n_classes
        self.class_to_label = {}
        self.label_to_class = {}
        self.last_loss = 0.0

    def weighted_sum(self, x):
        z = [0.0] * self.n_classes

        for class_index in range(self.n_classes):
            for input_index in range(self.n_inputs):
                z[class_index] += x[input_index] * self.weights[class_index][input_index]

            z[class_index] += self.bias[class_index]

        return z

    def softmax(self, z):
        # Изваждаме най-голямата стойност, за да няма проблем с големи числа.
        max_value = max(z)

        exponential_values = []

        for value in z:
            exponential_values.append(math.exp(value - max_value))

        total = sum(exponential_values)

        probabilities = []

        for value in exponential_values:
            probabilities.append(value / total)

        return probabilities

    def one_hot_encode(self, y):
        unique_labels = sorted(list(set(y)))

        for index, label in enumerate(unique_labels):
            self.class_to_label[index] = label
            self.label_to_class[label] = index

    def calculate_loss(self, probabilities, correct_class):
        # Прост loss: 1 - вероятността на правилния клас.
        return 1 - probabilities[correct_class]

    def train(self, dataset_x, dataset_y, epochs=2000, lr=0.1):
        self.one_hot_encode(dataset_y)

        total_loss = 0.0

        for _ in range(epochs):
            total_loss = 0.0

            for x, label in zip(dataset_x, dataset_y):
                correct_class = self.label_to_class[label]

                z = self.weighted_sum(x)
                probabilities = self.softmax(z)

                total_loss += self.calculate_loss(probabilities, correct_class)

                # error = predicted probability - correct value
                for class_index in range(self.n_classes):
                    target = 0

                    if class_index == correct_class:
                        target = 1

                    error = probabilities[class_index] - target

                    for input_index in range(self.n_inputs):
                        self.weights[class_index][input_index] -= error * lr * x[input_index]

                    self.bias[class_index] -= error * lr

        self.last_loss = total_loss / len(dataset_x)

    def predict_proba(self, x):
        z = self.weighted_sum(x)
        return self.softmax(z)

    def predict(self, x):
        probabilities = self.predict_proba(x)

        best_class = 0

        for index in range(1, len(probabilities)):
            if probabilities[index] > probabilities[best_class]:
                best_class = index

        return self.class_to_label[best_class]