import math


class SoftmaxRegression:
    def __init__(self):
        # Клас I
        self.w1_i = 0
        self.w2_i = 0
        self.b_i = 0

        # Клас You
        self.w1_you = 0
        self.w2_you = 0
        self.b_you = 0

        # Клас He
        self.w1_he = 0
        self.w2_he = 0
        self.b_he = 0

    def calculate(self, x1, x2):
        # z за всеки клас
        z_i = self.w1_i * x1 + self.w2_i * x2 + self.b_i
        z_you = self.w1_you * x1 + self.w2_you * x2 + self.b_you
        z_he = self.w1_he * x1 + self.w2_he * x2 + self.b_he

        # Softmax: превръща z във вероятности
        e_i = math.exp(z_i)
        e_you = math.exp(z_you)
        e_he = math.exp(z_he)

        total = e_i + e_you + e_he

        p_i = e_i / total
        p_you = e_you / total
        p_he = e_he / total

        return p_i, p_you, p_he

    def predict(self, x1, x2):
        p_i, p_you, p_he = self.calculate(x1, x2)

        if p_i > p_you and p_i > p_he:
            return "I"

        if p_you > p_he:
            return "You"

        return "He"

    def train(self, dataset, lr=0.1, epochs=1000):
        for _ in range(epochs):
            for x1, x2, y in dataset:
                p_i, p_you, p_he = self.calculate(x1, x2)

                # Верният клас е 1, другите са 0
                y_i = 0
                y_you = 0
                y_he = 0

                if y == "I":
                    y_i = 1
                elif y == "You":
                    y_you = 1
                else:
                    y_he = 1

                # error = p - y
                error_i = p_i - y_i
                error_you = p_you - y_you
                error_he = p_he - y_he

                # Обновяване на weights и bias
                self.w1_i -= error_i * lr * x1
                self.w2_i -= error_i * lr * x2
                self.b_i -= error_i * lr

                self.w1_you -= error_you * lr * x1
                self.w2_you -= error_you * lr * x2
                self.b_you -= error_you * lr

                self.w1_he -= error_he * lr * x1
                self.w2_he -= error_he * lr * x2
                self.b_he -= error_he * lr