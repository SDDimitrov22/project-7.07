from flask import Flask, render_template, request

from models.linear_regression import LinearRegression
from models.perceptron import Perceptron

from data.linear_regression_data import TRAINING_DATA as LINEAR_DATA
from data.perceptron_data import TRAINING_DATA as PERCEPTRON_DATA

app = Flask(__name__)


linear_model = LinearRegression()
linear_model.train(LINEAR_DATA)

perceptron_model = Perceptron()
perceptron_model.train(PERCEPTRON_DATA)


@app.route("/")
@app.route("/linear-regression", methods=["GET", "POST"])
def linear_regression():
    result = None
    entered_x = None

    if request.method == "POST":
        try:
            entered_x = float(request.form["x"])
            result = round(linear_model.predict(entered_x), 2)
        except ValueError:
            result = "Моля, въведи число."

    return render_template(
        "linear_regression.html",
        result=result,
        entered_x=entered_x,
        weight=round(linear_model.w, 2),
        bias=round(linear_model.b, 2)
    )


@app.route("/perceptron", methods=["GET", "POST"])
def perceptron():
    result = None
    x1 = None
    x2 = None

    if request.method == "POST":
        try:
            x1 = float(request.form["x1"])
            x2 = float(request.form["x2"])

            result = perceptron_model.predict(x1, x2)
        except ValueError:
            result = "Моля, въведи 0 или 1."

    return render_template(
        "perceptron.html",
        result=result,
        x1=x1,
        x2=x2,
        w1=round(perceptron_model.w1, 2),
        w2=round(perceptron_model.w2, 2),
        bias=round(perceptron_model.b, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)