from flask import Flask, render_template, request, redirect, session

from models.linear_regression import LinearRegression
from models.perceptron import Perceptron
from models.logistic_regression import LogisticRegression
from models.softmax_regression import SoftmaxRegression

from data.linear_regression_data import TRAINING_DATA as LINEAR_DATA
from data.perceptron_data import TRAINING_DATA as PERCEPTRON_DATA
from data.logistic_regression_data import TRAINING_DATA_X, TRAINING_DATA_Y
from data.softmax_regression_data import TRAINING_DATA as SOFTMAX_DATA

from database.database import (
    create_tables,
    register_user,
    login_user,
    save_prediction,
    get_predictions
)

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret-key"

create_tables()


# Обучаване на моделите
linear_model = LinearRegression()
linear_model.train(LINEAR_DATA)

perceptron_model = Perceptron()
perceptron_model.train(PERCEPTRON_DATA)

logistic_model = LogisticRegression(n_features=2)
logistic_model.train(TRAINING_DATA_X, TRAINING_DATA_Y)

softmax_model = SoftmaxRegression()
softmax_model.train(SOFTMAX_DATA)


def is_logged_in():
    return "username" in session


@app.route("/")
def home():
    if is_logged_in():
        return redirect("/dashboard")

    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    message = None

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if register_user(username, password):
            return redirect("/login")

        message = "Това username вече съществува."

    return render_template("register.html", message=message)


@app.route("/login", methods=["GET", "POST"])
def login():
    message = None

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if login_user(username, password):
            session["username"] = username
            return redirect("/dashboard")

        message = "Грешно username или password."

    return render_template("login.html", message=message)


@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect("/")


@app.route("/dashboard")
def dashboard():
    if not is_logged_in():
        return redirect("/login")

    return render_template("dashboard.html", username=session["username"])


@app.route("/models")
def models_page():
    if not is_logged_in():
        return redirect("/login")

    return render_template("models.html")


@app.route("/history")
def history():
    if not is_logged_in():
        return redirect("/login")

    predictions = get_predictions(session["username"])
    return render_template("history.html", predictions=predictions)


@app.route("/linear-regression", methods=["GET", "POST"])
def linear_regression():
    if not is_logged_in():
        return redirect("/login")

    result = None
    x = None

    if request.method == "POST":
        x = float(request.form["x"])
        result = round(linear_model.predict(x), 2)

        save_prediction(
            session["username"],
            "Linear Regression",
            f"x = {x}",
            result
        )

    return render_template(
        "linear_regression.html",
        result=result,
        entered_x=x,
        weight=round(linear_model.w, 2),
        bias=round(linear_model.b, 2)
    )


@app.route("/perceptron", methods=["GET", "POST"])
def perceptron():
    if not is_logged_in():
        return redirect("/login")

    result = None
    x1 = None
    x2 = None

    if request.method == "POST":
        x1 = float(request.form["x1"])
        x2 = float(request.form["x2"])

        result = perceptron_model.predict(x1, x2)

        save_prediction(
            session["username"],
            "Perceptron",
            f"x1 = {x1}, x2 = {x2}",
            result
        )

    return render_template(
        "perceptron.html",
        result=result,
        x1=x1,
        x2=x2,
        w1=round(perceptron_model.w1, 2),
        w2=round(perceptron_model.w2, 2),
        bias=round(perceptron_model.b, 2)
    )


@app.route("/logistic-regression", methods=["GET", "POST"])
def logistic_regression():
    if not is_logged_in():
        return redirect("/login")

    result = None
    probability = None
    x1 = None
    x2 = None

    if request.method == "POST":
        x1 = float(request.form["x1"])
        x2 = float(request.form["x2"])

        probability = round(logistic_model.predict_proba([x1, x2]), 4)
        result = logistic_model.predict([x1, x2])

        save_prediction(
            session["username"],
            "Logistic Regression",
            f"x1 = {x1}, x2 = {x2}",
            result,
            probability
        )

    return render_template(
        "logistic_regression.html",
        result=result,
        probability=probability,
        x1=x1,
        x2=x2,
        weights=[round(weight, 2) for weight in logistic_model.w],
        bias=round(logistic_model.b, 2)
    )


@app.route("/softmax-regression", methods=["GET", "POST"])
def softmax_regression():
    if not is_logged_in():
        return redirect("/login")

    result = None
    x1 = None
    x2 = None
    probabilities = {}

    if request.method == "POST":
        x1 = float(request.form["x1"])
        x2 = float(request.form["x2"])

        p_i, p_you, p_he = softmax_model.calculate(x1, x2)

        probabilities = {
            "I": round(p_i, 4),
            "You": round(p_you, 4),
            "He": round(p_he, 4)
        }

        result = softmax_model.predict(x1, x2)

        save_prediction(
            session["username"],
            "Softmax Regression",
            f"x1 = {x1}, x2 = {x2}",
            result,
            probabilities
        )

    return render_template(
        "softmax_regression.html",
        result=result,
        x1=x1,
        x2=x2,
        probabilities=probabilities,
        loss="Simple Softmax model"
    )


if __name__ == "__main__":
    app.run(debug=True)