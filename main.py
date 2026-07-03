from flask import Flask, render_template, request, redirect, session

from models.linear_regression import LinearRegression
from models.perceptron import Perceptron
from models.logistic_regression import LogisticRegression
from models.softmax_regression import SoftmaxRegression

from data.linear_regression_data import TRAINING_DATA as LINEAR_DATA
from data.perceptron_data import TRAINING_DATA as PERCEPTRON_DATA
from data.logistic_regression_data import TRAINING_DATA_X, TRAINING_DATA_Y
from data.softmax_regression_data import TRAINING_DATA_X as SOFTMAX_DATA_X
from data.softmax_regression_data import TRAINING_DATA_Y as SOFTMAX_DATA_Y

from database.database import (
    create_tables,
    register_user,
    login_user,
    save_prediction,
    get_predictions
)

app = Flask(__name__)
app.config["SECRET_KEY"] = "my-secret-key-123"

create_tables()


# -----------------------------
# Train AI models
# -----------------------------
linear_model = LinearRegression()
linear_model.train(LINEAR_DATA)

perceptron_model = Perceptron()
perceptron_model.train(PERCEPTRON_DATA)

logistic_model = LogisticRegression(n_features=2)
logistic_model.train(TRAINING_DATA_X, TRAINING_DATA_Y)

softmax_model = SoftmaxRegression(n_classes=3, n_inputs=2)
softmax_model.train(SOFTMAX_DATA_X, SOFTMAX_DATA_Y)


# -----------------------------
# Home page
# -----------------------------
@app.route("/")
def home():
    if "username" in session:
        return redirect("/dashboard")

    return render_template("index.html")


# -----------------------------
# Register
# -----------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    message = None

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        success = register_user(username, password)

        if success:
            return redirect("/login")

        message = "Това username вече съществува."

    return render_template("register.html", message=message)


# -----------------------------
# Login
# -----------------------------
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


# -----------------------------
# Logout
# -----------------------------
@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect("/")


# -----------------------------
# Dashboard
# -----------------------------
@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect("/login")

    return render_template(
        "dashboard.html",
        username=session["username"]
    )


# -----------------------------
# Linear Regression
# -----------------------------
@app.route("/linear-regression", methods=["GET", "POST"])
def linear_regression():
    if "username" not in session:
        return redirect("/login")

    result = None
    entered_x = None

    if request.method == "POST":
        try:
            entered_x = float(request.form["x"])
            result = round(linear_model.predict(entered_x), 2)

            save_prediction(
                session["username"],
                "Linear Regression",
                f"x = {entered_x}",
                result
            )
        except ValueError:
            result = "Моля, въведи число."

    return render_template(
        "linear_regression.html",
        result=result,
        entered_x=entered_x,
        weight=round(linear_model.w, 2),
        bias=round(linear_model.b, 2)
    )


# -----------------------------
# Perceptron
# -----------------------------
@app.route("/perceptron", methods=["GET", "POST"])
def perceptron():
    if "username" not in session:
        return redirect("/login")

    result = None
    x1 = None
    x2 = None

    if request.method == "POST":
        try:
            x1 = float(request.form["x1"])
            x2 = float(request.form["x2"])

            result = perceptron_model.predict(x1, x2)

            save_prediction(
                session["username"],
                "Perceptron",
                f"x1 = {x1}, x2 = {x2}",
                result
            )

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


# -----------------------------
# Logistic Regression
# -----------------------------
@app.route("/logistic-regression", methods=["GET", "POST"])
def logistic_regression():
    if "username" not in session:
        return redirect("/login")

    result = None
    probability = None
    x1 = None
    x2 = None

    if request.method == "POST":
        try:
            x1 = float(request.form["x1"])
            x2 = float(request.form["x2"])

            probability = logistic_model.predict_proba([x1, x2])
            result = logistic_model.predict([x1, x2])

            probability = round(probability, 4)

            save_prediction(
                session["username"],
                "Logistic Regression",
                f"x1 = {x1}, x2 = {x2}",
                result,
                probability
            )

        except ValueError:
            result = "Моля, въведи 0 или 1."

    return render_template(
        "logistic_regression.html",
        result=result,
        probability=probability,
        x1=x1,
        x2=x2,
        weights=[round(weight, 2) for weight in logistic_model.w],
        bias=round(logistic_model.b, 2)
    )


# -----------------------------
# Softmax Regression
# -----------------------------
@app.route("/softmax-regression", methods=["GET", "POST"])
def softmax_regression():
    if "username" not in session:
        return redirect("/login")

    result = None
    x1 = None
    x2 = None
    probabilities_for_page = {}

    if request.method == "POST":
        try:
            x1 = float(request.form["x1"])
            x2 = float(request.form["x2"])

            probabilities = softmax_model.predict_proba([x1, x2])
            result = softmax_model.predict([x1, x2])

            for index, probability in enumerate(probabilities):
                label = softmax_model.class_to_label[index]
                probabilities_for_page[label] = round(probability, 4)

                save_prediction(
                    session["username"],
                    "Softmax Regression",
                    f"x1 = {x1}, x2 = {x2}",
                    result,
                    probabilities_for_page
                )

        except ValueError:
            result = "Моля, въведи 0 или 1."

    return render_template(
        "softmax_regression.html",
        result=result,
        x1=x1,
        x2=x2,
        probabilities=probabilities_for_page,
        loss=round(softmax_model.last_loss, 4)
    )

@app.route("/history")
def history():
    if "username" not in session:
        return redirect("/login")

    predictions = get_predictions(session["username"])

    return render_template(
        "history.html",
        predictions=predictions
    )

@app.route("/models")
def models_page():
    if "username" not in session:
        return redirect("/login")

    return render_template("models.html")


if __name__ == "__main__":
    app.run(debug=True)