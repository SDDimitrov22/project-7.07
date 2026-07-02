from flask import Flask, render_template, request
app = Flask(__name__)

class LinearRegression:
   def __init__(self):
       self.w = 0
       self.b = 0
   def predict(self, x):
       return self.w * x + self.b
   def train(self, dataset, learning_rate=0.01, epochs=1000):
       for _ in range(epochs):
           for x, y in dataset:
               prediction = self.predict(x)
               error = prediction - y
               self.w = self.w - learning_rate * error * x
               self.b = self.b - learning_rate * error

# Данните са приблизително y = 2x + 1
training_data = [
   (0, 1),
   (1, 3),
   (2, 5),
   (3, 7),
   (4, 9),
   (5, 11)
]
model = LinearRegression()
model.train(training_data)

@app.route("/", methods=["GET", "POST"])
def home():
   result = None
   entered_x = None
   if request.method == "POST":
       try:
           entered_x = float(request.form["x"])
           result = round(model.predict(entered_x), 2)
       except ValueError:
           result = "Моля, въведи число."
   return render_template(
       "input_form.html",
       result=result,
       entered_x=entered_x,
       weight=round(model.w, 2),
       bias=round(model.b, 2)
   )

if __name__ == "__main__":
   app.run(debug=True)