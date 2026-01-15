from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained ML 
with open("pickle_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["CRIM"]),
            float(request.form["ZN"]),
            float(request.form["INDUS"]),
            float(request.form["CHAS"]),
            float(request.form["NOX"]),
            float(request.form["RM"]),
            float(request.form["AGE"]),
            float(request.form["DIS"]),
            float(request.form["RAD"]),
            float(request.form["TAX"]),
            float(request.form["PTRATIO"]),
            float(request.form["B"]),
            float(request.form["LSTAT"])
        ]

        final_features = np.array(features).reshape(1, -1)
        prediction = model.predict(final_features)[0]

        return render_template(
            "index.html",
            prediction=f"${round(prediction, 2)}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction="Invalid input!"
        )

if __name__ == "__main__":
    app.run(debug=True)
