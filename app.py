from flask import Flask, request, render_template
import sklearn
import numpy as np
import joblib

# Load model
Obj = joblib.load(r'C:\Users\vaish\Documents\NareshIT\Machine_Learning\flask_examples\california.joblib')
model = Obj['Model']
columns = Obj['Columns']

print("Model Columns:", columns)

app = Flask(__name__)


@app.route('/')
def Main():
    return render_template('index.html', columns=columns)


@app.route('/predict', methods=['POST'])
def predict():

    Input = []

    for column in columns:
        val = request.form.get(column, type=float)
        Input.append(val)

    print("Input:", Input)

    # Prediction
    out = model.predict([Input])

    prediction = out[0]

    print("Prediction:", prediction)

    return render_template(
        'index.html',
        columns=columns,
        prediction=prediction,
        values=request.form
    )


if __name__ == "__main__":
    app.run(debug=True)