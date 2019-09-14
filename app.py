# Dependencies used to build the web app
from flask import Flask, render_template, jsonify, request
from werkzeug.exceptions import HTTPException
import traceback
import io

# Used to load and run the model
# Packages: scikit-learn, joblib
from sklearn.externals import joblib
import numpy as np

app = Flask(__name__)
model = None


def load_model():
    """
    This method loads our model, so that we can use it to perform predictions.
    """

    global model
    if not model:
        # print("--->>>> loading model...")
        # TODO: Change the filename to match your model's filename
        model = joblib.load("breast_cancer_classifier.pkl")
    return model


# Homepage: The form
@app.route('/')
def form():
    # Get the values if specified in the URL (so we can edit them)
    values = request.values

    return render_template('form.html', form_values=values)


@app.route('/process_form', methods=["POST"])
def process_form():
    # Get the values that were submitted in the form, and
    # convert them to correct numeric format (integer or floats)
    values = {
        'worst_perimeter': float(request.form['worst_perimeter']),
        'worst_concave_points': float(request.form['worst_concave_points']),
        'worst_radius': float(request.form['worst_radius']),
        'mean_concave_points': float(request.form['mean_concave_points']),
        'worst_concavity': float(request.form['worst_concavity']),
    }

    # These are the values that we will display on the results page
    input_values = {
        "Worst perimeter": values['worst_perimeter'],
        "Worst Concave Points": values['worst_concave_points'],
        "Worst Radius": values['worst_radius'],
        "Mean Concave Points": values['mean_concave_points'],
        "Worst Concavity": values['worst_concavity'],
    }

    # Load the model & model params
    model = load_model()
    model_params = [[
        values['worst_perimeter'],
        values['worst_concave_points'],
        values['worst_radius'],
        values['mean_concave_points'],
        values['worst_concavity'],
    ]]

    # Use our model to perform predictions

    # model.predict returns an array containing the prediction
    #    e.g. => [[0]]
    prediction = model.predict(model_params)[0]
    print(prediction)

    # model.predict_proba returns an array containing the probabilities of each class
    #    e.g. => [[0.65566831, 0.34433169]]
    probabilities = model.predict_proba(model_params)[0]
    print(probabilities)

    return render_template('results.html', prediction=prediction, probabilities=probabilities, input_values=input_values, form_values=values)


# Start the server
if __name__ == "__main__":
    print("* Starting Flask server..."
          "please wait until server has fully started")
    # debug=True options allows us to view our changes without restarting the server.
    app.run(host='0.0.0.0', debug=True)
