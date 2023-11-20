from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the models
rf_model = joblib.load('random_forest_heating_model.pkl')
gb_model = joblib.load('gradient_boosting_cooling_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')  # Render the input form

@app.route('/predict', methods=['POST'])


def predict():
    # Extract inputs from form
    input_values = {
        'X1' : request.form.get('X1', type=float),
        'X3' : request.form.get('X3', type=float),
        'X2' : request.form.get('X2', type=float),
        'X4' : request.form.get('X4', type=float),
        'X5' : request.form.get('X5', type=float),
        'X6' : request.form.get('X6', type=float),
        'X7' : request.form.get('X7', type=float),
        'X8' : request.form.get('X8', type=float)
    }

    # Prepare the input data in the format your model expects
    input_data = list(input_values.values())
    
    # Make predictions
    heating_load = rf_model.predict([input_data])[0]
    cooling_load = gb_model.predict([input_data])[0]

    # Combine the results with the input values
    results = {
        'inputs': input_values,
        'heating_load': heating_load,
        'cooling_load': cooling_load
    }

    # Render a new template to display the results
    return render_template('results.html', results=results)
if __name__ == '__main__':
    app.run(debug=True)
