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
        'Relative Compactness' : request.form.get('X1', type=float),
        'Surface Area' : request.form.get('X2', type=float),
        'Wall Area' : request.form.get('X3', type=float),
        'Roof Area' : request.form.get('X4', type=float),
        'Overall Height' : request.form.get('X5', type=float),
        'Orientation' : request.form.get('X6', type=int),
        'Glazing Area' : request.form.get('X7', type=float),
        'Glazing Area Distribution' : request.form.get('X8', type=int)
    }

    # Prepare the input data in the format your model expects
    input_data = [input_values['Relative Compactness'], input_values['Surface Area'], input_values['Wall Area'],
                  input_values['Roof Area'], input_values['Overall Height'], input_values['Orientation'],
                  input_values['Glazing Area'], input_values['Glazing Area Distribution']]
    
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
     port = int(os.environ.get('PORT', 5000))
     app.run(host='0.0.0.0', port=port, debug=False)
