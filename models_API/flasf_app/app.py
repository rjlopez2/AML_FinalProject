from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Set the FastAPI URL
FASTAPI_URL = 'http://localhost:8000/predict'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get data from the form
        coordinates = request.form.get('coordinates')
        zoom_level = int(request.form.get('zoom_level'))
        layer_name = request.form.get('layer_name')
        times = request.form.get('times')

        # Convert coordinates from string to a list of tuples
        coordinates = [tuple(map(float, coord.split(','))) for coord in coordinates.split(';')]
        times = times.split(',')

        # Prepare data for the API request
        data = {
            "coordinates": coordinates,
            "zoom_level": zoom_level,
            "layer_name": layer_name,
            "times": times
        }

        # Make a POST request to the FastAPI /predict endpoint
        response = requests.post(FASTAPI_URL, json=data)

        if response.status_code == 200:
            predictions = response.json()["predictions"]
            return render_template('index.html', predictions=predictions)
        else:
            return render_template('index.html', error="Failed to get predictions. Please check your input.")
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
