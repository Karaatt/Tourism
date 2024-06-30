from flask import Flask, render_template, jsonify
import requests

app = Flask(__name__)

YANDEX_API_KEY = ''

coordinates = [
    "45.044699,41.968149",
    "45.050061,41.977168",
    "45.044737,41.961861"
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_route')
def get_route():
    url = "https://api.routing.yandex.net/v2/route"
    params = {
        "apikey": YANDEX_API_KEY,
        "waypoints": "|".join(coordinates),
        "mode": "driving"
    }
    response = requests.get(url, params=params)
    data = response.json()
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
