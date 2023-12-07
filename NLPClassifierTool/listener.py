# Model-Extension Integration

import fakenews_prediction
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes


@app.route("/hello")
def hello():
    print("Hello")
    text = request.args.get('text')  # Get the 'name' query parameter, default to 'World'

    #text = "SHARE THIS EVERYWHERE! DISEASED REFUGEES Get SSN and Passport Upon Arrival to the U.S. [Video]"
    label, probability = fakenews_prediction.predict_label(text)
    print(f"Predicted Label: {label}")
    print(f"Probability: {probability}")
    return (f"{label}\n  {probability}")


if __name__ == "__main__":
    app.run(debug=True)



