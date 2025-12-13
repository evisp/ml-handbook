# app.py
from flask import Flask, request, jsonify, render_template
from model_loader import model

app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    furnishing = data.get("furnishing", "unfurnished")
    furnishing_map = {
        "unfurnished": {
            "furnishing_unfurnished": 1,
            "furnishing_fully_furnished": 0,
            "furnishing_partially_furnished": 0,
        },
        "fully_furnished": {
            "furnishing_unfurnished": 0,
            "furnishing_fully_furnished": 1,
            "furnishing_partially_furnished": 0,
        },
        "partially_furnished": {
            "furnishing_unfurnished": 0,
            "furnishing_fully_furnished": 0,
            "furnishing_partially_furnished": 1,
        },
    }

    user_input = {
        "area_m2": float(data.get("area_m2", 80)),
        "bathrooms": float(data.get("bathrooms", 1)),
        "bedrooms": float(data.get("bedrooms", 2)),
        "floor": float(data.get("floor", 2)),
        "dist_to_blloku_km": float(data.get("dist_to_blloku_km", 2.0)),
        "has_elevator": 1 if data.get("has_elevator", False) else 0,
        "has_parking": 1 if data.get("has_parking", False) else 0,
        "has_garage": 1 if data.get("has_garage", False) else 0,
        "has_terrace": 1 if data.get("has_terrace", False) else 0,
        **furnishing_map.get(furnishing, furnishing_map["unfurnished"]),
    }

    prediction = model.predict(user_input)
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
