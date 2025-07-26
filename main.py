from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
import os

app = Flask(__name__)
CORS(app)

model = joblib.load("toweriq_model_prob.pkl")
scaler = joblib.load("toweriq_scaler_prob.pkl")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)*2 + cos(lat1) * cos(lat2) * sin(dlon / 2)*2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def suggest_action(prob, alt_diff):
    if prob > 0.8:
        return "تنبيه فوري! احتمالية تصادم مرتفعة، راجع الارتفاعات والمسارات"
    elif prob > 0.5:
        return "تحذير: اقتراح بتغيير مسار أو سرعة إحدى الطائرتين"
    elif alt_diff < 300:
        return "مراجعة فرق الارتفاع: الفرق قليل"
    else:
        return "الوضع آمن، لا حاجة لإجراء حالياً"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        lat1 = float(data['lat1'])
        lon1 = float(data['lon1'])
        alt1 = float(data['alt1'])
        speed1 = float(data['speed1'])
        heading1 = float(data['heading1'])

        lat2 = float(data['lat2'])
        lon2 = float(data['lon2'])
        alt2 = float(data['alt2'])
        speed2 = float(data['speed2'])
        heading2 = float(data['heading2'])

        distance = haversine(lat1, lon1, lat2, lon2)
        alt_diff = abs(alt1 - alt2)
        speed_diff = abs(speed1 - speed2)
        heading_diff = abs((heading1 - heading2 + 180) % 360 - 180)

        features = pd.DataFrame([[
            lat1, lon1, alt1, speed1, heading1,
            lat2, lon2, alt2, speed2, heading2,
            distance, alt_diff, speed_diff, heading_diff
        ]], columns=[
            'lat1', 'lon1', 'alt1', 'speed1', 'heading1',
            'lat2', 'lon2', 'alt2', 'speed2', 'heading2',
            'distance', 'alt_diff', 'speed_diff', 'heading_diff'
        ])

        features_scaled = scaler.transform(features)
        prob = model.predict_proba(features_scaled)[0][1]
        prob_percent = f"{prob * 100:.2f}%"
        action = suggest_action(prob, alt_diff)

        return jsonify({
            'collision_probability': prob_percent,
            'action_required': action
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)