from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Cargar el modelo entrenado
model_path = 'models/charger_classifier.pkl'
loaded_model = joblib.load(model_path)

# Definir el mapeo de marcas
brand_mapping = {0: 'Xiaomi', 1: 'Samsung'}

# Define la puntuación y las recomendaciones
rating_mapping = {
    1: "Este cargador es PIRATA y de baja calidad, podría dañar su dispositivo. Se recomienda no usarlo.",
    1.5: "Este cargador es PIRATA y de baja calidad. Se recomienda encontrar una alternativa mejor.",
    2: "Este cargador es PIRATA y de baja calidad. Se recomienda encontrar una alternativa mejor.",
    2.5: "Este cargador es PIRATA y de una calidad no tan baja, pero podría no ser ideal para su dispositivo.",
    3: "Este cargador es de calidad ACEPTABLE. Puede usarlo, pero no es el mejor disponible.",
    3.5: "Este cargador es de calidad ACEPTABLE. Puede usarlo con confianza.",
    4: "Este cargador es de calidad ACEPTABLE. Puede usarlo con confianza.",
    4.5: "Este cargador es de muy buena calidad. Ideal para su dispositivo.",
    5: "Este cargador es excelente y original. Perfecto para su dispositivo."
}

def get_rating_and_recommendation(brand):
    if brand == "Xiaomi":
        return 5, rating_mapping[5]
    elif brand == "Samsung":
        return 5, rating_mapping[5]
    else:
        # Simulación de puntuación para pirata y aceptable
        import random
        rating = random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4])
        return rating, rating_mapping[rating]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model = data.get('model', 'desconocido')
    current = data['current']
    voltage = data['voltage']
    power = data['power']

    # Predecir la marca del cargador
    try:
        prediction = loaded_model.predict(np.array([[current, voltage, power]]))[0]
        predicted_brand = brand_mapping.get(prediction, "desconocido")
    except:
        predicted_brand = "desconocido"

    # Obtener la calificación y la recomendación
    rating, recommendation = get_rating_and_recommendation(predicted_brand)

    response = {
        'model': model,
        'current': current,
        'voltage': voltage,
        'power': power,
        'predicted_brand': predicted_brand,
        'rating': rating,
        'recommendation': recommendation
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
