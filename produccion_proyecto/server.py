import joblib  # importa las bibliotecas joblib para cargar el
# modelo y numpy y Flask para crear la aplicación web.
import numpy as np
from flask import Flask
from flask import jsonify
app = Flask(__name__)
# POSTMAN PARA PRUEBAS


@app.route('/predict', methods=['GET'])
def predict():
    X_test = np.array([
        660,3.28358208955224,12560,29,0.295749811731615
    ])
    prediction = model.predict(X_test.reshape(1, -1))

    print(list(prediction)[0])
    return jsonify({'prediccion': str(prediction[0])})


if __name__ == "__main__":
    model = joblib.load('./models/best_model_0.832.pkl')
    app.run(port=8080)
