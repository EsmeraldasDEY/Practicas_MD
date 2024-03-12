import joblib #importa las bibliotecas joblib para cargar el
#modelo y numpy y Flask para crear la aplicación web.
import numpy as np
from flask import Flask
from flask import jsonify
app = Flask(__name__)
#POSTMAN PARA PRUEBAS
@app.route('/predict', methods=['GET'])
def predict():
    X_test = np.array([6.568476814,6.181523186,1.870765686,1.27429688,0.710098088,0.604130983,0.33047387,0.439299256,1.14546442])
    prediction = model.predict(X_test.reshape(1,-1))
    return jsonify({'prediccion' : list(prediction)})
if __name__ == "__main__":
    model = joblib.load('./models/best_model_0.976.pkl')
    app.run(port=8080)
