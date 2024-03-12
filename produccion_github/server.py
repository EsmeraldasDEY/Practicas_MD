import joblib  # importa las bibliotecas joblib para cargar el
# modelo y numpy y Flask para crear la aplicación web.
import numpy as np
from flask import Flask
from flask import jsonify
app = Flask(__name__)
# POSTMAN PARA PRUEBAS


@app.route('/', methods=['GET'])
def predict():
    X_test = np.array([
2,2,4777,1,100,1108,162975,10482,0,18325,15017,8.4,2.5,1681,12.6,4368,241309,38949,42,254,2698,1314,46466,0,0,0,25940,1,5,4368,0,0,0.5,0.090988626,3007,1309,4269,3,1290,5,11219,228600,21,5,50
    ])
    prediction = model.predict(X_test.reshape(1, -1))

    print(list(prediction)[0])
    return jsonify({'prediccion': str(prediction[0])})


if __name__ == "__main__":
    model = joblib.load('./models/best_model_0.984.pkl')
    app.run(port=8080)
