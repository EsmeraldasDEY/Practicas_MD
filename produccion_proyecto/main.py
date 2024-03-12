from utils import Utils
from models import Models
from sklearn.preprocessing import StandardScaler  # Normalizar los datos

#from GridSearchCV import Models

import warnings
warnings.simplefilter("ignore")

utils = Utils()
models = Models()
ds = utils.load_from_csv('./in/GIT2.csv')
dt_features = StandardScaler().fit_transform(ds)  # Normalizamnos los datos

X, y = utils.features_target(dt_features, ['%Toxicos'],['%Toxicos'])

models.grid_training(X,y)
