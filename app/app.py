# Importing essential libraries
import os
import joblib
from joblib import dump
from flask import Flask, request, render_template
import numpy as np
import pickle
from flask_cors import CORS
from joblib import load
import sklearn



app = Flask(__name__)
CORS(app)

# Load the HDF5 model

#model_path = 'Random_Forest_model_final.pkl'
#model = pickle.load(open(model_path, 'rb'))

# Cargar el modelo al iniciar la aplicación
model_filename = os.path.abspath('app\Random_Forest_model_final')
model = joblib.load(model_filename)


@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        Age = int(request.form['Age'])
        RestingBP = int(request.form['RestingBP'])
        Cholesterol = int(request.form['Cholesterol'])
        FastingBS = int(request.form['FastingBS'])
        MaxHR = int(request.form['MaxHR'])
        Oldpeak = float(request.form['Oldpeak'])
        Sex_F = int(request.form.get('Sex_F',0))
        Sex_M = int(request.form.get('Sex_M',0))
        ChestPainType_ASY = int(request.form.get('ChestPainType_ASY', 0))
        ChestPainType_ATA = int(request.form.get('ChestPainType_ATA', 0))
        ChestPainType_NAP = int(request.form.get('ChestPainType_NAP', 0))
        ChestPainType_TA = int(request.form.get('ChestPainType_TA', 0))
        RestingECG_LVH = int(request.form.get('RestingECG_LVH', 0))
        RestingECG_Normal = int(request.form.get('RestingECG_Normal',0))
        RestingECG_ST = int(request.form.get('RestingECG_ST', 0))
        ExerciseAngina_N = int(request.form.get('ExerciseAngina_N',0))
        ExerciseAngina_Y = int(request.form.get('ExerciseAngina_Y',0))
        ST_Slope_Down = int(request.form.get('ST_Slope_Down',0))
        ST_Slope_Flat = int(request.form.get('ST_Slope_Flat', 0))
        ST_Slope_Up = int(request.form.get('ST_Slope_Up',0))
        
        
        features = np.array([[Age,RestingBP,Cholesterol,FastingBS,MaxHR,Oldpeak,Sex_F,Sex_M, ChestPainType_ASY, ChestPainType_ATA, ChestPainType_NAP,ChestPainType_TA, RestingECG_LVH, RestingECG_Normal, RestingECG_ST, ExerciseAngina_N, ExerciseAngina_Y, ST_Slope_Down, ST_Slope_Flat, ST_Slope_Up]])
        prediction = model.predict(features)
        
        return render_template('predict.html', prediction=prediction[0])


if __name__ == '__main__':
    app.run(debug=True)
