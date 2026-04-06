from django.shortcuts import render
import pickle
import numpy as np
import os

# Model load
model_path = os.path.join(os.path.dirname(__file__), '../../ml_model/model.pkl')
model = pickle.load(open(model_path, 'rb'))

def home(request):
    prediction = None

    if request.method == 'POST':
        area = float(request.POST['area'])
        bedrooms = int(request.POST['bedrooms'])
        bathrooms = int(request.POST['bathrooms'])

        data = np.array([[area, bedrooms, bathrooms]])
        prediction = model.predict(data)[0]

    return render(request, 'index.html', {'prediction': prediction})