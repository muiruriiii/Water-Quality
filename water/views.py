import joblib
from django.shortcuts import render, redirect
from django.contrib import messages

# Create your views here.
def predict(request):

    if request.method == 'POST':
        pH = request.POST['pH']
        hardness = request.POST['hardness']
        solids = request.POST['solids']
        chloramines = request.POST['chloramines']
        sulfate = request.POST['sulfate']
        conductivity = request.POST['conductivity']
        organic_carbon = request.POST['organic_carbon']
        trihalomethanes = request.POST['trihalomethanes']
        turbidity = request.POST['turbidity']

        classifier = joblib.load("model_svm.pkl")

        prediction = classifier.predict([[pH, hardness, solids, chloramines, sulfate, conductivity, organic_carbon,trihalomethanes,turbidity]])
        messages.success(request, prediction)
        return redirect("predict")
    else:
        context = {}
        return render(request, 'water/predict.html', context)