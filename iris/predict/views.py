import os
import pandas as pd

from .models import PredResult
from django.http import JsonResponse
from django.shortcuts import render


# Create your views here.
def predict(request):
    return render(request, 'predict/index.html')


def predict_chances(request):    # 4

    if request.POST.get('action') == 'post':

        # Receive data from client
        sepal_length = float(request.POST.get('sepal_length'))
        sepal_width = float(request.POST.get('sepal_width'))
        petal_length = float(request.POST.get('petal_length'))
        petal_width = float(request.POST.get('petal_width'))

        # Unpickle the model
        path = os.path.abspath('') + '\\model\\svc.pickle'
        #model = pd.read_pickle(r'.\model\svc.pickle')
        model = pd.read_pickle(path)
        # Make prediction
        result = model.predict(
            [[sepal_length, sepal_width, petal_length, petal_width]]
        )
        # Write into database
        PredResult.objects.create(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
            classification=result[0]
        )
        # Return for JavaScript
        return JsonResponse(
            {
                'result': result[0],
                'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width
             },
            safe=False
        )


def view_results(request):
    # Submit prediction and show all
    data = {"dataset": PredResult.objects.all()}
    return render(request, 'predict/results.html', data)
