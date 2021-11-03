import os
import pandas as pd
import numpy as np
import seaborn as sns

from django.utils import timezone
from django.http import JsonResponse
from django.shortcuts import render

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

from .models import TrainResult


# Create your views here.
def train(request):
    return render(request, 'train/index.html')

def train_svc(request):

    # Receive data from client
    hp_c = float(request.POST.get('hp_c'))

    # Load dataset
    csv_path = os.path.abspath('') + '\\train\\data\\iris.csv'
    #df = pd.read_csv(r"../data/iris.csv")
    df = pd.read_csv(csv_path)
    # Split into feature and label
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['classification']
    # Split train test datasets
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=1
    )
    # Train and evaluate the model
    model = SVC(C=hp_c).fit(x_train.values, y_train.values)
    score_train = np.mean(cross_val_score(model, x_train.values, y_train.values, cv=5))
    score_test = model.score(x_test.values, y_test)
    conf_matrix = confusion_matrix(y_test, model.predict(x_test.values))
    # Pickle the model
    mod_path = os.path.abspath('') + '\\model\\svc.pickle'
    pd.to_pickle(model, mod_path)

    # Save confusion matrix as heatmap
    df_cm = pd.DataFrame(
        conf_matrix,
        index=[
            'Iris-setosa',
            'Iris-versicolor',
            'Iris-virginica'
        ],
        columns=[
            'Iris-setosa',
            'Iris-versicolor',
            'Iris-virginica'
        ]
    )
    '''
    fig = sns.heatmap(
        df_cm,
        cmap='Blues',
        square=True, linewidths=1,
        linecolor='black',
        annot=True
    )
    img = fig.get_figure()
    fig_path = os.path.abspath('') + '\\static\\images\\cm'
    img.savefig(fig_path, dpi=400)
    '''

    # Write into database
    TrainResult.objects.create(
        model_name='SCV',
        hp_c=hp_c,
        train_score=score_train,
        test_score=score_test,
        train_date=timezone.now()
    )

    return JsonResponse(
        {
            'hp_c': hp_c,
            'train_score': score_train,
            'test_score': score_test,
        },
        safe=False
    )


def view_models(request):
    # Submit training and show all
    data = {"dataset": TrainResult.objects.all()}
    return render(request, "train/models.html", data)
