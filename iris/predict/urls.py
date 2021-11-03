from django.urls import path
from . import views

app_name = 'predict'

urlpatterns = [
    path('', views.predict, name='predict_page'),
    path('submit/', views.predict_chances, name='submit_prediction'),
    path('results/', views.view_results, name='results'),
]

