from django.urls import path
from . import views

app_name = 'train'

urlpatterns = [
    path('', views.train, name='train_page'),
    path('train/', views.train_svc, name='submit_training'),
    path('models/', views.view_models, name='models'),
]
