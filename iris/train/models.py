from django.db import models


# Create your models here.
class TrainResult(models.Model):

    model_name = models.CharField(max_length=30)
    hp_c = models.FloatField(default=1e-3)
    train_score = models.FloatField()
    test_score = models.FloatField()
    train_date = models.DateTimeField('date trained')

    def __str__(self):
        return f"{self.model_name} with {self.test_score} accuracy"
