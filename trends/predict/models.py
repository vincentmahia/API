from django.db import models

# Create your models here.
class TrainModel(models.Model):
    model_file = models.FileField(upload_to="models/")
    created_at = models.DateTimeField(auto_now_add=True)
    target_column = models.CharField(max_length=200, null=True)