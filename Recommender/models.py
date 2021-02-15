from django.db import models
from django.contrib.auth.models import User
from movies.models import Movies


class Recommend(models.Model):
    recommend_id = models.IntegerField(primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    moviesToReturn = models.ManyToManyField(Movies)

    def __str__(self):
        return self.moviesToReturn
