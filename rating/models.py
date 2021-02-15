from django.db import models
from django.contrib.auth.models import User
from movies.models import Movies


class Rating(models.Model):
    movie_id = models.IntegerField()#Foreign Keylemeli miyiz? idk
    user_id = models.IntegerField()#Foreign Keylemeli miyiz? idk
    rating = models.FloatField()
    #user = models.ForeignKey(User,on_delete=models.CASCADE)
    #movie = models.ForeignKey(Movies,on_delete=models.CASCADE)

    def __str__(self):
        return str(self.rating)
