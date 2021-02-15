from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone

class Movies(models.Model):
	movie_id = models.IntegerField(primary_key = True)
	movie_title = models.CharField(max_length=100)
	imdb_url = models.CharField(max_length=100)
	decade = models.CharField(max_length=1000)
	genres = models.CharField(max_length=1000)
	avg_rating = models.FloatField(default=5)
	poster = models.CharField(max_length=1000)
	cast = models.CharField(max_length=10000,default="No cast found.")
	director =models.CharField(max_length=100,default="No director found.")
	description =models.CharField(max_length=10000,default="No description found.")
	keyword = models.CharField(max_length=10000,default="No keyword found.")

	def __str__(self):
		return self.movie_title


class Review(models.Model):
    movie = models.ForeignKey(Movies, related_name='reviews', on_delete=models.CASCADE)
    user = models.ForeignKey(User, related_name='reviews', on_delete=models.CASCADE)
    content = models.TextField(blank=True, null=True)
    stars = models.FloatField()
    date_added = models.DateTimeField(auto_now_add=True)