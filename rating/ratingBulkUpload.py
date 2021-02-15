from django.contrib.auth.models import User
from rating.models import Rating
from csv import reader

with open('output_6.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    for line in csv_reader:
        rating = Rating.objects.create_user(user_id =line[0],movie_id=line[1],rating=line[2])
        rating.save()

"""
with open('pro.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    for line in csv_reader:
        user = User.objects.get(username=line[4])
        custom = CustomUser(user=user,pro_id=line[0], age=line[1],gender=line[2],occupation=line[3])
        custom.save()
"""
