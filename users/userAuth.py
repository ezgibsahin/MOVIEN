import pandas as pd
import time
import csv
import os
from django.contrib.auth.models import User
from users.models import CustomUser
from csv import reader

with open('pro.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    for line in csv_reader:
        user = User.objects.create_user(line[4],password=line[5])
        user.save()

with open('pro.csv', 'r') as read_obj:
    csv_reader = reader(read_obj)
    for line in csv_reader:
        user = User.objects.get(username=line[4])
        custom = CustomUser(user=user,pro_id=line[0], age=line[1],gender=line[2],occupation=line[3])
        custom.save()