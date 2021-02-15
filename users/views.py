"""
from django.contrib.auth import authenticate
from django.shortcuts import render, redirect
from django.contrib import messages
#from .forms import UserRegisterForm, UserUpdateForm, ProfileUpdateForm,ProfileForm
from django.contrib.auth.decorators import login_required
import csv, io
from django.contrib import messages
from django.contrib.auth.decorators import permission_required
from .models import User
"""
from typing import re

from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth.decorators import login_required, permission_required
# from .forms import UserRegisterForm, UserUpdateForm, ProfileUpdateForm
from django.urls import reverse

from .forms import ExtendedUserCreationForm, CustomUserForm
from django.contrib.auth import authenticate

from django.contrib.auth.models import User
from users.models import CustomUser, WatchList, BucketList
from csv import reader
import csv, io
import sys
import urllib
from csv import reader
import os.path
from django.dispatch import receiver
from django.db.models.signals import post_save
from django.contrib.auth.models import User
from movies.models import Movies
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger


def register(request):
    if request.method == 'POST':
        form = ExtendedUserCreationForm(request.POST)
        custom_form = CustomUserForm(request.POST)
        if form.is_valid() and custom_form.is_valid():
            user = form.save()
            custom = custom_form.save(commit=False)
            custom.user = user

            custom.save()

            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            messages.success(request, f'Account created for {username}! You can log in now!')
            return redirect('login')
    else:
        form = ExtendedUserCreationForm()
        custom_form = CustomUserForm()

    return render(request, 'users/register.html', {'form': form, 'custom_form': custom_form})


@login_required
def custom(request):
    return render(request, 'users/custom.html')


def userProCsv(request):
    with open('pro.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        for line in csv_reader:
            user = User.objects.create_user(line[4], password=line[5])
            user.save()

    with open('pro.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        for line in csv_reader:
            user = User.objects.get(username=line[4])
            custom = CustomUser(user=user, pro_id=line[0], age=line[1], gender=line[2], occupation=line[3])
            custom.save()

    template = 'users/userUpload.html'
    return render(request, template)


@login_required
def display_watchlist(request):
    watchlists = WatchList.objects.all()
    movies = Movies.objects.all()
    watchlist_user = watchlists.filter(user=request.user)
    watchlist = watchlist_user.values('movie_list')
    print(type(watchlist))
    watchlist_not_parsed = str(watchlist)
    watchlist_parsed = watchlist_not_parsed.split(",")

    watchlist_movie_ids = []
    template = 'users/watchlist.html'

    for i in watchlist_parsed:
        non = ''.join(c for c in i if c.isdigit())
        watchlist_movie_ids.append(non)

    print(watchlist_movie_ids)
    # proper display
    watchlist_movies = []

    if watchlist_movie_ids[0] == '':
        context = {
            'title': 'Watch List'
        }
    else:
        for i in watchlist_movie_ids:
            for movie in movies:
                if movie.pk == int(i):
                    watchlist_movies.append(movie)
        print(len(watchlist_movies))

        page = request.GET.get('page', 1)
        paginator = Paginator(watchlist_movies, 15)
        try:
            watchlist_movies = paginator.page(page)
        except PageNotAnInteger:
            watchlist_movies = paginator.page(1)
        except EmptyPage:
            watchlist_movies = paginator.page(paginator.num_pages)

        context = {
            'title': 'Watch List',
            'watchlist_movies': watchlist_movies
        }

    return render(request, template, context)


@login_required
def display_bucketlist(request):
    bucketlists = BucketList.objects.all()
    movies = Movies.objects.all()
    bucketlist_user = bucketlists.filter(user=request.user)
    bucketlist = bucketlist_user.values('movie_list')
    print(type(bucketlist))
    bucketlist_not_parsed = str(bucketlist)
    bucketlist_parsed = bucketlist_not_parsed.split(",")
    bucketlist_movie_ids = []
    template = 'users/bucketlist.html'

    for i in bucketlist_parsed:
        non = ''.join(c for c in i if c.isdigit())
        bucketlist_movie_ids.append(non)

    print(bucketlist_movie_ids)
    # proper display
    bucketlist_movies = []

    if bucketlist_movie_ids[0] == '':
        context = {
            'title': 'Bucket List'
        }
    else:
        for i in bucketlist_movie_ids:
            for movie in movies:
                if movie.pk == int(i):
                    bucketlist_movies.append(movie)
        print(len(bucketlist_movies))

        page = request.GET.get('page', 1)
        paginator = Paginator(bucketlist_movies, 15)
        try:
            bucketlist_movies = paginator.page(page)
        except PageNotAnInteger:
            bucketlist_movies = paginator.page(1)
        except EmptyPage:
            bucketlist_movies = paginator.page(paginator.num_pages)

        context = {
            'title': 'Bucketlist',
            'bucketlist_movies': bucketlist_movies
        }

    return render(request, template, context)


@permission_required('admin.can_add_log_entry')
def user_upload(request):
    template = "movies/movie_upload.html"
    # data = Movies.objects.all()
    prompt = {
        'order': 'Order of CSV should be user_id,age,gender,occupation,generation,email,password'
    }
    if request.method == "GET":
        return render(request, template, prompt)
    csv_file = request.FILES['file']
    if not csv_file.name.endswith('.csv'):
        messages.error(request, 'This is not a CSV file.')
    data_set = csv_file.read().decode('UTF-8')
    io_string = io.StringIO(data_set)
    next(io_string)
    for column in csv.reader(io_string, delimiter=',', quotechar='"'):
        _, created = User.objects.update_or_create(
            user_id=column[0],
            age=column[1],
            gender=column[2],
            occupation=column[3],
            generation=column[4],
            email=column[5],
            password=column[6]
        )
        context = {}
        return render(request, template, context)
