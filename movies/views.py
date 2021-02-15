import csv
import io

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.decorators import permission_required
from django.http import Http404

from users.models import WatchList, BucketList
from .models import Movies
from rating.models import Rating
from django.views.generic import DetailView
from django.shortcuts import render, get_object_or_404
from users.models import User, CustomUser
from Recommender.views import knn
from .models import Movies, Review
from django.contrib.auth.models import User
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger


@login_required()
def movie_detail(request, pk):
    movies = Movies.objects.all()
    movie = movies.get(movie_id=pk)
    print("movie detail print done" + pk)
    if request.method == "POST":
        stars = request.POST.get('stars', 5)
        content = request.POST.get('content', '')
        if Review.objects.filter(movie=movie, user=request.user):
            Rating.objects.filter(movie_id=pk, user_id=request.user.id).delete()
            Review.objects.filter(movie=movie, user=request.user).delete()
            review = Review.objects.create(movie=movie, user=request.user, stars=stars, content=content)
            user_rating = Rating.objects.create(movie_id=pk, user_id=request.user.id, rating=stars)
            print('review already exists, so it is deleted and updated')
        else:
            review = Review.objects.create(movie=movie, user=request.user, stars=stars, content=content)
            user_rating = Rating.objects.create(movie_id=pk, user_id=request.user.id, rating=stars)

        theMovie = movies.get(movie_id=pk)
        allGivenRatingsForTheMovie = Rating.objects.filter(movie_id=pk)
        total = 0
        for r in allGivenRatingsForTheMovie:
            total = total + r.rating

        newAvg = total / allGivenRatingsForTheMovie.count()

        theMovie.avg_rating = newAvg
        theMovie.save()

        if 'watchlist' in request.POST:
            print('watchlist')
            watchlistAdd(request, pk)
        elif 'bucketlist' in request.POST:
            print('bucketlist')
            bucketlist_add(request, pk)

    movie_title = movie.movie_title  # Movie title is required for KNN recommendation.
    knn_recommendation_list = knn(request, movie_title)
    # print(knn_recommendation_list)
    allReviewsForTheMovie = Review.objects.filter(movie=movie)
    if allReviewsForTheMovie.count() > 1:
        allReviewsForTheMovie = allReviewsForTheMovie[:2]

    context = {
        'movie': movie,
        'knn_recommendation_list': knn_recommendation_list,
        'allReviewsForTheMovie': allReviewsForTheMovie
    }
    template = 'movies/movie_detail.html'
    return render(request, template, context)


@login_required
def movies(request):
    movie_list = Movies.objects.all()
    page = request.GET.get('page', 1)

    paginator = Paginator(movie_list, 15)
    try:
        movies = paginator.page(page)
    except PageNotAnInteger:
        movies = paginator.page(1)
    except EmptyPage:
        movies = paginator.page(paginator.num_pages)

    context = {
        'title': 'Movie',
        'movies': movies
    }
    return render(request, 'movies/movies.html', context)


@permission_required('admin.can_add_log_entry')
def movie_upload(request):
    template = "movies/movie_upload.html"
    # data = Movies.objects.all()
    prompt = {
        'order': 'Order of CSV should be movie_id, movie_title, imbd_url, decades,genres,poster,cast, director,description,keyword'
    }
    if request.method == "GET":
        return render(request, template, prompt)

    csv_file = request.FILES['file']

    if not csv_file.name.endswith('.csv'):
        messages.error(request, 'This is not a CSV file.')

    data_set = csv_file.read().decode('UTF-8')
    # print(data_set)
    # lines = data_set.split("\n")
    io_string = io.StringIO(data_set)
    next(io_string)

    for column in csv.reader(io_string, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL):
        print(column)
        _, created = Movies.objects.update_or_create(
            movie_id=column[0],
            movie_title=column[1],
            # release_date = column[2],
            imdb_url=column[2],
            decade=column[3],
            genres=column[4],
            avg_rating=column[5],
            poster=column[6],
            cast=column[7],
            director=column[8],
            description=column[9],
            keyword=column[10]

        )

    context = {
        'movies': Movies.objects.all()
    }
    print(Movies.objects.all())
    return render(request, template, context)


@login_required
def watchlistAdd(request, movie_id):
    watchlist_user = WatchList.objects.filter(user=request.user)
    if watchlist_user.exists():
        user_watch = WatchList.objects.get(user=request.user)
        list = user_watch.movie_list
        if list.__contains__(movie_id):
            pass
        else:
            if str(list) == "":
                user_watch.movie_list = str(list) + str(movie_id)
            else:
                user_watch.movie_list = str(list) + "," + str(movie_id)
                WatchList.objects.filter(user=request.user).update(movie_list=user_watch.movie_list)
    else:
        # print("slm")
        instance = WatchList.objects.create(user=request.user, movie_list=movie_id)
        instance.save()

    movies = Movies.objects.all()
    template = "movies/movie_detail.html"
    movie = movies.get(movie_id=movie_id)
    context = {
        'movie': movie
    }
    return render(request, template, context)


@login_required()
def bucketlist_add(request, movie_id):
    bucketlist_user = BucketList.objects.filter(user=request.user)
    if bucketlist_user.exists():
        print(BucketList.objects.get(user=request.user))
        user_bucket = BucketList.objects.get(user=request.user)
        list = user_bucket.movie_list
        if list.__contains__(movie_id):
            pass
        else:
            if str(list) == "":
                user_bucket.movie_list = str(list) + str(movie_id)
            else:
                user_bucket.movie_list = str(list) + "," + str(movie_id)
                # print(user_watch.movie_list)
                BucketList.objects.filter(user=request.user).update(movie_list=user_bucket.movie_list)
    else:
        instance = BucketList.objects.create(user=request.user, movie_list=movie_id)
        # list = User.objects.filter(movie_list=movie_id)
        # instance = BucketList.objects.create(user= request.user)
        # instance.movie_list.set(list)
        instance.save()
    movies = Movies.objects.all()
    template = "movies/movie_detail.html"
    movie = movies.get(movie_id=movie_id)
    context = {
        'movie': movie
    }
    return render(request, template, context)


def watchlist_add_user(request):
    users_distinct = Rating.objects.all().values('user_id').distinct()
    # print(users_distinct)
    print("movies of 2")

    for i in range(1, 943):
        print(i)
        allusers = User.objects.all()
        un = "username" + str(i)
        currentuser = allusers.get(username=un)
        movies_distinct = Rating.objects.filter(user_id=i).values('movie_id')
        movies_of_i = ','.join([str(j) for j in Rating.objects.filter(user_id=i).values('movie_id')])
        movie_id_list = ""

        qset_splitted = movies_of_i.split(",")
        for line in qset_splitted:
            line_split = line.split(":")
            id = ''.join(i for i in str(line_split[1]) if i.isdigit())
            print(id)
            if movie_id_list == "":
                movie_id_list += str(id)
            else:
                movie_id_list += "," + str(id)
        print(movie_id_list)
        instance = WatchList.objects.create(user=currentuser, movie_list=movie_id_list)
        instance.save()

    template = "movies/watchlist_add_user.html"
    return render(request, template)