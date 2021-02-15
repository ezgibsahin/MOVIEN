from django.shortcuts import render
from django.views.generic import (
    ListView,
    DetailView,
    CreateView
)
from movies.models import Movies

from django.db.models import Q
from operator import attrgetter
from Recommender.views import recommend_user_based_result


def homePageView(request):
    movies = Movies.objects.all()
    user_based_recommendation_list = recommend_user_based_result(request)
    context = {
        'movies': movies[:5],
        'user_based_recommendation_list': user_based_recommendation_list
    }
    # print(context)
    return render(request, 'organisation/home.html', context)


def about(request):
    return render(request, 'organisation/about.html', {'title': 'About'})

def contact(request):
    return render(request, 'organisation/contact.html', {'title': 'Contact'})


def search(request):
    # 'title': 'Search'
    context = {}

    query = ""
    if request.GET:
        query = request.GET['search']
        context['query'] = str(query)

        results = sorted(search_bar(query), key=attrgetter('movie_id'), reverse=False)
        context['results'] = results
    return render(request, 'organisation/search.html', context)


def search_bar(query=None):
    querySet = []
    queries = query.split(" ")
    for q in queries:
        movies = Movies.objects.filter(
            Q(movie_title__icontains=q)
        ).distinct()

        for movie in movies:
            querySet.append(movie)
    return list(set(querySet))
