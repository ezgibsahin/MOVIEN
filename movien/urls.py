"""movien URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import path, include
from users import views as user_views
from django.conf import settings
from django.conf.urls.static import static
from movies import views as movie_views
from rating import views as rating_views
from Recommender import views as recommend_views


urlpatterns = [
    path('',include('organisation.urls')),
    path('register/',user_views.register, name='register'),
    path('login/',auth_views.LoginView.as_view(template_name='users/login.html'), name='login'),
    path('logout/',auth_views.LogoutView.as_view(template_name='users/logout.html'), name='logout'),
    path('admin/', admin.site.urls),
    path('movies/',movie_views.movies,name='allMovies'),
    path('uploadCSV/',movie_views.movie_upload,name='uploadCSV'),
    path('uploadRating/',rating_views.rating_upload,name='uploadRating'),
    path('movieDetail/<pk>/',movie_views.movie_detail,name='movieDetail'),
    path('recommended/',recommend_views.hybrid_recommendation,name='recommendedHybrid'),
    path('userUpload/',user_views.userProCsv,name='userUpload'),
    path('watchlist/',user_views.display_watchlist,name='watchlist'),
    path('bucketlist/',user_views.display_bucketlist,name='bucketlist'),
    path('addWatchlistUser/',movie_views.watchlist_add_user,name='watchlistExists'),
    path('', include('pwa.urls'))
    #path('profile/', user_views.profile, name='profile'),

]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
