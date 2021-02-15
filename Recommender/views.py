# import necessary libraries
import random

from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import Http404
from django.shortcuts import render, redirect
import numpy as np
import pandas as pd
# Models
from pandas import DataFrame

from movies.models import Movies
from rating.models import Rating
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
# libraries used for KNN implementation
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from fuzzywuzzy import process
from users.models import CustomUser

# Content-Based Recommendation (Description Only) Term Frequency-Inverse Document Frequency Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# Content-Based Recommendation Linear Kernel for Cosine Similarity
from sklearn.metrics.pairwise import linear_kernel
# Content-Based Recommendation Parsing the stringified features into their corresponding python objects
from ast import literal_eval
# Content-Based Recommendation CountVectorizer  for Creation of the Count Matrix
from sklearn.feature_extraction.text import CountVectorizer
# Content-Based Recommendation Computation of the Cosine Similarity Matrix Based
from sklearn.metrics.pairwise import cosine_similarity
from django.db.models import Q
from sklearn.model_selection import train_test_split
from sklearn import metrics


def recommend(request):
    if not request.user.is_authenticated:
        return redirect("login")
    if not request.user.is_active:
        raise Http404


@login_required
def knn(request, movie_title):
    data_frame_of_rating_records = pd.DataFrame.from_records(
        Rating.objects.all().values_list('user_id', 'movie_id', 'rating'),
        columns=['user_id', 'movie_id', 'rating'])

    is_duplicated = np.where(data_frame_of_rating_records.index.duplicated())
    num_users = len(data_frame_of_rating_records.user_id.unique())
    num_items = len(data_frame_of_rating_records.movie_id.unique())

    df_ratings_cnt_tmp = pd.DataFrame(data_frame_of_rating_records.groupby('rating').size(), columns=['count'])
    total_cnt = num_users * num_items
    rating_zero_cnt = total_cnt - data_frame_of_rating_records.shape[0]
    full_df_cnt = df_ratings_cnt_tmp.append(
        pd.DataFrame({'count': rating_zero_cnt}, index=[0.0]),
        verify_integrity=True,
    ).sort_index()

    full_df_cnt['log_count'] = np.log(full_df_cnt['count'])

    full_df_movies_cnt = pd.DataFrame(data_frame_of_rating_records.groupby('movie_id').size(), columns=['count'])

    movies_users = data_frame_of_rating_records.reset_index().pivot_table(index='movie_id', columns='user_id',
                                                                          values='rating').fillna(0)
    mat_movies_users = csr_matrix(movies_users.values)
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    model_knn.fit(mat_movies_users)

    movie_list_recommended_knn = recommenderKNN(movie_title, mat_movies_users, model_knn, 20)
    movie_list_not_parsed = str(movie_list_recommended_knn)
    movie_list_parsed = movie_list_not_parsed.split('\n')
    del movie_list_parsed[0]
    id_list_of_recommended_movies = []
    for movie in movie_list_parsed:
        split_movies = movie.split(" ", 1)
        title = split_movies[1].strip()
        if split_movies[0] != "Name:":
            converted_num = int(split_movies[0]) + 1
            id_list_of_recommended_movies.append(str(converted_num))

    movies = Movies.objects.all()
    recommend_knn_list = []
    for i in id_list_of_recommended_movies:
        for movie in movies:
            if movie.pk == int(i):
                recommend_knn_list.append(movie)
    return recommend_knn_list


def recommenderKNN(movie_name, data, model, n_recommendations):
    data_frame_of_movie_records = pd.DataFrame.from_records(
        Movies.objects.all().values_list('movie_id', 'movie_title', 'imdb_url', 'decade', 'genres', 'avg_rating',
                                         'poster', 'cast', 'director', 'description', 'keyword'),
        columns=['movie_id', 'movie_title', 'imdb_url', 'decade', 'genres', 'avg_rating', 'poster', 'cast', 'director',
                 'description', 'keyword'])
    model.fit(data)
    idx = process.extractOne(movie_name, data_frame_of_movie_records['movie_title'])[2]
    distances, indices = model.kneighbors(data[idx], n_neighbors=n_recommendations)
    movie_list = []
    for i in indices:
        movie_list.append((data_frame_of_movie_records['movie_title'][i].where(i != idx)))
    return movie_list


def train_test_knn(request):
    data_frame_of_rating_records = pd.DataFrame.from_records(
        Rating.objects.all().values_list('user_id', 'movie_id', 'rating'),
        columns=['user_id', 'movie_id', 'rating'])
    # splited_list = [i.split(',') for i in data_frame_of_rating_records]
    X = data_frame_of_rating_records.drop('rating', axis=1)
    y = data_frame_of_rating_records['rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1)
    movies_users = data_frame_of_rating_records.reset_index().pivot_table(index='movie_id', columns='user_id',
                                                                          values='rating').fillna(0)
    mat_movies_users = csr_matrix(movies_users.values)
    model_knn = KNeighborsClassifier(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    model_knn.fit(X_train, y_train)
    y_pred = model_knn.predict(X_test)
    # converted_list = [int(''.join(i)) for i in splited_list]
    # X = data_frame_of_rating_records.iloc[:, :2]
    # y = data_frame_of_rating_records.iloc[:, 2]
    # X = data_frame_of_rating_records.drop('rating', axis=1)
    # y = data_frame_of_rating_records['rating']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1)
    # model_knn = KNeighborsClassifier(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    # model_knn.fit(X_train, y_train)
    # y_pred = model_knn.predict(X_test)

    print("Precision : ", metrics.precision_score(y_test, y_pred, average='weighted'))
    print("Accuracy : ", metrics.accuracy_score(y_test, y_pred))
    print("Recall : ", metrics.recall_score(y_test, y_pred, average='macro'))
    print("F1 Score : ", metrics.f1_score(y_test, y_pred, average='weighted'))
    accuracy = metrics.accuracy_score(y_test, y_pred)
    context = {
        'accuracy': accuracy
    }
    template = 'recommender/recommendation.html'
    return render(request, template, context)


# A function to predict the rating a user would give to a movie (which he has not rated yet) based on the weighted
# average of all other users (weighted because users who are more similar to a user x will be given more weight than the
# users who are not so similar to user x)
def predict_fast_simple(rating, similarity, kind='user'):
    #
    return similarity.dot(rating) / np.array([np.abs(similarity).sum(axis=1)]).T
    # elif kind == 'item':
    #    return rating.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])


"""
def train_test_split(ratings):
    test = np.zeros(ratings.shape)  # dim(test) = (943, 1682)
    train = ratings.copy()  # dim(train) = (943, 1682)

    # for each user
    for user in range(ratings.shape[0]):  # loop from 0 to (943 - 1)

        # Select the indices of 10 MovieIDs which have been rated by the user (nonzero rating)
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], size=10, replace=False)

        # In the train set, fill those indices with 0 (since we have to predict these and then compare with the test
        # set)
        train[user, test_ratings] = 0.

        # In the test set, fill those indices with the rating given by the user (this will be our actual value which
        # will be compared to the predicted value)
        test[user, test_ratings] = ratings[user, test_ratings]

    # Ensure that test and train sets are truly disjoint
    assert (np.all((train * test) == 0))
    return train, test

"""


# A function to split the data into train set and test set (to be used to calculate the accuracy of our model)
def recommend_user_based(UserID):
    data_frame_movie_records = pd.DataFrame.from_records(
        Movies.objects.all().values_list('movie_id', 'movie_title', 'imdb_url', 'decade', 'genres', 'avg_rating',
                                         'poster', 'cast', 'director', 'description', 'keyword'),
        columns=['movie_id', 'movie_title', 'imdb_url', 'decade', 'genres', 'avg_rating', 'poster', 'cast',
                 'director',
                 'description', 'keyword'])


    data_frame_rating_records = pd.DataFrame.from_records(
        Rating.objects.all().values_list('user_id', 'movie_id', 'rating'),
        columns=['user_id', 'movie_id', 'rating'])

    # Create a pivot table with UserID as rows and MovieID as columns and each cell (x, y) represents the rating
    # given by a user x to a movie y.
    df_new = data_frame_rating_records.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(
        0)

    # Convert the pivot table to an array
    ratings = np.array(df_new)

    # Get a sparse matrix which only consists of the non zero data (this is done to avoid unnecessary processing)
    matrix = csr_matrix(df_new.values)

    # Calculate the similarity between each user using the cosine similarity matrix
    cosine_sim = cosine_similarity(matrix, matrix)  # dim(cosine_sim) = (943, 943),
    """
    # Split the data
        train, test = train_test_split(ratings)

        # Calculate cosine similarity for the train set
        train_cosine_sim = cosine_similarity(train, train)

        # Make prediction on the train set
        user_prediction = predict_fast_simple(train, train_cosine_sim, 'user')
    """

    # Now make predictions using the actual DataSet
    dff = pd.DataFrame(predict_fast_simple(ratings, cosine_sim, 'user'))

    dff.columns += 1
    dff.index += 1
    dff.index.rename('UserID', inplace=True)

    result = {}
    for j in df_new.columns:
        if df_new.iloc[UserID - 1][j] == 0 and dff.iloc[UserID - 1][j] != 0:
            # The dictionary result contains keys as the movies which were not initially rated by the user (whose values
            # we predicted) and values as the corresponding predicted rating
            result.__setitem__(j, dff.iloc[UserID - 1][j])

    # Sort the dictionary wrt to values i.e. in decreasing order of the predicted ratings
    h = sorted(result.items(), key=lambda x: x[1], reverse=True)

    # Consider only the top 5 movies with the highest predicted ratings
    h = h[:5]

    # Take the MovieIDs of the top 5 movies
    result1 = [i[0] for i in h]
    # print(type(result1))
    print(result1)
    print(data_frame_movie_records['movie_title'].iloc[result1].drop_duplicates())

    # Return the Name of the movies
    print(type(data_frame_movie_records.iloc[result1][['movie_id', 'movie_title']]))
    return data_frame_movie_records.iloc[result1][['movie_id', 'movie_title']]



def recommendation_user_profile(request):
    data_frame_movie_records = pd.DataFrame.from_records(
        Movies.objects.all().values_list('movie_id', 'movie_title', 'imdb_url', 'decade', 'genres', 'avg_rating',
                                         'poster', 'cast', 'director', 'description', 'keyword'),
        columns=['movie_id', 'movie_title', 'imdb_url', 'decade', 'genres', 'avg_rating', 'poster',
                 'cast', 'director', 'description', 'keyword'])

    data_frame_rating_records = pd.DataFrame.from_records(
        Rating.objects.all().values_list('user_id', 'movie_id', 'rating'),
        columns=['user_id', 'movie_id', 'rating'])

    pd_rating_movie = pd.merge(data_frame_movie_records, data_frame_rating_records, on="movie_id")

    data_frame_custom_user = pd.DataFrame.from_records(
        CustomUser.objects.all().values_list('user', 'pro_id', 'age', 'gender', 'occupation'),
        columns=['user_id', 'pro_id', 'age', 'gender', 'occupation'])

    pd_rating_movie_user = pd.merge(pd_rating_movie, data_frame_custom_user, on="user_id", how='left')
    #pd_rating_movie_user.to_csv(r'ratingmovieuser.csv', index=False)
    current_user = User.objects.get(username=request.user.username)
    custom_user = CustomUser.objects.get(user=current_user)
    custom_user_age = int(custom_user.age)
    custom_user_gender = custom_user.gender.upper()
    custom_user_occupation = custom_user.occupation.lower()
    movie_list = []
    if (custom_user_age <= 19) and custom_user_age >= 10:
        if custom_user_gender == 'M':
            movie_list = pd_rating_movie_user.loc[
                (pd_rating_movie_user['age'] <= '19') & (pd_rating_movie_user['age'] >= '10') & (
                        pd_rating_movie_user['gender'] == 'M') & (pd_rating_movie_user['rating'] > 3)]
            movies = movie_list['movie_id'].unique()
            movies_random = (random.choices(movies, k=5))
        elif custom_user.gender == 'F':
            movie_list = pd_rating_movie_user.loc[
                (pd_rating_movie_user['age'] <= '19') & (pd_rating_movie_user['age'] >= '10') & (
                        pd_rating_movie_user['gender'] == 'F') & (pd_rating_movie_user['rating'] > 3)]
            movies = movie_list['movie_id'].unique()
            movies_random = (random.choices(movies, k=5))
        else:
            movie_list = pd_rating_movie_user.loc[
                (pd_rating_movie_user['age'] <= '19') & (pd_rating_movie_user['age'] >= '10') & (
                        pd_rating_movie_user['gender'] == 'F') & (pd_rating_movie_user['rating'] > 3)]
            movies = movie_list['movie_id'].unique()
            movies_random = (random.choices(movies, k=5))
    elif (custom_user_age <= 29) and custom_user_age >= 20:
        if custom_user_gender == 'M':
            movie_list = pd_rating_movie_user.loc[
                (pd_rating_movie_user['age'] <= '29') & (pd_rating_movie_user['age'] >= '20') & (
                        pd_rating_movie_user['gender'] == 'M') & (pd_rating_movie_user['rating'] > 3)]
            movies = movie_list['movie_id'].unique()
            movies_random = (random.choices(movies, k=5))
        elif custom_user.gender == 'F':
            movie_list = pd_rating_movie_user.loc[
                (pd_rating_movie_user['age'] <= '29') & (pd_rating_movie_user['age'] >= '20') & (
                        pd_rating_movie_user['gender'] == 'F') & (pd_rating_movie_user['rating'] > 3)]
            movies = movie_list['movie_id'].unique()
            movies_random = (random.choices(movies, k=5))
        else:
            movie_list = pd_rating_movie_user.loc[
                (pd_rating_movie_user['age'] <= '29') & (pd_rating_movie_user['age'] >= '20') & (
                        pd_rating_movie_user['gender'] == 'F') & (pd_rating_movie_user['rating'] > 3)]
            movies = movie_list['movie_id'].unique()
            movies_random = (random.choices(movies, k=5))
    elif (custom_user_age <= 39) and custom_user_age >= 30:
        if custom_user_gender == 'M':
            movie_list = pd_rating_movie_user.loc[
                (pd_rating_movie_user['age'] <= '39') & (pd_rating_movie_user['age'] >= '30') & (
                        pd_rating_movie_user['gender'] == 'M') & (pd_rating_movie_user['rating'] > 3)]
            movies = movie_list['movie_id'].unique()
            movies_random = (random.choices(movies, k=5))
        elif custom_user.gender == 'F':
            movie_list = pd_rating_movie_user.loc[
                (pd_rating_movie_user['age'] <= '39') & (pd_rating_movie_user['age'] >= '30') & (
                        pd_rating_movie_user['gender'] == 'F') & (pd_rating_movie_user['rating'] > 3)]
            movies = movie_list['movie_id'].unique()
            movies_random = (random.choices(movies, k=5))
        else:
            movie_list = pd_rating_movie_user.loc[
                (pd_rating_movie_user['age'] <= '39') & (pd_rating_movie_user['age'] >= '30') & (
                        pd_rating_movie_user['gender'] == 'F') & (pd_rating_movie_user['rating'] > 3)]
            movies = movie_list['movie_id'].unique()
            movies_random = (random.choices(movies, k=5))
    elif (custom_user_age <= 49) and custom_user_age >= 40:
        if custom_user_gender == 'M':
            movie_list = pd_rating_movie_user.loc[
                (pd_rating_movie_user['age'] <= '49') & (pd_rating_movie_user['age'] >= '40') & (
                        pd_rating_movie_user['gender'] == 'M') & (pd_rating_movie_user['rating'] > 3)]
            movies = movie_list['movie_id'].unique()
            movies_random = (random.choices(movies, k=5))
        elif custom_user.gender == 'F':
            movie_list = pd_rating_movie_user.loc[
                (pd_rating_movie_user['age'] <= '49') & (pd_rating_movie_user['age'] >= '40') & (
                        pd_rating_movie_user['gender'] == 'F') & (pd_rating_movie_user['rating'] > 3)]
            movies = movie_list['movie_id'].unique()
            movies_random = (random.choices(movies, k=5))
        else:
            movie_list = pd_rating_movie_user.loc[
                (pd_rating_movie_user['age'] <= '49') & (pd_rating_movie_user['age'] >= '40') & (
                        pd_rating_movie_user['gender'] == 'F') & (pd_rating_movie_user['rating'] > 3)]
            movies = movie_list['movie_id'].unique()
            movies_random = (random.choices(movies, k=5))
    elif (custom_user_age <= 59) and custom_user_age >= 50:
        if custom_user_gender == 'M':
            movie_list = pd_rating_movie_user.loc[
                (pd_rating_movie_user['age'] <= '59') & (pd_rating_movie_user['age'] >= '50') & (
                        pd_rating_movie_user['gender'] == 'M') & (pd_rating_movie_user['rating'] > 3)]
            movies = movie_list['movie_id'].unique()
            movies_random = (random.choices(movies, k=5))
        elif custom_user.gender == 'F':
            movie_list = pd_rating_movie_user.loc[
                (pd_rating_movie_user['age'] <= '59') & (pd_rating_movie_user['age'] >= '50') & (
                        pd_rating_movie_user['gender'] == 'F') & (pd_rating_movie_user['rating'] > 3)]
            movies = movie_list['movie_id'].unique()
            movies_random = (random.choices(movies, k=5))
        else:
            movie_list = pd_rating_movie_user.loc[
                (pd_rating_movie_user['age'] <= '59') & (pd_rating_movie_user['age'] >= '50') & (
                        pd_rating_movie_user['gender'] == 'F') & (pd_rating_movie_user['rating'] > 3)]
            movies = movie_list['movie_id'].unique()
            movies_random = (random.choices(movies, k=5))
    elif (custom_user_age <= 69) and custom_user_age >= 60:
        if custom_user_gender == 'M':
            movie_list = pd_rating_movie_user.loc[
                (pd_rating_movie_user['age'] <= '69') & (pd_rating_movie_user['age'] >= '60') & (
                        pd_rating_movie_user['gender'] == 'M') & (pd_rating_movie_user['rating'] > 3)]
            movies = movie_list['movie_id'].unique()
            movies_random = (random.choices(movies, k=5))
        elif custom_user.gender == 'F':
            movie_list = pd_rating_movie_user.loc[
                (pd_rating_movie_user['age'] <= '69') & (pd_rating_movie_user['age'] >= '60') & (
                        pd_rating_movie_user['gender'] == 'F') & (pd_rating_movie_user['rating'] > 3)]
            movies = movie_list['movie_id'].unique()
            movies_random = (random.choices(movies, k=5))
        else:
            movie_list = pd_rating_movie_user.loc[
                (pd_rating_movie_user['age'] <= '69') & (pd_rating_movie_user['age'] >= '60') & (
                        pd_rating_movie_user['gender'] == 'F') & (pd_rating_movie_user['rating'] > 3)]
            movies = movie_list['movie_id'].unique()
            movies_random = (random.choices(movies, k=5))
    elif (custom_user_age <= 79) and custom_user_age >= 70:
        if custom_user_gender == 'M':
            movie_list = pd_rating_movie_user.loc[
                (pd_rating_movie_user['age'] <= '79') & (pd_rating_movie_user['age'] >= '70') & (
                        pd_rating_movie_user['gender'] == 'M') & (pd_rating_movie_user['rating'] > 3)]
            movies = movie_list['movie_id'].unique()
            movies_random = (random.choices(movies, k=5))
        elif custom_user.gender == 'F':
            movie_list = pd_rating_movie_user.loc[
                (pd_rating_movie_user['age'] <= '79') & (pd_rating_movie_user['age'] >= '70') & (
                        pd_rating_movie_user['gender'] == 'F') & (pd_rating_movie_user['rating'] > 3)]
            movies = movie_list['movie_id'].unique()
            movies_random = (random.choices(movies, k=5))
        else:
            movie_list = pd_rating_movie_user.loc[
                (pd_rating_movie_user['age'] <= '79') & (pd_rating_movie_user['age'] >= '70') & (
                        pd_rating_movie_user['gender'] == 'F') & (pd_rating_movie_user['rating'] > 3)]
            movies = movie_list['movie_id'].unique()
            movies_random = (random.choices(movies, k=5))
    else:  # Anyone older than 79
        if custom_user_gender == 'M':
            movie_list = pd_rating_movie_user.loc[
                (pd_rating_movie_user['age'] <= '79') & (pd_rating_movie_user['age'] >= '70') & (
                        pd_rating_movie_user['gender'] == 'M') & (pd_rating_movie_user['rating'] > 3)]
            movies = movie_list['movie_id'].unique()
            movies_random = (random.choices(movies, k=5))
        elif custom_user.gender == 'F':
            movie_list = pd_rating_movie_user.loc[
                (pd_rating_movie_user['age'] <= '79') & (pd_rating_movie_user['age'] >= '70') & (
                        pd_rating_movie_user['gender'] == 'F') & (pd_rating_movie_user['rating'] > 3)]
            movies = movie_list['movie_id'].unique()
            movies_random = (random.choices(movies, k=5))
        else:
            movie_list = pd_rating_movie_user.loc[
                (pd_rating_movie_user['age'] <= '79') & (pd_rating_movie_user['age'] >= '70') & (
                        pd_rating_movie_user['gender'] == 'F') & (pd_rating_movie_user['rating'] > 3)]
            movies = movie_list['movie_id'].unique()
            movies_random = (random.choices(movies, k=5))
    print(len(movies_random))
    print(movies_random)
    movie_indices = []
    for i in movies_random:
        movie_indices.append(i - 1)
    print(movie_indices)
    # movies_to_return = data_frame_movie_records.iloc[movie_indices][['movie_id', 'movie_title']]
    print(data_frame_movie_records.iloc[movie_indices][['movie_id', 'movie_title']])
    return data_frame_movie_records.iloc[movie_indices][['movie_id', 'movie_title']]
    # return movies


@login_required
def recommend_user_based_result(request):
    movies = Movies.objects.all()
    recommend_user_based_result_list = []
    existing_ratings = Rating.objects.all()
    existing_users = existing_ratings.filter().values('user_id').distinct()
    user_exists = Rating.objects.filter(user_id=request.user.id)
    if user_exists.count() == 0:
        result = recommendation_user_profile(request)
        titles_wo_rating = result['movie_title'].values.tolist()
        for title_wo_rating in titles_wo_rating:
            for movie in movies:
                if movie.movie_title == title_wo_rating:
                    recommend_user_based_result_list.append(movie)
    else:
        result_rating = recommend_user_based(request.user.id)
        titles = result_rating['movie_title'].values.tolist()
        for title in titles:
            for movie in movies:
                if movie.movie_title == title:
                    recommend_user_based_result_list.append(movie)
    return recommend_user_based_result_list


# only using movies' descriptions
def description_based_recommendation(title):
    data_frame_movie_records = pd.DataFrame.from_records(
        Movies.objects.all().values_list('movie_id', 'movie_title', 'imdb_url', 'decade', 'genres', 'avg_rating',
                                         'poster', 'cast', 'director', 'description', 'keyword'),
        columns=['movie_id', 'movie_title', 'imdb_url', 'decade', 'genres', 'avg_rating', 'poster', 'cast', 'director',
                 'description', 'keyword'])
    descriptions = data_frame_movie_records['description']

    # Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')  # This was used first which is correct.
    # tfidf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0,stop_words='english') #This is an alternative of the above.

    # Replace NaN with an empty string
    # descriptions = descriptions.fillna('')

    # Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(descriptions)
    # print(tfidf_matrix)

    # Output the shape of tfidf_matrix
    tfidf_matrix.shape
    # print(tfidf_matrix.shape)

    # all the words in descriptions
    feature_names = tfidf.get_feature_names()
    # print(feature_names)

    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    print(cosine_sim.shape)
    print(cosine_sim[2])

    # Construct a reverse map of indices and movie titles
    indices = pd.Series(data_frame_movie_records.index, index=data_frame_movie_records['movie_title'])
    print(indices)  # indices can be incremented 1

    # Get the index of the movie that matches the title
    idx = indices[title]
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return data_frame_movie_records['movie_title'].iloc[movie_indices]


def description_recommendations(request):
    title = 'Godfather, The (1972)'
    result = description_based_recommendation(title)
    print(result)
    template = 'recommender/recommendation.html'
    return render(request, template)


"""
    def create_soup(x):
        return ' '.join(x['genres']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(
            x['description'] + ' ' + ' '.join(x['keyword']))


    def clean_data(x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            # Check if director exists. If not, return empty string
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            else:
                return ''
"""


def content_based_recommendation(title):
    data_frame_movie_records = pd.DataFrame.from_records(
        Movies.objects.all().values_list('movie_id', 'movie_title', 'imdb_url', 'decade', 'genres', 'avg_rating',
                                         'poster', 'cast', 'director', 'description', 'keyword'),
        columns=['movie_id', 'movie_title', 'imdb_url', 'decade', 'genres', 'avg_rating', 'poster', 'cast', 'director',
                 'description', 'keyword'])
    # descriptions = data_frame_movie_records['description']
    # keywords = data_frame_movie_records['keyword']

    features = ['genres', 'cast', 'director', 'description', 'keyword']

    # for feature in features:
    #    data_frame_movie_records[feature] = data_frame_movie_records[feature].apply(literal_eval)
    # print(data_frame_movie_records)
    # print(data_frame_movie_records[['genres', 'cast', 'director', 'description','keyword']].head(3))

    # for feature in features:
    #    data_frame_movie_records[feature] = data_frame_movie_records[feature].apply(clean_data)

    data_frame_movie_records['soup'] = ''
    for feature in features:
        data_frame_movie_records['soup'] += data_frame_movie_records[feature]
    # data_frame_movie_records['soup'] = data_frame_movie_records.apply(create_soup, axis=1)
    # soup_temp = data_frame_movie_records[['soup']].head(2)

    count = CountVectorizer(stop_words='english')  # This was used first which is correct.
    # count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0,stop_words='english') #This is an alternative of the above.
    count_matrix = count.fit_transform(data_frame_movie_records['soup'])
    print(count_matrix.shape)

    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    # Reset index of your main DataFrame and construct reverse mapping as before
    metadata = data_frame_movie_records.reset_index()
    indices = pd.Series(data_frame_movie_records.index, index=data_frame_movie_records['movie_title'])

    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    # return data_frame_movie_records.iloc[movie_indices][['movie_title', 'movie_id']]
    return data_frame_movie_records.iloc[movie_indices][['movie_title']]


def content_based_recommendation_result(request):
    title = 'Godfather, The (1972)'
    result = content_based_recommendation(title)
    print(result)
    print("title al ve öyle hesapla")
    template = 'recommender/recommendation.html'
    return render(request, template)


def hybrid_recommendation(request):
    user_id = request.user.id
    existing_ratings = Rating.objects.all()
    existing_users = existing_ratings.filter().values('user_id').distinct()
    print(existing_users.count())
    #print(existing_users)
    answers_list = list(existing_users)
    #print(answers_list)
    df_rating_users = pd.DataFrame(answers_list)
    df_rating_users.to_csv(r'df_rating_users.csv', index=False)
    print(request.user.id)
    user_exists = Rating.objects.filter(user_id=request.user.id)
    if user_exists.count() == 0:
        user_based_results = recommendation_user_profile(request)
    else:
        user_based_results = recommend_user_based(user_id)

    hybrid_results = user_based_results['movie_title'].apply(
        lambda movie_title: content_based_recommendation(movie_title))

    hybrid_movies = []
    df = hybrid_results.to_frame()
    deneme = df.values.tolist()
    stx = deneme
    tostr = str(stx)
    # print(tostr)
    result = []
    split = tostr.split(",")
    for row in split:
        # print("---------------------")

        den = row.split("\n")
        for r in den:
            result.append(r)

    deneme = []
    # print(result[1])
    for k in result:
        x = k.split(" ", 1)
        # print(x[1].strip())
        deneme.append(x[1].strip())
    # print(deneme[4])
    movie_titles = []
    movies = Movies.objects.all()
    for i in deneme:
        title = str(i).replace('(1...', '').replace(']', '').replace('[', '').strip()
        movie_titles.append(title)
    for title in movie_titles:
        for movie in movies:
            if title == 'movie_title':
                #print('if')
                pass
            elif movie.movie_title == title:
                #print('elif')
                hybrid_movies.append(movie)

    print(hybrid_movies)
    hybrid_movies = list(dict.fromkeys(hybrid_movies))
    print(hybrid_movies)

    context = {
        'hybrid_movies': hybrid_movies
    }
    template = 'recommender/recommendation.html'
    return render(request, template, context)


def hybrid_recom_knn(request):
    movie_title = 'Toy Story (1995)'
    knn_results = knn(request, movie_title)
    print(type(knn_results))
    hybrid_results = []
    func = lambda movietitle: content_based_recommendation(movietitle)
    for row in knn_results:
        hybrid_results.append(func(row.movie_title))
    print(hybrid_results)

    template = 'recommender/recommendation.html'
    return render(request, template)
