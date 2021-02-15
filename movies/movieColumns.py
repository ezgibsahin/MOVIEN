import pandas as pd

import os

#print(os.getcwd())

movie_wo_posters = 'moviesUpdated.csv'
dataframeBeginning = pd.read_csv(movie_wo_posters)
# print(dataframe)

#print(dataframeBeginning)

movieRating = 'movies_with_avgrating.csv'
averageRatingDataframe = pd.read_csv(movieRating)
averageRatingDataframe.drop(averageRatingDataframe.columns[[0, 1, 2, 3, 4, 5]], axis=1, inplace=True)
# averageRatingDataframe.reset_index(inplace=True)
#print(averageRatingDataframe)
averageRatingDataframe.to_csv(r'avg_rating.csv', index=False)

posterCSV = 'movie_poster.csv'
dataframePoster = pd.read_csv(posterCSV, names=["movie_id", "poster"])
dataframePoster.drop(dataframePoster.columns[[0]], axis=1, inplace=True)
# dataframePoster.reset_index(inplace=True)
#print(dataframePoster)

cast = "movie_cast.csv"
dataframe_cast = pd.read_csv(cast, names=["movie_id", "cast"],encoding = "ISO-8859-1")
dataframe_cast.drop(dataframe_cast.columns[[0]], axis=1, inplace=True)
print(dataframe_cast)

director = "movie_director.csv"
dataframe_director = pd.read_csv(director, names=["movie_id", "director"],encoding = "ISO-8859-1")
dataframe_director.drop(dataframe_director.columns[[0]], axis=1, inplace=True)
print(dataframe_director)

description = "movie_description.csv"
dataframe_description = pd.read_csv(description, names=["movie_id", "description"],encoding = "ISO-8859-1")
dataframe_description.drop(dataframe_description.columns[[0]], axis=1, inplace=True)
print(dataframe_description)

keyword = "movie_keyword.csv"
dataframe_keyword = pd.read_csv(keyword, names=["movie_id", "keyword"],encoding = "ISO-8859-1")
dataframe_keyword.drop(dataframe_keyword.columns[[0]], axis=1, inplace=True)
print(dataframe_keyword)


dataframeBeginning['avg_rating'] = averageRatingDataframe
dataframeBeginning['poster'] = dataframePoster
dataframeBeginning['cast'] = dataframe_cast
dataframeBeginning['director'] = dataframe_director
dataframeBeginning['description'] = dataframe_description
dataframeBeginning['keyword'] = dataframe_keyword

print(dataframeBeginning)

# dataframe.reset_index(inplace=True)
dataframeBeginning.to_csv(r'movie.csv', index=False)
