import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



credits = pd.read_csv("C:/Users/Aditya/Desktop/Projects/Facial Expression Recognition/Project-Movie_Recommender/tmdb_5000_credits.csv")
movies = pd.read_csv("C:/Users/Aditya/Desktop/Projects/Facial Expression Recognition/Project-Movie_Recommender/tmdb_5000_movies.csv")

movies = movies.merge(credits,on='title')

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

movies.dropna(inplace=True)

#print(movies.duplicated().sum)

import ast

#'[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
#['Action', 'Adventure', 'FFantasy', 'SciFi']
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 

movies['cast'] = movies['cast'].apply(convert3)

def fetch_director(text):
    L=[]
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

movies['overview'].apply(lambda x:x.split())

#'Sam Worthington' -> 'SamWorthington'

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ", "") for i in x])

movies['tags'] = movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))

new_df['tags'].apply(lambda x:x.lower())

#Vectorization
# Step-1: concantate all tags
# Step-2: calculate frequency of each unique word
# Step-3: Select the top 5000 words
# Step-4: For each movie, calculate frequency of all 5000 words
# Step-5: A matrix will be formed of shape (5000, 5000)
# Note: not to consider stop words so remove those words

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000, stop_words = 'english')

vectors = cv.fit_transform(new_df['tags']).toarray()

# Problem: Repetition in words like actor - actors; love - loving; etc
# Solution: Apply Stemming

import nltk

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)


# We will cluster the vectors(or movie) through Cosine Distance - not Euclidian Distance
# Applying clustering Algorithm

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)

# With all the distances, now we need to find the final results
print("------------------------------------")
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse = True, key = lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
    return

recommend('Batman Begins')

# Connecting with front end

import pickle
pickle.dump(new_df.to_dict(), open('C:/Users/Aditya/Desktop/Projects/Facial Expression Recognition/Project-Movie_Recommender/movie_dict.pkl', 'wb'))
pickle.dump(similarity, open('C:/Users/Aditya/Desktop/Projects/Facial Expression Recognition/Project-Movie_Recommender/similarity.pkl', 'wb'))