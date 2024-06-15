import streamlit as st
import pickle
import requests
import pandas as pd

def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=50d5722b3b79374ab81a710f551babea&language=en-US".format(movie_id)
    response = requests.get(url)
    data = response.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse = True, key = lambda x:x[1])[1:6]

    recommended_movies_names = []
    #recommended_movies_poster = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        #fetch poster from api
        #recommended_movies_poster.append(fetch_poster(movie_id))
        recommended_movies_names.append(movies.iloc[i[0]].title)
    return recommended_movies_names#, recommended_movies_poster

st.title('Movie Recommender System')

movie_dict = pickle.load(open('C:/Users/Aditya/Desktop/Projects/Facial Expression Recognition/Project-Movie_Recommender/movie_dict.pkl','rb'))
movies = pd.DataFrame(movie_dict)

similarity = pickle.load(open('C:/Users/Aditya/Desktop/Projects/Facial Expression Recognition/Project-Movie_Recommender/similarity.pkl','rb'))


selected_movie_name = st.selectbox(
    "Type or select a movie from the Drop Down box",
    movies['title'].values)

if st.button('Show Recommendation'):
    recommendations = recommend(selected_movie_name)
    for i in recommendations:
        st.write(i)
    