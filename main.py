import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import re

# Загрузка датасетов
load_dir = "movierecommenderdataset/"
movies = pd.read_csv(load_dir + "movies.csv")
ratings = pd.read_csv(load_dir + "ratings.csv")

# Совместить датасет фильмов и рейтингов по полю идентификатора фильма
merged_list = pd.merge(movies, ratings, on="movieId")


def get_movie_recommendation_by_genre(movie_genre):
    n_movies_to_recommend = 10  # Количество рекомендуемых фильмов

    print('')
    print('---------- (2) CREATE TRAINING DATA SET ----------')

    # extract movies under input genre (Contains)
    #    movie_list = merged_list[merged_list['genres'].str.contains(movie_genre)]

    # extract movies under input genre (Match)

    print('')
    print('--------------- (2a) Get All Movies Under Genre: ', movie_genre,
          ' ---------------')
    # Список отзывов к фильмам жанра movie_genre - самый популярный в отзывах
    # пользователя с id user_id
    movie_list = merged_list[merged_list['genres'].str.match(movie_genre)]
    print(movie_list)

    # Выбрать фильмы, которые имеют жанр не Триллер
    #    movie_list = movie_list[~movie_list['genres'].str.contains('Thriller')]

    # Выбрать фильмы по году выпуска
    #    movie_list = movie_list[movie_list['title'].str.contains('19')]

    # Построение разреженной матрицы оценок пользователями фильмов
    genre_ratings = movie_list.pivot(index='movieId', columns='userId', values='rating')

    # Если фильм пользователем не оценен, то считаем оценку = 0
    genre_ratings.fillna(0, inplace=True)
    print('')
    print(genre_ratings)

    # Из матрицы genre_ratings выбираем фильмы, которые были оценены более
    # 15 раз пользователями, которые оценили как минимум 50 фильмов
    # Таким образом формирует тренировочный датасет

    # Выбираем фильмы, которые были оценены как минимум 15 раз
    print('')
    print('--------------- (2b) Must Be Movies That Have Been Rated At Least 15 Times ---------------')
    # Подсчет, сколько человек оценило каждый из фильмов
    no_user_voted = movie_list.groupby('movieId')['rating'].agg('count')
    genre_ratings = genre_ratings.loc[no_user_voted[no_user_voted > 15].index, :]
    print(genre_ratings)

    # Выбрать фильмы которые были оценены как минимум рейтингом 3.0
    #    genre_ratings = genre_ratings.loc["rating"["rating" > 3].index,:]

    # Пользователь должен оценить как минимум 50 фильмов
    print('')
    print('--------------- (2c) Must Be Users Who Have Rated At Least 50 Times ---------------')
    no_movies_voted = movie_list.groupby('userId')['rating'].agg('count')
    genre_ratings = genre_ratings.loc[:,no_movies_voted[no_movies_voted > 50].index]
    print(genre_ratings)

    # Количество оценок у фильмов из тренировочного датасета
    f, ax = plt.subplots(1, 1, figsize=(16, 4))
    f, ax.set_xlim(0, 100)
    plt.scatter(no_user_voted.index, no_user_voted, color='mediumblue')
    plt.axhline(y=15, color='r')
    plt.xlabel('Идентификатор фильма')
    plt.ylabel('Кол-во оценивших')
    plt.title('Количество оценок у фильмов из тренировочного датасета')
    plt.show()

    # График количества оценок по пользователям
    f, ax = plt.subplots(1, 1, figsize=(16, 4))
    plt.scatter(no_movies_voted.index, no_movies_voted, color='mediumseagreen')
    plt.axhline(y=50, color='r')
    plt.xlabel('Идентификатор пользователя')
    plt.ylabel('Кол-во голосов')
    plt.title('График количества оценок по пользователям')
    plt.show()

    # Убрать разреженность и получить сжатые разреженные строки (ненулевые значения)
    csr_data = csr_matrix(genre_ratings.values)
    print('')
    print('--------------- (2d) Training Data Set Created! (In CSR Format: X, Y, Rating) ---------------')
    print(csr_data)

    # reset index i.e., row index starts at 0 instead of 1
    genre_ratings.reset_index(inplace=True)

    # Установить параметры knn модели (https://scikit-learn.org/stable/modules/neighbors.html)
    # Метрика - metric: cosine / manhattan / euclidean (default)
    # Алгоритм изучения ближайших соседей - algorithm: brute / ball_tree / kd_tree / auto (default)
    # Количество исследуемых соседей - n_neighbors
    # Параллелизм - n_jobs: 1 (no joblib parallelism) / -1 (use all CPUs)
    # Например: knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)

    print('')
    print('---------- (3) START TRAINING ----------')
    knn = NearestNeighbors(algorithm='brute', n_neighbors=20, n_jobs=-1)

    # Загрузка тренировочного датасета
    knn.fit(csr_data)
    print('')
    print('--------------- Training Done! ---------------')

    # Если матрица genre_ratings не пуста
    if len(genre_ratings):
        # get movieId with the highest total rating
        #        total_rating = movie_list.groupby('movieId')['rating'].sum()
        #        movie_idx = genre_ratings.iloc[total_rating[total_rating > 75].index,:]
        #        print(total_rating)
        #        genre_ratings.sort(reverse=True)

        # Получить id первого фильма из user_list(список фильмов, оцененных пользователем)
        user_movie_list = user_list[user_list['genres'].str.match(movie_genre)]

        print('')
        print('---------- (4) START PREDICTING ----------')
        print('')
        print('--------------- (4a) Get All Movies Watched By User Under Genre:', movie_genre, '---------------')
        print(user_movie_list)

        # Построение разреженной матрицы оценок фильмов выбранным юзером
        user_genre_ratings = user_movie_list.pivot(index='movieId', columns='userId', values='rating')
        user_genre_ratings.fillna(0, inplace=True)
        user_genre_ratings.reset_index(inplace=True)
        movie_idx = user_genre_ratings.iloc[0]['movieId']  # id первого по счету фильма, оцененного пользователем

        # get first movieId from genre_ratings
        #        movie_idx = genre_ratings.iloc[0]['movieId']

        print('')
        print('--------------- (4b) Select A Movie From User List (Independent Variable) ----------')
        print('Movie Id: ', movie_idx, 'selected!')

        # Получить индекс movieId
        movie_idx = genre_ratings[genre_ratings['movieId'] == movie_idx].index[0]

        print('')
        print('Index No.:', movie_idx)
        print(genre_ratings)

        # 5. Выводом модели будет 10 фильмов, основанные на входных фильмах

        # Получение рекомендаций
        print('')
        print(f'--------------- (4c) Getting Nearest K={movie_genre} Neighbors To Selected Movie ---------------')
        print(csr_data[movie_idx])
        distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=n_movies_to_recommend + 1)

        # Перевод расстояний в читабельный формат
        rec_movie_indices = sorted(list(
            zip(indices.squeeze().tolist(), distances.squeeze().tolist())),
                                   key=lambda x: x[1])[:0:-1]
        #        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1],reverse=True)[:0:-1]

        print('')
        print('--------------- (4d) Getting Neighbors Done! (Movie Index No., Distance) ---------------')
        print(rec_movie_indices)

        recommend_frame = []

        print('')
        print(f'--------------- (4e) Display {n_movies_to_recommend} Movie Recommendations ---------------')

        # Выходная таблица рекомендаций
        for val in rec_movie_indices:
            movie_idx = genre_ratings.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            print('Processing Movie: ', 'ID -', str(int(movie_idx)).rjust(4, ' '), 'Title -', movies.iloc[idx]['title'].values[0])
            recommend_frame.append({'ID': movies.iloc[idx]['movieId'].values[0], 'Title': movies.iloc[idx]['title'].values[0],
                                    'Genres': movies.iloc[idx]['genres'].values[0], 'Rating': ratings.iloc[idx]['rating'].values[0], 'Distance': val[1]})
        print('')
        df = pd.DataFrame(recommend_frame, index=range(1, n_movies_to_recommend+1))
        return df
    else:
        return "No movies found. Please check your input"


# Выбор юзера
user_id = 8  # Genre 'Comedy'
# user_id = 2 Genre 'Drama'
# user_id = 3 Genre 'Drama'

print('MOVIE RECOMMENDATIONS FOR USERID:', user_id, '\n')
print('---------- (1) GET USER TOP GENRE ----------')

# Выбор всех фильмов, оцененных пользователем
user_list = merged_list[merged_list['userId'] == user_id]

# Генерация списка с количеством оцененных пользователем фильмов (по жанрам)
print('---------- Displaying all genres for userId:', user_id, '----------')
genre_list = user_list['genres'].str.split('|', expand=True).stack().value_counts()
print(genre_list)

# Получить наиболее популярный для пользователя жанр
top_movie_genre = genre_list.index[0]
print('---------- Top Genre for userId', user_id, ':', top_movie_genre, '----------')

get_movie_recommendation_by_genre(top_movie_genre)