from itertools import groupby

from sklearn import cross_validation

from sklearn.neighbors import KNeighborsClassifier
from scipy import spatial

from Movie import Movie
from Rating import Rating

movie1 = [3, 5, 0, 0, 0, 4, 2, 0, 0]
movie2 = [0, 0, 3, 5, 0, 2, 0, 1, 0]
movie3 = [3, 4, 0, 0, 0, 4, 3, 0, 3]

test = [2, 5, 0, 1, 0, 3, 2, 0, 2]

training = [movie1, movie2, movie3]

m1c = [1, 0, 0, 1]
m2c = [0, 0, 1, 0]
m3c = [1, 1, 0, 1]


def mydist(x, y, **kwargs):
    slim_x = []
    slim_y = []
    for xx, yy in zip(x,y):
        if xx != 0 or yy != 0:
            slim_x.append(xx)
            slim_y.append(yy)

    result =  spatial.distance.cosine(slim_x, slim_y)
    # print(result)
    return result


# knn = KNeighborsClassifier(n_neighbors=2, metric=mydist)
# knn.fit(training, [m1c, m2c, m3c])

# print(knn.predict(test))


def parseMoviesFile():
    movies = {}
    file = open("resources/u.item", 'r')
    for line in file.readlines():
        splitted = line.replace("\n", '').split("|")
        movies[splitted[0]] = Movie(splitted[0], splitted[1], map(lambda x: int(x), splitted[5:]))
    file.close()
    return movies


def parseRatingsFile():
    ratings = []
    x = open("resources/u.data", 'r')
    for line in x.readlines():
        splitted = line.replace("\n", '').split("\t")
        ratings.append(Rating(int(splitted[0]), splitted[1], float(splitted[2])))
    x.close()
    return ratings


def getMoviesRatingsVectors(sortedUsersIds, ratings):
    movies_ratings = {}
    for movie, ratings in groupby(ratings, lambda x: x.movie):
        movie_ratings = [0] * len(sortedUsersIds)
        for rating in ratings:
            movie_ratings[rating.user - 1] = rating.rating
        movies_ratings[movie] = movie_ratings
    return movies_ratings


def get_train_genres(m_train, movies):
    train_movies_ids = map(lambda x: x[0], m_train)
    l = map(lambda movie_id: movies[movie_id].genres, train_movies_ids)
    return list(l)


def validate(test_movies_ids, movies, all_predicted_genres):
    true_positive = false_positive = false_negative = 0
    all_original_genres = map(lambda movie_id: movies[movie_id].genres, test_movies_ids)
    for predicted_genres, original_genres in zip(all_predicted_genres, all_original_genres):
        for single_predicted_genre, single_original_genre in zip(predicted_genres, original_genres):
            if single_predicted_genre == 1 and single_original_genre == 1:
                true_positive += 1
            if single_predicted_genre == 1 and single_original_genre == 0:
                false_positive += 1
            if single_predicted_genre == 0 and single_original_genre == 1:
                false_negative += 1
    print("true positive:" + str(true_positive))
    print("false positive:" + str(false_positive))
    print("false negative:" + str(false_negative))
    pass


if __name__ == "__main__":
    movies = parseMoviesFile()
    ratings = parseRatingsFile()
    sortedUsersIds = sorted(set(map(lambda rating: rating.user, ratings)))
    movies_ratings = getMoviesRatingsVectors(sortedUsersIds, ratings)

    m_train, m_test = cross_validation.train_test_split(movies_ratings.items(), test_size=0.3, random_state=0)
    train_genres = get_train_genres(m_train, movies)

    knn = KNeighborsClassifier(n_neighbors=5, metric=mydist, algorithm='brute')
    # knn = KNeighborsClassifier(n_neighbors=5, metric=mydist)
    knn.fit(map(lambda x: x[1], m_train), train_genres)

    predicted_genres = knn.predict(map(lambda x: x[1], m_test))

    validate(map(lambda x: x[0], m_test), movies, predicted_genres)
