import abc
import numpy
from typing import Tuple
import pandas as pd
import numpy as np
import scipy
import time
from datetime import datetime
import random


class Recommender(abc.ABC):
    def __init__(self, ratings: pd.DataFrame):
        self.initialize_predictor(ratings)

    @abc.abstractmethod
    def initialize_predictor(self, ratings: pd.DataFrame):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        raise NotImplementedError()

    def rmse(self, true_ratings) -> float:
        """
        :param true_ratings: DataFrame of the real ratings
        :return: RMSE score
        """
        sum1 = 0
        count = 0
        for index in true_ratings.index:
            predicted = self.predict(true_ratings['user'][index], true_ratings['item'][index],
                                     true_ratings['timestamp'][index])
            sum1 += (predicted - true_ratings['rating'][index]) ** 2
        return np.sqrt(sum1 / len(true_ratings.index))


class BaselineRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        self.R = np.mean(ratings['rating'])
        self.users_dict = {}
        self.items_dict = {}
        for u in np.unique(ratings['user']):
            user_ratings = ratings[ratings['user'] == u]
            self.users_dict[u] = [np.mean(user_ratings['rating']) - self.R, user_ratings]
        for i in np.unique(ratings['item']):
            items_ratings = ratings[ratings['item'] == i]
            self.items_dict[i] = [np.mean(items_ratings['rating']) - self.R, items_ratings]

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        prediction = self.R + self.items_dict[item][0] + self.users_dict[user][0]
        return prediction if 0.5 <= prediction <= 5 else (5 if prediction > 5 else 0.5)


class NeighborhoodRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        self.baseline = BaselineRecommender(ratings)
        user_num = len(ratings[['user']].drop_duplicates())
        item_num = len(ratings[['item']].drop_duplicates())
        self.R_gal = np.zeros((user_num, item_num))

        for i, row in ratings.iterrows():
            self.R_gal[int(row['user'])][int(row['item'])] = row['rating'] - self.baseline.R

        self.sim_mat = np.zeros((user_num, user_num))
        for i in range(user_num):
            for j in range(user_num):
                self.sim_mat[i][j] = self.user_similarity(i, j)

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        i = 0
        sim_list = []
        for s in self.sim_mat[int(user)]:
            sim_list.append((s, i))
            i += 1

        three_best = sorted(sim_list, reverse=True)[:3]

        corr_sum = 0
        for i in range(3):
            corr_sum += three_best[i][0] * self.R_gal[three_best[i][1]][int(item)]

        abs_corr_sum = 0
        for i in range(3):
            abs_corr_sum += np.abs(three_best[i][0])
        pre = self.baseline.predict(user, item, timestamp) + corr_sum / abs_corr_sum
        if 0.5 <= pre <= 5:
            return pre
        if pre > 5:
            return 5
        return 0.5

    def user_similarity(self, user1: int, user2: int) -> float:
        """
        :param user1: User identifier
        :param user2: User identifier
        :return: The correlation of the two users (between -1 and 1)
        """
        if user1 == user2:
            return -1
        r1 = self.R_gal[int(user1)]
        r2 = self.R_gal[int(user2)]
        cov = r1.dot(r2)
        corr = cov / (np.linalg.norm(r1) * np.linalg.norm(r2))
        return corr


class LSRecommender(Recommender):

    def initialize_predictor(self, ratings: pd.DataFrame):
        self.users_count = len(np.unique(ratings['user']))
        self.items_count = len(np.unique(ratings['item']))
        num_ratings = len(ratings.index)
        self.b = np.zeros(self.users_count + self.items_count + 3)
        self.X = np.zeros((num_ratings, self.users_count + self.items_count + 3))

        for index, row in ratings.iterrows():
            self.X[index][int(row['user'])] = 1
            self.X[index][int(row['item'] + self.users_count)] = 1
            daytime = datetime.fromtimestamp(row['timestamp'])
            if 6 <= daytime.time().hour <= 18:
                day = 1
            else:
                day = 0
            if daytime.weekday() == 5 or daytime.weekday() == 4:
                self.X[index][-1] = 1
            self.X[index][-2 - day] = 1

        # create y
        self.R_avg = ratings['rating'].mean()
        self.y = np.array(ratings['rating'] - self.R_avg)

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        user = int(user)
        item = self.users_count + int(item)
        daytime = datetime.fromtimestamp(timestamp)
        weekend = 0
        day = 0
        if 6 <= daytime.time().hour <= 18:
            day = 1
        if daytime.weekday() == 5 or daytime.weekday() == 4:
            weekend = 1
        prediction = self.R_avg + self.b[user] + self.b[item] + self.b[-1] * weekend + self.b[-3] * day + (1 - day) * \
                     self.b[-2]
        return prediction if 0.5 <= prediction <= 5 else (5 if prediction > 5 else 0.5)

    def solve_ls(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates and solves the least squares regression
        :return: Tuple of X, b, y such that b is the solution to min ||Xb-y||
        """
        self.b = np.linalg.lstsq(self.X, self.y, rcond=None)[0]
        return self.X, self.b, self.y


class CompetitionRecommender(Recommender):

    def initialize_predictor(self, ratings: pd.DataFrame):
        users_count = len(np.unique(ratings['user']))
        items_count = len(np.unique(ratings['item']))
        self.R_avg = ratings['rating'].mean()
        self.rating_mat = np.zeros((users_count, items_count))
        self.in_list = np.zeros(items_count)

        start = time.time()

        for i, row in ratings.iterrows():
            self.in_list[int(row['item'])] += 1
            self.rating_mat[int(row['user'])][int(row['item'])] = row['rating'] - self.R_avg
        self.clusters = self.k_means(11,4)

    def compare_to(self, l1, l2):
        if len(l1) != len(l2):
            return False
        i = 0
        for v in l1:
            if v != l2[i]:
                return False
            i += 1
        return True

    def user_similarity(self, user1, user2):

        r1 = self.rating_mat[int(user1)]
        r2 = self.rating_mat[int(user2)]
        cov = r1.dot(r2)
        corr = cov / (np.linalg.norm(r1) * np.linalg.norm(r2))
        return corr

    def k_means(self, k, iter_num):
        start_time = time.time()
        users = np.arange(len(self.rating_mat))
        iterations = 0
        clusters = [random.randint(0, len(users)) for i in range(k)]
        old_clusters = [-1 for i in range(k)]
        clusters_dict_real = {}
        while (not self.compare_to(clusters, old_clusters)) and iterations < iter_num:
            old_clusters = clusters.copy()
            clusters_dict = {}
            for c in clusters:
                clusters_dict[c] = []
            for u in range(len(users)):
                best_cluster = clusters[0]
                for c in clusters:
                    if self.user_similarity(u, best_cluster) < self.user_similarity(u, c):
                        best_cluster = c
                clusters_dict[best_cluster].append(u)
            for i in range(len(clusters)):
                users_in_cluster = len(clusters_dict[clusters[i]])
                if users_in_cluster != 0:
                    max_val = 0
                    new_cluster = 0
                    for u1 in clusters_dict[clusters[i]]:
                        c_val = 0
                        for u2 in clusters_dict[clusters[i]]:
                            c_val += self.user_similarity(u1, u2)
                        c_val /= users_in_cluster
                        if c_val > max_val:
                            max_val = c_val
                            new_cluster = u1
                clusters[i] = new_cluster
            iterations += 1
            clusters_dict_real = clusters_dict
        return clusters_dict_real

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """

        user_cluster = -1
        for i in (self.clusters.keys()):
            if user in self.clusters[i]:
                user_cluster = i
                break
        count = 0
        sum_rating = 0
        for u in self.clusters[user_cluster]:
            if self.rating_mat[int(u)][int(item)] != 0:
                sum_rating += self.rating_mat[int(u)][int(item)]
                count += 1
        if count == 0:
            return 0.5
        return (sum_rating / count) + self.R_avg
