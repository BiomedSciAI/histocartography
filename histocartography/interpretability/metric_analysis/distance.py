import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import directed_hausdorff
from sklearn import svm


class Distance:
    def __init__(self, distance):
        distance_ = {'pair' : self.pair_wise_distance,
                    'chamfer' : self.chamfer_distance,
                    'hausdorff' : self.hausdorff_distance,
                    'svm': self.svm_distance}
        self.distance = distance_[distance]


    def pair_wise_distance(self, x, y, metric='euclidean'):
        '''
        :param x: (ndarray) [n_points_1, n_dims]
        :param y: (ndarray) [n_points_2, n_dims]
        :param metric: ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’,
                       ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’, ‘mahalanobis’, ‘matching’,
                       ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’,
                       ‘sqeuclidean’, ‘wminkowski’, ‘yule’
        :return distance: Compute distance between each pair of the two collections of inputs.
        '''
        distance = cdist(x, y, metric=metric)
        return np.mean(distance)

    def svm_distance(self, x, y, metric='euclidean'):
        '''
        :param x: (ndarray) [n_points_1, n_dims]
        :param y: (ndarray) [n_points_2, n_dims]
        :param metric: Unused here
        :return distance: Fit an SVM model - then return accuracy as measure of "distance" (the larger the better)
        '''
        labels = np.concatenate(([0 for _ in range(x.shape[0])], [1 for _ in range(y.shape[0])]), axis=0)
        feats = np.concatenate((x, y), axis=0)

        clf = svm.SVC()
        clf.fit(feats, labels)
        predictions = clf.predict(feats)
        accuracy = sum(predictions == labels) / labels.shape[0]

        return accuracy

    def chamfer_distance(self, x, y, metric='euclidean'):
        """
        Compute the point cloud distance between 2 sets of points.
        The distance is based on the L2 bidirectional Chamfer loss.
        Note: for categorical variables (eg nuclei type), use a one-hot encoding
              of the category.
        Param:
        :param x: (ndarray) [n_points_1, n_dims]
        :param y: (ndarray) [n_points_2, n_dims]
        :return chamfer_dist: (float) the chamfer distance between the 2 points clouds
        """
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
        return chamfer_dist


    def hausdorff_distance(self, x, y, metric='euclidean'):
        '''
        The function h(A, B) is called the directed Hausdorff distance from A to B,
        which identifies the point a ∈ A that is farthest from any point of B,
        and measures the distance from a to its nearest neighbor in B.
        Param:
        :param x: (ndarray) [n_points_1, n_dims]
        :param y: (ndarray) [n_points_2, n_dims]
        :return hausdorff_dist: (float) the hausdorff distance between the 2 points clouds
        '''
        return max(directed_hausdorff(x, y)[0], directed_hausdorff(y, x)[0])



