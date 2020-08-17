import h5py
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
#from histocartography.data_generation.nuclei_features.models.dataloader import *
from dataloader_patch_single_scale import *


class Analyze_Features:
    def __init__(self, config, embedding_dim):
        self.device = config.device
        self.embedding_dim = embedding_dim
        self.data_split_dir = config.base_path + 'data_split_temp/'

        self.nClusters = 10
        self.clustering_algorithm = 'kmeans'
        self.tumor_types = config.tumor_types
        self.nuclei_features_path = config.nuclei_features_path
    # enddef

    def get_filenames(self):
        self.train_filenames = []
        self.val_filenames = []
        self.test_filenames = []

        def read_txt_file(path, filelist):
            with open(path, 'r') as f:
                for line in f:
                    line = line.split('\n')[0]
                    if line != '':
                        filelist.append(line)
        # enddef

        for tumor_type in self.tumor_types:
            read_txt_file(
                self.data_split_dir +
                'train_list_' +
                tumor_type +
                '.txt',
                self.train_filenames)
            read_txt_file(
                self.data_split_dir +
                'val_list_' +
                tumor_type +
                '.txt',
                self.val_filenames)
            read_txt_file(
                self.data_split_dir +
                'test_list_' +
                tumor_type +
                '.txt',
                self.test_filenames)
        # endfor
        print(
            'Train:', len(
                self.train_filenames), ' Val:', len(
                self.val_filenames), ' Test:', len(
                self.test_filenames))
    # enddef

    def prepare_data(self):
        def read_h5_file(filelist):
            embeddings = np.zeros(shape=(1, self.embedding_dim))
            for i in range(len(filelist)):
                tumor_type = filelist[i].split('_')[1]
                path = self.nuclei_features_path + \
                    tumor_type + '/' + filelist[i] + '.h5'

                if os.path.isfile(path):
                    with h5py.File(path, 'r') as f:
                        embeddings_ = np.array(f['embeddings'])
                        embeddings = np.vstack((embeddings, embeddings_))
                    # end
                # endif
            # endfor
            embeddings = np.delete(embeddings, 0, axis=0)
            return embeddings
        # enddef

        self.train_embeddings = read_h5_file(self.train_filenames)
        self.val_embeddings = read_h5_file(self.val_filenames)
        self.test_embeddings = read_h5_file(self.test_filenames)
    # enddef

    def feature_analysis(self):
        print('\nCLUSTER ANALYSIS\n')

        def create_clusters(embeddings):
            if self.clustering_algorithm == 'gmm':
                model = GaussianMixture(
                    n_components=self.nClusters,
                    covariance_type='diag')
                model.fit(embeddings)

            elif self.clustering_algorithm == 'kmeans':
                model = KMeans(
                    n_clusters=self.nClusters,
                    random_state=0).fit(embeddings)

            return model
        # enddef

        def compute_metrics(embeddings, cluster_labels):
            sh_score = silhouette_score(embeddings, cluster_labels)
            ch_score = calinski_harabasz_score(embeddings, cluster_labels)
            db_score = davies_bouldin_score(embeddings, cluster_labels)
            return round(sh_score, 4), round(ch_score, 4), round(db_score, 4)
        # enddef

        # Prepare data
        self.get_filenames()
        self.prepare_data()

        # Create clusters and predict cluster labels
        cluster_model = create_clusters(self.train_embeddings)
        train_cluster_labels = cluster_model.predict(self.train_embeddings)
        val_cluster_labels = cluster_model.predict(self.val_embeddings)
        test_cluster_labels = cluster_model.predict(self.test_embeddings)

        print('Cluster assignments:')
        print('Train: ', np.unique(train_cluster_labels, return_counts=True))
        print('Val: ', np.unique(val_cluster_labels, return_counts=True))
        print('Test: ', np.unique(test_cluster_labels, return_counts=True))

        # Analyze clusters
        print('\nDavies-Bouldin scores:')
        sh, ch, db = compute_metrics(
            self.train_embeddings, train_cluster_labels)
        print('Train: sh= ', sh, ' ch=', ch, ' db=', db)
        sh, ch, db = compute_metrics(self.val_embeddings, val_cluster_labels)
        print('Val: sh= ', sh, ' ch=', ch, ' db=', db)
        sh, ch, db = compute_metrics(self.test_embeddings, test_cluster_labels)
        print('Test: sh= ', sh, ' ch=', ch, ' db=', db)

    # enddef
