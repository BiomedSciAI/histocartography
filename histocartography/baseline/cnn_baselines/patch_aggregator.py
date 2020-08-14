import os
from majority_voting import *
from learned_fusion import *
from base_penultimate import *
from aggregate_penultimate import *

class PatchAggregator:
    def __init__(self, config):
        self.base_data_split_path = config.base_data_split_path
        self.model_save_path = config.model_save_path
        self.aggregator = config.aggregator
        self.tumor_types = config.tumor_types

        self.extract_prediction(config)

    def get_troi_ids(self, evalmode, tumor_type):
        trois = []
        filename = self.base_data_split_path + evalmode + '_list_' + tumor_type + '.txt'
        with open(filename, 'r') as f:
            for line in f:
                line = line.split('\n')[0]
                if line != '':
                    trois.append(line)
        return trois


    def prediction(self, evalmode, tumor_type):
        embeddings = np.array([])
        probabilities = np.array([])
        patch_count = np.array([])
        troi_ids = self.get_troi_ids(evalmode, tumor_type)

        for i in range(len(troi_ids)):
            pred_embedding, pred_probabilities = self.pred.predict(troi_ids[i])
            patch_count = np.append(patch_count, pred_embedding.shape[0])

            if i == 0:
                embeddings = pred_embedding
                probabilities = pred_probabilities
            else:
                embeddings = np.vstack((embeddings, pred_embedding))
                probabilities = np.vstack((probabilities, pred_probabilities))

        patch_count = patch_count.astype(int)

        print(embeddings.shape)
        print(probabilities.shape)
        print(patch_count.shape, '\n')

        h5_fout = h5py.File(self.model_save_path + evalmode + '_' + tumor_type + '.h5', 'w')
        h5_fout.create_dataset('patch_count', data=patch_count, dtype='int32')
        h5_fout.create_dataset('patch_embeddings', data=embeddings, dtype='float32')
        h5_fout.create_dataset('patch_probabilities', data=probabilities, dtype='float32')
        h5_fout.close()


    def extract_prediction(self, config):
        if config.mode in ['single_scale_10x', 'single_scale_20x', 'single_scale_40x']:
            from predict_s import Predict
            self.pred = Predict(config, modelmode='f1')

        if config.mode in ['late_fusion_1020x', 'late_fusion_102040x']:
            from predict_lf import Predict
            self.pred = Predict(config, modelmode='f1')

        for tumor_type in self.tumor_types:
            print('Tumor type: ', tumor_type)
            if not os.path.isfile(self.model_save_path + 'train_' + tumor_type + '.h5'):
                self.prediction('train', tumor_type)

            if not os.path.isfile(self.model_save_path + 'val_' + tumor_type + '.h5'):
                self.prediction('val', tumor_type)

            if not os.path.isfile(self.model_save_path + 'test_' + tumor_type + '.h5'):
                self.prediction('test', tumor_type)


    def evaluate_troi(self, config):
        if self.aggregator == 'majority_voting':
            eval = MajorityVoting(config=config)
            eval.process(config=config)

        elif self.aggregator == 'learned_fusion':
            eval = LearnedFusion(config=config)
            eval.process(config=config)

        elif self.aggregator == 'base_penultimate':
            eval = BasePenultimate(config=config)
            eval.process(config=config)

        elif self.aggregator == 'aggregate_penultimate':
            eval = AggregatePenultimate(config=config)
            eval.process(config=config)



