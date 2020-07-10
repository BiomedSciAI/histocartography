import time
import mlflow
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

from histocartography.evaluation.confusion_matrix import ConfusionMatrix
from histocartography.evaluation.classification_report import ClassificationReport
from histocartography.baseline.dataloader import *

# -------------------------------
# Patch level evaluation
# -------------------------------


class PatchEvaluation:

    def __init__(self, config):
        self.num_classes = config.num_classes
        self.num_epochs = config.num_epochs
        self.learning_rate = config.learning_rate
        self.model_save_path = config.model_save_path
        self.is_pretrained = config.is_pretrained
        self.device = config.device

        self.early_stopping_min_delta = config.early_stopping_min_delta
        self.early_stopping_patience = config.early_stopping_patience

        self.learning_rate_decay = config.learning_rate_decay
        self.min_learning_rate = config.min_learning_rate
        self.reduce_lr_patience = config.reduce_lr_patience

        self.conf_matrix = ConfusionMatrix(return_img=True)
        self.class_report = ClassificationReport()

    def adjust_learning_rate(self, optimizer):
        lr_prev = self.learning_rate

        if self.reduce_lr_count == self.reduce_lr_patience:
            self.learning_rate *= self.learning_rate_decay
            self.reduce_lr_count = 0
            print(
                'Reducing learning rate from', round(
                    lr_prev, 6), 'to', round(
                    self.learning_rate, 6))

            for param_group in optimizer.param_groups:
                param_group['lr'] = self.learning_rate

        if self.learning_rate <= self.min_learning_rate:
            print('Minimum learning rate reached')
            self.min_reduce_lr_reached = True

        return optimizer

    def train(self, models, dataloaders, criterion, optimizer):
        print('\nTraining ...\n')
        embedding_model, classifier_model = models
        embedding_model = embedding_model.to(self.device)
        classifier_model = classifier_model.to(self.device)
        train_loader = dataloaders['train']
        valid_loader = dataloaders['val']

        best_val_acc = 0
        best_val_f1 = 0
        best_val_loss = float('Inf')

        self.reduce_lr_count = 0
        self.early_stopping_count = 0
        self.min_reduce_lr_reached = False

        log_train_acc = np.array([])
        log_train_f1 = np.array([])
        log_train_loss = np.array([])
        log_val_acc = np.array([])
        log_val_f1 = np.array([])
        log_val_loss = np.array([])

        for epoch in range(self.num_epochs):
            start_time = time.time()
            print('Epoch {}/{}'.format(epoch + 1, self.num_epochs))

            self.adjust_learning_rate(optimizer=optimizer)
            if self.min_reduce_lr_reached:
                break

            # ------------------------------------------------------------------- TRAIN MODE
            embedding_model.train()
            classifier_model.train()

            train_loss = 0.0
            true_labels = np.array([])
            predicted_labels = np.array([])

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                targets = targets.view(targets.shape[0])
                inputs, targets = inputs.to(
                    self.device), targets.to(
                    self.device)
                optimizer.zero_grad()

                embedding = embedding_model(inputs)
                embedding = embedding.squeeze(dim=2)
                embedding = embedding.squeeze(dim=2)
                outputs = classifier_model(embedding)
                loss = criterion(outputs, targets)

                # measure accuracy and record loss
                predicted_labels_ = torch.argmax(outputs, dim=1)
                true_labels = np.concatenate(
                    (true_labels, targets.cpu().detach().numpy()))
                predicted_labels = np.concatenate(
                    (predicted_labels, predicted_labels_.cpu().detach().numpy()))
                train_loss += loss.cpu().detach().numpy() * inputs.shape[0]

                loss.backward()
                optimizer.step()

            train_acc = accuracy_score(true_labels, predicted_labels)
            train_f1 = f1_score(
                true_labels,
                predicted_labels,
                average='weighted')
            train_loss = train_loss / len(true_labels)
            log_train_acc = np.append(log_train_acc, train_acc)
            log_train_f1 = np.append(log_train_f1, train_f1)
            log_train_loss = np.append(log_train_loss, train_loss)

            # ------------------------------------------------------------------- EVAL MODE
            embedding_model.eval()
            classifier_model.eval()

            val_loss = 0.0
            true_labels = np.array([])
            predicted_labels = np.array([])

            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(valid_loader):
                    targets = targets.view(targets.shape[0])
                    inputs, targets = inputs.to(
                        self.device), targets.to(
                        self.device)

                    embedding = embedding_model(inputs)
                    embedding = embedding.squeeze(dim=2)
                    embedding = embedding.squeeze(dim=2)
                    outputs = classifier_model(embedding)
                    loss = criterion(outputs, targets)

                    # measure accuracy and record loss
                    predicted_labels_ = torch.argmax(outputs, dim=1)
                    true_labels = np.concatenate(
                        (true_labels, targets.cpu().detach().numpy()))
                    predicted_labels = np.concatenate(
                        (predicted_labels, predicted_labels_.cpu().detach().numpy()))
                    val_loss += loss.cpu().detach().numpy() * inputs.shape[0]

            val_acc = accuracy_score(true_labels, predicted_labels)
            val_f1 = f1_score(
                true_labels,
                predicted_labels,
                average='weighted')
            val_loss = val_loss / len(true_labels)
            log_val_acc = np.append(log_val_acc, val_acc)
            log_val_f1 = np.append(log_val_f1, val_f1)
            log_val_loss = np.append(log_val_loss, val_loss)

            # ------------------------------------------------------------------- SAVING
            if val_acc > best_val_acc:
                print(
                    'val_acc improved from', round(
                        best_val_acc, 4), 'to', round(
                        val_acc, 4), 'saving models to', self.model_save_path)
                torch.save(
                    embedding_model,
                    self.model_save_path +
                    'embedding_model_best_acc.pt')
                torch.save(
                    classifier_model,
                    self.model_save_path +
                    'classifier_model_best_acc.pt')

                if (val_acc - best_val_acc) < self.early_stopping_min_delta:
                    self.early_stopping_count += 1
                else:
                    self.early_stopping_count = 0

                best_val_acc = val_acc
                self.reduce_lr_count = 0
            else:
                self.reduce_lr_count += 1
                self.early_stopping_count += 1

            if val_f1 > best_val_f1:
                print(
                    'val_f1 improved from', round(
                        best_val_f1, 4), 'to', round(
                        val_f1, 4), 'saving models to', self.model_save_path)
                torch.save(
                    embedding_model,
                    self.model_save_path +
                    'embedding_model_best_f1.pt')
                torch.save(
                    classifier_model,
                    self.model_save_path +
                    'classifier_model_best_f1.pt')
                best_val_f1 = val_f1

            if val_loss < best_val_loss:
                print(
                    'val_loss improved from', round(
                        best_val_loss, 4), 'to', round(
                        val_loss, 4), 'saving models to', self.model_save_path)
                torch.save(
                    embedding_model,
                    self.model_save_path +
                    'embedding_model_best_loss.pt')
                torch.save(
                    classifier_model,
                    self.model_save_path +
                    'classifier_model_best_loss.pt')
                best_val_loss = val_loss

            print(' - ' + str(int(time.time() - start_time)) + 's - loss:',
                  round(train_loss,
                        4),
                  '- acc:',
                  round(train_acc,
                        4),
                  '- val_loss:',
                  round(val_loss,
                        4),
                  '- val_acc:',
                  round(val_acc,
                        4),
                  '\n')

            # ------------------------------------------------------------------- MLFLOW LOGGING
            mlflow.log_metric('avg_val_loss', val_loss)
            mlflow.log_metric('avg_val_acc', val_acc)
            mlflow.log_metric('avg_val_f1', val_f1)
            mlflow.log_metric('avg_train_loss', train_loss)
            mlflow.log_metric('avg_train_acc', train_acc)
            mlflow.log_metric('avg_train_f1', train_f1)

            # ------------------------------------------------------------------- EARLY STOPPING
            if self.early_stopping_count == self.early_stopping_patience:
                print('Early stopping count reached patience limit')
                break

        np.savez(
            self.model_save_path +
            'logs_results.npz',
            log_train_acc=log_train_acc,
            log_train_f1=log_train_f1,
            log_train_loss=log_train_loss,
            log_val_acc=log_train_acc,
            log_val_f1=log_val_f1,
            log_val_loss=log_val_loss)

    def test(self, models, dataloaders, criterion, logging_text=''):
        embedding_model, classifier_model = models
        test_loader = dataloaders['test']
        embedding_model = embedding_model.to(self.device)
        classifier_model = classifier_model.to(self.device)

        # ------------------------------------------------------------------- EVAL MODE
        embedding_model.eval()
        classifier_model.eval()

        test_loss = 0.0
        true_labels = np.array([])
        predicted_labels = np.array([])

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                targets = targets.view(targets.shape[0])
                inputs, targets = inputs.to(
                    self.device), targets.to(
                    self.device)

                embedding = embedding_model(inputs)
                embedding = embedding.squeeze(dim=2)
                embedding = embedding.squeeze(dim=2)
                outputs = classifier_model(embedding)
                loss = criterion(outputs, targets)

                # measure accuracy and record loss
                predicted_labels_ = torch.argmax(outputs, dim=1)
                true_labels = np.concatenate(
                    (true_labels, targets.cpu().detach().numpy()))
                predicted_labels = np.concatenate(
                    (predicted_labels, predicted_labels_.cpu().detach().numpy()))
                test_loss += loss.cpu().detach().numpy() * inputs.shape[0]

        test_acc = accuracy_score(true_labels, predicted_labels)
        test_f1 = f1_score(true_labels, predicted_labels, average='weighted')
        test_loss = test_loss / len(true_labels)

        target_names = np.arange(self.num_classes).tolist()
        target_names = [str(x) for x in target_names]

        print(
            classification_report(
                true_labels,
                predicted_labels,
                target_names=target_names))
        print('accuracy:', round(test_acc, 4))
        print('weighted F1:', round(test_f1, 4))
        print('loss:', round(test_loss, 4))
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        print(conf_matrix)
        print('\n')

        mlflow.log_metric('patch_test_acc_' + logging_text, round(test_acc, 4))
        mlflow.log_metric('patch_test_f1_' + logging_text, round(test_f1, 4))


# -----------------------------------------------------------------------------------------------------------------------
# TRoI level evaluation
# -----------------------------------------------------------------------------------------------------------------------


class MajorityVotingEvaluation:
    def __init__(self, config):
        self.num_classes = config.num_classes
        self.tumor_types = config.tumor_types
        self.class_to_idx = config.class_to_idx
        self.base_patches_path = config.base_patches_path
        self.batch_size = config.batch_size
        self.device = config.device
        self.data_transform = get_transform(
            config.patch_size,
            config.patch_scale,
            config.is_pretrained,
            is_train=False)

    def test(self, models, config, logging_text=''):
        embedding_model, classifier_model = models
        embedding_model = embedding_model.to(self.device)
        classifier_model = classifier_model.to(self.device)
        embedding_model.eval()
        classifier_model.eval()

        # ------------------------------------------------------------------- BEGIN PREDICTION
        true_labels = np.array([])
        predicted_labels = np.array([])

        with torch.no_grad():
            count = 0
            for t in range(len(self.tumor_types)):
                true_label_ = self.class_to_idx[t]

                test_ids = get_troi_ids(
                    config=config, mode='test', tumor_type=self.tumor_types[t])

                for k in range(len(test_ids)):
                    true_labels = np.append(true_labels, true_label_)

                    test_patches = []
                    patch_paths = get_patches(
                        config=config,
                        tumor_type=self.tumor_types[t],
                        troi_id=test_ids[k])
                    count += len(patch_paths)

                    for i in range(len(patch_paths)):
                        img_ = Image.open(patch_paths[i])
                        img = self.data_transform(img_)
                        img_.close()
                        test_patches.append(img)

                    test_patches = torch.stack(test_patches)
                    test_patches = test_patches.to(self.device)

                    patch_labels = np.array([])
                    for i in range(0, test_patches.shape[0], self.batch_size):
                        data = test_patches[i: i + self.batch_size, :, :, :]

                        embedding = embedding_model(data)
                        embedding = embedding.squeeze(dim=2)
                        embedding = embedding.squeeze(dim=2)
                        outputs = classifier_model(embedding)

                        patch_labels = np.append(
                            patch_labels, torch.argmax(
                                outputs, dim=1).cpu().detach().numpy())

                    pred_count = np.arange(self.num_classes)
                    for i in range(self.num_classes):
                        pred_count[i] = np.sum(patch_labels == i)
                    pred_label = np.where(pred_count == np.max(pred_count))[0]

                    if len(pred_label) == 1:
                        predicted_labels = np.append(
                            predicted_labels, pred_label)
                    else:
                        if true_label_ in pred_label:
                            predicted_labels = np.append(
                                predicted_labels, true_label_)
                        else:
                            predicted_labels = np.append(
                                predicted_labels, pred_label[0])

            print('Test patch count=', count)

        test_acc = accuracy_score(true_labels, predicted_labels)
        test_f1 = f1_score(true_labels, predicted_labels, average='weighted')

        target_names = list(np.arange(self.num_classes))
        target_names = [str(x) for x in target_names]
        cls_report = classification_report(
            true_labels, predicted_labels, target_names=target_names)
        conf_matrix = confusion_matrix(true_labels, predicted_labels)

        print('classification report:', cls_report)
        print('accuracy:', round(test_acc, 4))
        print('weighted F1:', round(test_f1, 4), '\n')
        print(conf_matrix)
        print('\n')


class Weighted_Embedding_Evaluation:
    def __init__(self, data):
        self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels = data

    def mlp_evaluate(self):
        from sklearn.neural_network import MLPClassifier
        self.clf = MLPClassifier(hidden_layer_sizes=(64,), random_state=1)
        self.clf.fit(self.train_data, self.train_labels)
        train_f1, val_f1, test_f1 = self.evaluate()
        print(
            'F1: train=', round(
                train_f1, 4), ', val=', round(
                val_f1, 4), ', test=', round(
                test_f1, 4))

        pred_test = self.clf.predict(self.test_data)
        conf_matrix = confusion_matrix(self.test_labels, pred_test)
        print(conf_matrix)

    def cross_C_evaluate(self, config):
        from sklearn.svm import SVC

        C = [1, 10, 100, 1000]
        f1_train = np.array([])
        f1_val = np.array([])
        f1_test = np.array([])
        for c in C:
            self.clf = SVC(C=c, kernel='rbf', gamma='auto')
            self.clf.fit(self.train_data, self.train_labels)
            train_f1, val_f1, test_f1 = self.evaluate()
            f1_train = np.append(f1_train, train_f1)
            f1_val = np.append(f1_val, val_f1)
            f1_test = np.append(f1_test, test_f1)

        idx = np.argmax(f1_val)
        print(
            'Best C=', C[idx], ' F1: train=', round(
                f1_train[idx], 4), ', val=', round(
                f1_val[idx], 4), ', test=', round(
                f1_test[idx], 4))

        self.clf = SVC(C=C[idx], kernel='rbf', gamma='auto')
        self.clf.fit(self.train_data, self.train_labels)
        pred_test = self.clf.predict(self.test_data)
        conf_matrix = confusion_matrix(self.test_labels, pred_test)
        print(conf_matrix)

        target_names = np.arange(config.num_classes).tolist()
        target_names = [str(x) for x in target_names]
        print(
            classification_report(
                self.test_labels,
                pred_test,
                target_names=target_names))

    def evaluate(self):
        pred_train = self.clf.predict(self.train_data)
        pred_val = self.clf.predict(self.val_data)
        pred_test = self.clf.predict(self.test_data)

        train_f1 = f1_score(self.train_labels, pred_train, average='weighted')
        val_f1 = f1_score(self.val_labels, pred_val, average='weighted')
        test_f1 = f1_score(self.test_labels, pred_test, average='weighted')

        return train_f1, val_f1, test_f1


# -----------------------------------------------------------------------------------------------------------------------
# Supporting functions
# -----------------------------------------------------------------------------------------------------------------------


def evaluate_patch(config, mode, eval, dataloaders, criterion):
    print('PATCH LEVEL TESTING: model with best validation ', mode)
    emb_model = torch.load(
        config.model_save_path +
        'embedding_model_best_' +
        mode +
        '.pt')
    classifier_model = torch.load(
        config.model_save_path +
        'classifier_model_best_' +
        mode +
        '.pt')
    eval.test(
        models=[
            emb_model,
            classifier_model],
        dataloaders=dataloaders,
        criterion=criterion,
        logging_text='model_best_' +
        mode)


def evaluate_troi_majority_voting(config, mode):
    print('TROI: Majority voting: model with best validation ', mode)
    eval = MajorityVotingEvaluation(config=config)
    emb_model = torch.load(
        config.model_save_path +
        'embedding_model_best_' +
        mode +
        '.pt')
    classifier_model = torch.load(
        config.model_save_path +
        'classifier_model_best_' +
        mode +
        '.pt')
    eval.test(
        models=[
            emb_model,
            classifier_model],
        config=config,
        logging_text='model_best_' +
        mode)


def evaluate_troi_weighted_embedding(config, mode):
    print('TROI: Weighted embedding: model with best validation ', mode)

    emb_model = torch.load(
        config.model_save_path +
        'embedding_model_best_' +
        mode +
        '.pt')
    classifier_model = torch.load(
        config.model_save_path +
        'classifier_model_best_' +
        mode +
        '.pt')
    print('Models loaded !')

    train_data, train_labels, val_data, val_labels, test_data, test_labels = troi_loaders(
        config, models=[emb_model, classifier_model], mode='f1')

    eval = Weighted_Embedding_Evaluation(
        data=[
            train_data,
            train_labels,
            val_data,
            val_labels,
            test_data,
            test_labels])
    eval.cross_C_evaluate(config)
    # eval.mlp_evaluate()
