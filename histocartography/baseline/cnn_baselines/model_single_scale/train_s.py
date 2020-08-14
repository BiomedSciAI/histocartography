import time
import numpy as np
import mlflow
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from cnn_model import *
from dataloader_patch_single_scale import *

class Train:
    def __init__(self, config):
        self.base_img_path = config.base_img_path
        self.base_patches_path = config.base_patches_path
        self.model_save_path = config.model_save_path
        self.is_finetune = config.is_finetune
        self.num_classes = config.num_classes
        self.device = config.device

        self.get_models(config)
        self.get_train_parameters(config)
        self.get_dataloaders(config)


    def get_dataloaders(self, config):
        self.dataloaders = patch_loaders(config)


    def get_models(self, config):
        model = ModelComponents(config)
        self.embedding_model = model.embedding_model
        self.classification_model = model.classification_model

        for param in self.embedding_model.parameters():
            param.requires_grad = self.is_finetune

        self.embedding_model = self.embedding_model.to(self.device)
        self.classification_model = self.classification_model.to(self.device)


    def get_train_parameters(self, config):
        self.num_epochs = config.num_epochs
        self.learning_rate = config.learning_rate

        self.early_stopping_min_delta = config.early_stopping_min_delta
        self.early_stopping_patience = config.early_stopping_patience

        self.learning_rate_decay = config.learning_rate_decay
        self.min_learning_rate = config.min_learning_rate
        self.reduce_lr_patience = config.reduce_lr_patience


    def adjust_learning_rate(self, optimizer):
        lr_prev = self.learning_rate

        if self.reduce_lr_count == self.reduce_lr_patience:
            self.learning_rate *= self.learning_rate_decay
            self.reduce_lr_count = 0
            print('Reducing learning rate from', round(lr_prev, 6), 'to', round(self.learning_rate, 6))

            for param_group in optimizer.param_groups:
                param_group['lr'] = self.learning_rate

        if self.learning_rate <= self.min_learning_rate:
            print('Minimum learning rate reached')
            self.min_reduce_lr_reached = True

        return optimizer


    def train(self):
        print('\nTraining ...\n')

        # Loss function
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        # Optimizer
        if self.is_finetune:
            params_to_update = list(self.embedding_model.parameters()) + list(self.classification_model.parameters())
        else:
            params_to_update = list(self.classification_model.parameters())
        optimizer = optim.Adam(params_to_update, lr=self.learning_rate, weight_decay=0.0001)

        # Dataloader
        train_loader = self.dataloaders['train']
        valid_loader = self.dataloaders['val']

        # Loggers
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
            self.embedding_model.train()
            self.classification_model.train()

            train_loss = 0.0
            true_labels = np.array([])
            pred_labels = np.array([])

            for (inputs, targets) in tqdm(train_loader, unit='batch'):
                targets = targets.view(targets.shape[0])
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()

                embedding = self.embedding_model(inputs)
                embedding = embedding.squeeze(dim=2)
                embedding = embedding.squeeze(dim=2)
                outputs = self.classification_model(embedding)
                loss = criterion(outputs, targets)

                # measure accuracy and record loss
                pred_labels_ = torch.argmax(outputs, dim=1)
                true_labels = np.concatenate(
                    (true_labels, targets.cpu().detach().numpy()))
                pred_labels = np.concatenate(
                    (pred_labels, pred_labels_.cpu().detach().numpy()))
                train_loss += loss.cpu().detach().numpy() * targets.shape[0]

                loss.backward()
                optimizer.step()

            train_acc = accuracy_score(true_labels, pred_labels)
            train_f1 = f1_score(true_labels, pred_labels, average='weighted')
            train_loss = train_loss / len(true_labels)

            log_train_acc = np.append(log_train_acc, train_acc)
            log_train_f1 = np.append(log_train_f1, train_f1)
            log_train_loss = np.append(log_train_loss, train_loss)

            # ------------------------------------------------------------------- EVAL MODE
            self.embedding_model.eval()
            self.classification_model.eval()

            val_loss = 0.0
            true_labels = np.array([])
            pred_labels = np.array([])

            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(valid_loader):
                    targets = targets.view(targets.shape[0])
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    embedding = self.embedding_model(inputs)
                    embedding = embedding.squeeze(dim=2)
                    embedding = embedding.squeeze(dim=2)
                    outputs = self.classification_model(embedding)
                    loss = criterion(outputs, targets)

                    # measure accuracy and record loss
                    pred_labels_ = torch.argmax(outputs, dim=1)
                    true_labels = np.concatenate(
                        (true_labels, targets.cpu().detach().numpy()))
                    pred_labels = np.concatenate(
                        (pred_labels, pred_labels_.cpu().detach().numpy()))
                    val_loss += loss.cpu().detach().numpy() * targets.shape[0]

            val_acc = accuracy_score(true_labels, pred_labels)
            val_f1 = f1_score(true_labels, pred_labels, average='weighted')
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
                    self.embedding_model,
                    self.model_save_path +
                    'embedding_model_best_acc.pt')
                torch.save(
                    self.classification_model,
                    self.model_save_path +
                    'classification_model_best_acc.pt')
                best_val_acc = val_acc

            if val_f1 > best_val_f1:
                print(
                    'val_f1 improved from', round(
                        best_val_f1, 4), 'to', round(
                        val_f1, 4), 'saving models to', self.model_save_path)
                torch.save(
                    self.embedding_model,
                    self.model_save_path +
                    'embedding_model_best_f1.pt')
                torch.save(
                    self.classification_model,
                    self.model_save_path +
                    'classification_model_best_f1.pt')

                if (val_f1 - best_val_f1) < self.early_stopping_min_delta:
                    self.early_stopping_count += 1
                else:
                    self.early_stopping_count = 0

                best_val_f1 = val_f1
                self.reduce_lr_count = 0
            else:
                self.reduce_lr_count += 1
                self.early_stopping_count += 1

            if val_loss < best_val_loss:
                print(
                    'val_loss improved from', round(
                        best_val_loss, 4), 'to', round(
                        val_loss, 4), 'saving models to', self.model_save_path)
                torch.save(
                    self.embedding_model,
                    self.model_save_path +
                    'embedding_model_best_loss.pt')
                torch.save(
                    self.classification_model,
                    self.model_save_path +
                    'classification_model_best_loss.pt')
                best_val_loss = val_loss

            print(' - ' + str(int(time.time() - start_time)) +
                  's - loss:', round(train_loss, 4),
                  '- acc:', round(train_acc, 4),
                  '- val_loss:', round(val_loss, 4),
                  '- val_acc:', round(val_acc,4),
                  '\n')

            # ------------------------------------------------------------------- MLFLOW LOGGING
            '''
            mlflow.log_metric('avg_val_loss', val_loss)
            mlflow.log_metric('avg_val_acc', val_acc)
            mlflow.log_metric('avg_val_f1', val_f1)
            mlflow.log_metric('avg_train_loss', train_loss)
            mlflow.log_metric('avg_train_acc', train_acc)
            mlflow.log_metric('avg_train_f1', train_f1)
            #'''

            # ------------------------------------------------------------------- EARLY STOPPING
            if self.early_stopping_count == self.early_stopping_patience:
                print('Early stopping count reached patience limit')
                break

        np.savez(
            self.model_save_path +
            'logs_training.npz',
            log_train_acc=log_train_acc,
            log_train_f1=log_train_f1,
            log_train_loss=log_train_loss,
            log_val_acc=log_train_acc,
            log_val_f1=log_val_f1,
            log_val_loss=log_val_loss)


    def test(self, modelmode, logging_text=''):
        embedding_model = torch.load(self.model_save_path + 'embedding_model_best_' + modelmode + '.pt')
        classification_model = torch.load(self.model_save_path + 'classification_model_best_' + modelmode + '.pt')

        test_loader = self.dataloaders['test']
        embedding_model = embedding_model.to(self.device)
        classification_model = classification_model.to(self.device)

        # Loss function
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        # ------------------------------------------------------------------- EVAL MODE
        embedding_model.eval()
        classification_model.eval()

        test_loss = 0.0
        true_labels = np.array([])
        pred_labels = np.array([])

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                targets = targets.view(targets.shape[0])
                inputs, targets = inputs.to(
                    self.device), targets.to(
                    self.device)

                embedding = embedding_model(inputs)
                embedding = embedding.squeeze(dim=2)
                embedding = embedding.squeeze(dim=2)
                outputs = classification_model(embedding)
                loss = criterion(outputs, targets)

                # measure accuracy and record loss
                pred_labels_ = torch.argmax(outputs, dim=1)
                true_labels = np.concatenate(
                    (true_labels, targets.cpu().detach().numpy()))
                pred_labels = np.concatenate(
                    (pred_labels, pred_labels_.cpu().detach().numpy()))
                test_loss += loss.cpu().detach().numpy() * targets.shape[0]

        test_acc = accuracy_score(true_labels, pred_labels)
        test_f1 = f1_score(true_labels, pred_labels, average='weighted')
        test_loss = test_loss / len(true_labels)

        target_names = np.arange(self.num_classes).tolist()
        target_names = [str(x) for x in target_names]

        print(
            classification_report(
                true_labels,
                pred_labels,
                target_names=target_names))
        print('accuracy:', round(test_acc, 4))
        print('weighted F1:', round(test_f1, 4))
        print('loss:', round(test_loss, 4))
        conf_matrix = confusion_matrix(true_labels, pred_labels)
        print(conf_matrix)
        print('\n')

        mlflow.log_metric('patch_test_acc_' + logging_text, round(test_acc, 4))
        mlflow.log_metric('patch_test_f1_' + logging_text, round(test_f1, 4))

