import time
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from dataloader import *
from cnn_model import *

class Train:
    def __init__(self, config, args):
        self.config = config
        self.args = args
        self.pretrained = eval(self.args.pretrained)
        self.finetune = eval(self.args.finetune)

        self.model_save_path = self.config.model_save_path
        self.device = self.config.device

        self.get_num_classes()
        self.get_models()
        self.get_train_parameters()
        self.get_dataloaders()

    def get_num_classes(self):
        self.nuclei_types = copy.deepcopy(self.config.nuclei_types)
        self.nuclei_labels = copy.deepcopy(self.config.nuclei_labels)
        self.nuclei_types = [x.lower() for x in self.nuclei_types]

        # Remove nuclei type = 'NA'
        idx = self.nuclei_labels.index(-1)
        del self.nuclei_labels[-idx]
        del self.nuclei_types[-idx]

    def get_models(self):
        model = ModelComponents(self.config, self.args)
        self.embedding_model = model.embedding_model
        self.classification_model = model.classification_model

        for param in self.embedding_model.parameters():
            param.requires_grad = self.finetune

        self.embedding_model = self.embedding_model.to(self.device)
        self.classification_model = self.classification_model.to(self.device)

    def get_train_parameters(self):
        self.num_epochs = self.args.epochs
        self.learning_rate = self.args.lr

        self.early_stopping_min_delta = 0.0001
        self.early_stopping_patience = 20

        self.learning_rate_decay = 0.5
        self.min_learning_rate = 0.000001
        self.reduce_lr_patience = 5

    def get_dataloaders(self):
        self.dataloaders, self.class_weights = patch_loaders(self.config, self.args)

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

    def save_checkpoint(self, modelmode, epoch):
        stateE = {'epoch': epoch + 1, 'state_dict': self.embedding_model.state_dict()}
        stateC = {'epoch': epoch + 1, 'state_dict': self.classification_model.state_dict()}
        torch.save(stateE, self.model_save_path + 'embedding_model_best_' + modelmode + '.th')
        torch.save(stateC, self.model_save_path + 'classification_model_best_' + modelmode + '.th')

    def load_checkpoint(self, modelmode):
        embedding_checkpoint = torch.load(self.model_save_path + 'embedding_model_best_' + modelmode + '.th', map_location=self.device)
        classification_checkpoint = torch.load(self.model_save_path + 'classification_model_best_' + modelmode + '.th', map_location=self.device)

        components = ModelComponents(self.config, self.args)
        components.get_embedding_model()
        components.get_classification_model()

        modelE = components.embedding_model
        modelC = components.classification_model

        modelE.load_state_dict(embedding_checkpoint['state_dict'], strict=True)
        modelC.load_state_dict(classification_checkpoint['state_dict'], strict=True)

        return modelE, modelC

    def mlflow_log_checkpoint(self):
        modelmode = ['acc', 'f1', 'loss']

        for m in modelmode:
            modelE, modelC = self.load_checkpoint(m)
            mlflow.pytorch.log_model(modelE, 'embedding_model_best_' + m)
            mlflow.pytorch.log_model(modelC, 'classification_model_best_' + m)

    def mlflow_log_params(self):
        mlflow.log_params({'mode': self.args.mode})
        mlflow.log_params({'arch': self.args.arch})
        mlflow.log_params({'epochs': self.args.epochs})
        mlflow.log_params({'batch_size': self.args.batch_size})
        mlflow.log_params({'learning_rate': self.args.lr})
        mlflow.log_params({'pretrained': self.args.pretrained})
        mlflow.log_params({'finetune': self.args.finetune})
        mlflow.log_params({'weighted_loss': self.args.weighted_loss})
        mlflow.log_params({'classes': 'NormalVsAtypicalVsTumorVsStromalVsLymphocyteVsDead'})
        mlflow.log_params({'patch_size': self.config.patch_size})

    def train(self):
        print('\nTraining ...\n')

        # Loss function
        if self.args.weighted_loss:
            criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights).to(self.device)
        else:
            criterion = torch.nn.CrossEntropyLoss().to(self.device)

        # Optimizer
        if self.finetune:
            params_to_update = list(self.embedding_model.parameters()) + list(self.classification_model.parameters())
        else:
            params_to_update = list(self.classification_model.parameters())
        optimizer = optim.Adam(
            params_to_update,
            lr=self.learning_rate,
            weight_decay=0.0001)

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
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()

                embedding = self.embedding_model(inputs)
                outputs = self.classification_model(embedding)
                loss = criterion(outputs, targets)

                # measure accuracy and record loss
                pred_labels_ = torch.argmax(outputs, dim=1)
                true_labels = np.concatenate((true_labels, targets.cpu().detach().numpy()))
                pred_labels = np.concatenate((pred_labels, pred_labels_.cpu().detach().numpy()))
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
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    embedding = self.embedding_model(inputs)
                    outputs = self.classification_model(embedding)
                    loss = criterion(outputs, targets)

                    # measure accuracy and record loss
                    pred_labels_ = torch.argmax(outputs, dim=1)
                    true_labels = np.concatenate((true_labels, targets.cpu().detach().numpy()))
                    pred_labels = np.concatenate((pred_labels, pred_labels_.cpu().detach().numpy()))
                    val_loss += loss.cpu().detach().numpy() * targets.shape[0]

            val_acc = accuracy_score(true_labels, pred_labels)
            val_f1 = f1_score(true_labels, pred_labels, average='weighted')
            val_loss = val_loss / len(true_labels)

            log_val_acc = np.append(log_val_acc, val_acc)
            log_val_f1 = np.append(log_val_f1, val_f1)
            log_val_loss = np.append(log_val_loss, val_loss)

            # ------------------------------------------------------------------- SAVING CHECKPOINTS
            if val_acc > best_val_acc:
                print('val_acc improved from', round(best_val_acc, 4), 'to', round(val_acc, 4), 'saving models to', self.model_save_path)
                self.save_checkpoint(modelmode='acc', epoch=epoch)
                best_val_acc = val_acc

            if val_f1 > best_val_f1:
                print('val_f1 improved from', round(best_val_f1, 4), 'to', round(val_f1, 4), 'saving models to', self.model_save_path)
                self.save_checkpoint(modelmode='f1', epoch=epoch)
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
                print('val_loss improved from', round(best_val_loss, 4), 'to', round(val_loss, 4), 'saving models to', self.model_save_path)
                self.save_checkpoint(modelmode='loss', epoch=epoch)
                best_val_loss = val_loss

            print(' - ' + str(int(time.time() - start_time)) +
                  's - loss:', round(train_loss, 4),
                  '- acc:', round(train_acc, 4),
                  '- f1:', round(train_f1, 4),
                  '- val_loss:', round(val_loss, 4),
                  '- val_acc:', round(val_acc, 4),
                  '- val_f1:', round(val_f1, 4), '\n')

            # ------------------------------------------------------------------- MLFLOW LOGGING METRICS
            mlflow.log_metric('avg_val_acc', val_acc)
            mlflow.log_metric('avg_val_f1', val_f1)
            mlflow.log_metric('avg_val_loss', val_loss)
            mlflow.log_metric('avg_train_acc', train_acc)
            mlflow.log_metric('avg_train_f1', train_f1)
            mlflow.log_metric('avg_train_loss', train_loss)

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

        # ------------------------------------------------------------------- MLFLOW LOGGING
        self.mlflow_log_checkpoint()
        self.mlflow_log_params()

    def test(self):
        modelmode = ['acc', 'f1', 'loss']

        for m in modelmode:
            embedding_model, classification_model = self.load_checkpoint(m)

            test_loader = self.dataloaders['test']
            embedding_model = embedding_model.to(self.device)
            classification_model = classification_model.to(self.device)

            # Loss function
            criterion = torch.nn.CrossEntropyLoss().to(self.device)

            # ------------------------------------------------------------------- EVALUATE
            embedding_model.eval()
            classification_model.eval()

            test_loss = 0.0
            true_labels = np.array([])
            pred_labels = np.array([])

            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    embedding = embedding_model(inputs)
                    outputs = classification_model(embedding)
                    loss = criterion(outputs, targets)

                    # measure accuracy and record loss
                    pred_labels_ = torch.argmax(outputs, dim=1)
                    true_labels = np.concatenate((true_labels, targets.cpu().detach().numpy()))
                    pred_labels = np.concatenate((pred_labels, pred_labels_.cpu().detach().numpy()))
                    test_loss += loss.cpu().detach().numpy() * targets.shape[0]

            # ------------------------------------------------------------------- METRICS
            test_acc = round(accuracy_score(true_labels, pred_labels), 4)
            test_f1 = round(f1_score(true_labels, pred_labels, average='weighted'), 4)
            test_loss = round(test_loss / len(true_labels), 4)
            conf_matrix = confusion_matrix(true_labels, pred_labels)

            print('******************************************************************************************', m)
            print(classification_report(true_labels, pred_labels, target_names=self.nuclei_types))
            print('accuracy:', test_acc)
            print('weighted F1:', test_f1)
            print('loss:', test_loss)
            print(conf_matrix, '\n')

            plot_confusion_matrix(conf_matrix.astype(int), self.nuclei_types, self.model_save_path + 'confusion_matrix.png')

            # ------------------------------------------------------------------- LOGGING
            mlflow.log_metric('test_acc_best_' + m, test_acc)
            mlflow.log_metric('test_f1_best_' + m, test_f1)
            mlflow.log_artifact(self.model_save_path + 'confusion_matrix.png', artifact_path='confusion_matrix_best_' + m)
