import time
import mlflow
import torch.optim as optim
from tqdm import tqdm
#from histocartography.data_generation.nuclei_features.models.dataloader import *
#from histocartography.data_generation.nuclei_features.models.denseVAE import *
from dataloader_patch_single_scale import *
from denseVAE import *


# -----------------------------------------------------------------------------------------------------------------------
# Patch level evaluation
# -----------------------------------------------------------------------------------------------------------------------

class Patch_Evaluation:
    def __init__(self, config):
        self.num_epochs = config.num_epochs
        self.learning_rate = config.learning_rate
        self.device = config.device
        self.model_save_path = config.model_save_path
        self.kl_weight = config.kl_weight

        self.early_stopping_min_delta = config.early_stopping_min_delta
        self.early_stopping_patience = config.early_stopping_patience

        self.learning_rate_decay = config.learning_rate_decay
        self.min_learning_rate = config.min_learning_rate
        self.reduce_lr_patience = config.reduce_lr_patience

        # define model
        self.vae = dense_vae(
            embedding_dim=config.embedding_dim,
            encoder_layers_per_block=config.encoder_layers_per_block,
            patch_size=config.patch_size,
            device=self.device)
        #summary(self.vae, (3, 72, 72))

        # define data loaders
        dataloaders = patch_loaders(config)
        self.train_loader = dataloaders['train']
        self.valid_loader = dataloaders['val']
        self.test_loader = dataloaders['test']

        # define optimizer
        self.optimizer = optim.Adam(
            self.vae.parameters(),
            lr=self.learning_rate,
            weight_decay=0.0001)

        self.is_mask = config.is_mask
    # enddef

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
        # endif

        if self.learning_rate <= self.min_learning_rate:
            print('Minimum learning rate reached')
            self.min_reduce_lr_reached = True
        # endif
        return optimizer
    # enddef

    # Reconstruction + KL divergence losses
    def loss_function(self, x, x_reconstructed, mu, logvar):
        #reconstuction_loss = torch.nn.BCELoss(reduction='mean')(x_reconstructed, x)
        reconstuction_loss = torch.nn.BCELoss(
            reduction='sum')(
            x_reconstructed,
            x) / x.shape[0]
        kl_divergence_loss = ((mu ** 2 + logvar.exp() - 1 - logvar) / 2).mean()

        total_loss = reconstuction_loss + self.kl_weight * kl_divergence_loss
        return reconstuction_loss, kl_divergence_loss, total_loss
    # enddef

    def train(self):
        print('\nTRAINING ...\n')
        model = self.vae.to(self.device)

        self.reduce_lr_count = 0
        self.early_stopping_count = 0
        self.min_reduce_lr_reached = False

        best_val_loss = float('Inf')
        log_train_recon_loss = np.array([])
        log_train_kl_loss = np.array([])
        log_train_loss = np.array([])
        log_val_recon_loss = np.array([])
        log_val_kl_loss = np.array([])
        log_val_loss = np.array([])

        for epoch in range(self.num_epochs):
            start_time = time.time()
            print('Epoch {}/{}'.format(epoch + 1, self.num_epochs))

            self.adjust_learning_rate(optimizer=self.optimizer)
            if self.min_reduce_lr_reached:
                break

            # ------------------------------------------------------------------- TRAIN MODE
            model.train()
            patch_count = 0
            train_recon_loss = 0.0
            train_kl_loss = 0.0
            train_loss = 0.0

            for (patch, mask, _) in tqdm(self.train_loader, unit='batch'):
                patch, mask = patch.to(self.device), mask.to(self.device)
                n_patches = patch.shape[0]

                # mask input patches
                patch_masked = torch.mul(
                    patch, mask) if self.is_mask else patch

                # flush gradients and run the model forward
                self.optimizer.zero_grad()

                x_reconstructed, mu, logvar = model(patch_masked)
                recon_loss, kl_loss, loss = self.loss_function(
                    x=patch_masked, x_reconstructed=x_reconstructed, mu=mu, logvar=logvar)

                train_recon_loss += recon_loss.cpu().detach().numpy() * n_patches
                train_kl_loss += kl_loss.cpu().detach().numpy() * n_patches
                train_loss += loss.cpu().detach().numpy() * n_patches

                #train_recon_loss += recon_loss.item()
                #train_kl_loss += kl_loss.item()
                #train_loss += loss.item() * n_patches

                patch_count += n_patches

                # backprop gradients from the loss
                loss.backward()
                self.optimizer.step()
            # endfor
            train_recon_loss = train_recon_loss / patch_count
            train_kl_loss = train_kl_loss / patch_count
            train_loss = train_loss / patch_count

            log_train_recon_loss = np.append(
                log_train_recon_loss, train_recon_loss)
            log_train_kl_loss = np.append(log_train_kl_loss, train_kl_loss)
            log_train_loss = np.append(log_train_loss, train_loss)

            # ------------------------------------------------------------------- EVAL MODE
            model.eval()
            patch_count = 0
            val_recon_loss = 0.0
            val_kl_loss = 0.0
            val_loss = 0.0

            with torch.no_grad():
                for (patch, mask, _) in self.valid_loader:
                    patch, mask = patch.to(self.device), mask.to(self.device)
                    n_patches = patch.shape[0]

                    # mask input patches
                    patch_masked = torch.mul(
                        patch, mask) if self.is_mask else patch

                    x_reconstructed, mu, logvar = model(patch_masked)
                    recon_loss, kl_loss, loss = self.loss_function(
                        x=patch_masked, x_reconstructed=x_reconstructed, mu=mu, logvar=logvar)

                    val_recon_loss += recon_loss.cpu().detach().numpy() * n_patches
                    val_kl_loss += kl_loss.cpu().detach().numpy() * n_patches
                    val_loss += loss.cpu().detach().numpy() * n_patches

                    #val_recon_loss += recon_loss.item()
                    #val_kl_loss += kl_loss.item()
                    #val_loss += loss.item() * n_patches

                    patch_count += n_patches
                # endfor
            # end
            val_recon_loss = val_recon_loss / patch_count
            val_kl_loss = val_kl_loss / patch_count
            val_loss = val_loss / patch_count

            log_val_recon_loss = np.append(log_val_recon_loss, val_recon_loss)
            log_val_kl_loss = np.append(log_val_kl_loss, val_kl_loss)
            log_val_loss = np.append(log_val_loss, val_loss)

            # ------------------------------------------------------------------- SAVING
            if val_loss < best_val_loss:
                print(
                    'val_loss improved from', round(
                        best_val_loss, 4), 'to', round(
                        val_loss, 4), 'saving models to', self.model_save_path)
                torch.save(
                    model,
                    self.model_save_path +
                    'vae_model_best_loss.pt')

                if (best_val_loss - val_loss) < self.early_stopping_min_delta:
                    self.early_stopping_count += 1
                else:
                    self.early_stopping_count = 0
                # endif
                best_val_loss = val_loss
                self.reduce_lr_count = 0
            else:
                self.reduce_lr_count += 1
                self.early_stopping_count += 1
            # endif

            print(' - ' + str(int(time.time() - start_time)) + 's '
                  '- loss:',
                  round(train_loss,
                        4),
                  '- rl:',
                  round(train_recon_loss,
                        4),
                  '- kl:',
                  round(train_kl_loss,
                        4),
                  '- val_loss:',
                  round(val_loss,
                        4),
                  '- val_rl:',
                  round(val_recon_loss,
                        4),
                  '- val_kl:',
                  round(val_kl_loss,
                        4),
                  '- e:',
                  self.early_stopping_count,
                  '\n')

            # ------------------------------------------------------------------- MLFLOW LOGGING
            # '''
            mlflow.log_metric('avg_train_recon_loss', train_recon_loss)
            mlflow.log_metric('avg_train_kl_loss', train_kl_loss)
            mlflow.log_metric('avg_train_loss', train_loss)

            mlflow.log_metric('avg_val_recon_loss', val_recon_loss)
            mlflow.log_metric('avg_val_kl_loss', val_kl_loss)
            mlflow.log_metric('avg_val_loss', val_loss)
            # '''

            # ------------------------------------------------------------------- EARLY STOPPING
            if self.early_stopping_count == self.early_stopping_patience:
                print('Early stopping count reached patience limit')
                break
            # endif
        # endfor

        np.savez(
            self.model_save_path +
            'logs_results.npz',
            log_train_loss=log_train_loss,
            log_val_loss=log_val_loss)
    # enddef

    # Only for evaluating loss values.
    def test(self):
        print('\nTESTING ...\n')
        self.vae = torch.load(self.model_save_path + 'vae_model_best_loss.pt')
        model = self.vae.to(self.device)

        # ------------------------------------------------------------------- EVAL MODE
        model.eval()
        patch_count = 0
        test_recon_loss = 0.0
        test_kl_loss = 0.0
        test_loss = 0.0

        with torch.no_grad():
            for (patch, mask, _) in self.test_loader:
                patch, mask = patch.to(self.device), mask.to(self.device)
                n_patches = patch.shape[0]

                # mask input patches
                patch_masked = torch.mul(
                    patch, mask) if self.is_mask else patch

                x_reconstructed, mu, logvar = model(patch_masked)
                recon_loss, kl_loss, loss = self.loss_function(
                    x=patch_masked, x_reconstructed=x_reconstructed, mu=mu, logvar=logvar)

                test_recon_loss += recon_loss.cpu().detach().numpy() * n_patches
                test_kl_loss += kl_loss.cpu().detach().numpy() * n_patches
                test_loss += loss.cpu().detach().numpy() * n_patches
                patch_count += n_patches
            # endfor
        # end
        test_recon_loss = test_recon_loss / patch_count
        test_kl_loss = test_kl_loss / patch_count
        test_loss = test_loss / patch_count

        print(
            'loss:',
            round(
                test_loss,
                4),
            '- rl:',
            round(
                test_recon_loss,
                4),
            '- kl:',
            round(
                test_kl_loss,
                4),
            '\n')

        '''
        mlflow.log_metric('avg_test_recon_loss', round(test_recon_loss, 4))
        mlflow.log_metric('avg_test_kl_loss', round(test_kl_loss, 4))
        mlflow.log_metric('avg_test_loss', round(test_loss, 4))
        #'''
    # enddef
# end
