from utils import experimental_utils, utils
from datasets.linear import generate_linear_example_ab
from datasets.multiple_lotka_volterra import MultiLotkaVolterra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import senn, usad, inet, lstm, RecAD_model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn import preprocessing
import torch.utils.data as data_utils
from tqdm import tqdm
from statsmodels.tsa.api import VAR
import random
import os

class RecAD_p2(object):
    def __init__(self, options):
        super().__init__()
        self.options = options
        self.K = options['K']
        self.ad_model_K = options['ad_model_K']
        self.hidden_layer_size = options['hidden_layer_size']
        self.num_epochs = options['num_epochs']
        self.batch_size = options['batch_size']
        self.seed = options['seed']
        self.num_hidden_layers = options['num_hidden_layers']
        self.initial_lr = options['initial_lr']
        self.beta_1 = options['beta_1']
        self.beta_2 = options['beta_2']
        self.lmbd = options['lambda']
        self.gamma = options['gamma']
        self.dataset_name = options['dataset_name']
        self.T = options['T']
        self.device = options['device']
        self.quantile = options['quantile']
        self.training_size = options['training_size']
        self.testing_size = options['testing_size']
        self.training_gvar = options['training_gvar']
        self.training_ad_model = options['training_ad_model']
        self.ad_model_hidden_size = options['ad_model_hidden_size']
        self.ad_model_n_epochs = options['ad_model_n_epochs']
        self.ad_model_batch_size = options['ad_model_batch_size']
        self.ad_model_alpha = options['ad_model_alpha']
        self.ad_model_beta = options['ad_model_beta']
        self.ad_model_downsampling = options['ad_downsampling']
        self.recourse_model_max_epoch = options['recourse_model_max_epoch']
        self.recourse_model_lr = options['recourse_model_lr']
        self.recourse_model_training = options['recourse_model_training']
        self.recourse_model_alpha = options['recourse_model_alpha']
        self.recourse_model_beta = options['recourse_model_beta']
        self.recourse_model_gamma = options['recourse_model_gamma']
        self.recourse_model_early_stop = options['recourse_model_early_stop']
        self.recourse_model_hidden_dim = options['recourse_model_hidden_dim']
        self.preprocessing = options['preprocessing_data']

        self.get_baseline_GVAR = options['get_baseline_GVAR']
        self.get_baseline_VAR = options['get_baseline_VAR']
        self.get_baseline_fc = options['get_baseline_fc']
        self.get_baseline_linear = options['get_baseline_linear']
        self.get_baseline_lstm = options['get_baseline_lstm']

        self.loss_mse_none_reduce = nn.MSELoss(reduce=False)
        self.loss_mse_reduce = nn.MSELoss()
        self._get_data()
        self._get_ad_model()
        self._get_mse()
        self._anomaly_detection()
        self._get_gvar()
        self._get_recourse_model()

        if self.dataset_name == 'MSDS':
            _, _, _, _, _, _ = self._get_recourse_model_results()
        else:
            _, _, _, _, _ = self._get_recourse_model_results()


    def _get_data(self):
        print('Preprocessing the dataset.')
        if self.dataset_name == 'linear':
            self.p = 4
            self.adlength = self.options['adlength']
            self.adtype = self.options['adtype']
            self.save_parm = self.options['dataset_name'] + '_' + str(self.options['adlength']) + '_' + str(self.options['adtype'])
            if self.preprocessing == 1:
                x_n_list, x_ab_list, eps_n_list, eps_ab_list, causal_struct, causal_struct_value, signed_causal_struct, label_list, a = \
                    generate_linear_example_ab(n=int(self.training_size +  self.testing_size), t=self.T, mul=2, seed=self.seed, adlength=self.adlength, adtype=self.adtype)
                np.save(f'data/Linear_{self.save_parm}_x_n_list', x_n_list)
                np.save(f'data/Linear_{self.save_parm}_eps_n_list', eps_n_list)
                np.save(f'data/Linear_{self.save_parm}_x_ab_list', x_ab_list)
                np.save(f'data/Linear_{self.save_parm}_eps_ab_list', eps_ab_list)
                np.save(f'data/Linear_{self.save_parm}_label_list', label_list)
                np.save(f'data/Linear_{self.save_parm}_causal_struct', causal_struct_value)
                np.save(f'data/Linear_{self.save_parm}_a', a)
            else:
                x_n_list = np.load(f'data/Linear_{self.save_parm}_x_n_list.npy')
                eps_n_list = np.load(f'data/Linear_{self.save_parm}_eps_n_list.npy')
                x_ab_list = np.load(f'data/Linear_{self.save_parm}_x_ab_list.npy')
                eps_ab_list = np.load(f'data/Linear_{self.save_parm}_eps_ab_list.npy')
                label_list = np.load(f'data/Linear_{self.save_parm}_label_list.npy')
                causal_struct_value = np.load(f'data/Linear_{self.save_parm}_causal_struct.npy')
                a = np.load(f'data/Linear_{self.save_parm}_a.npy')

            self.a = a
            self.train_x = x_n_list[:self.training_size]
            self.train_x_concate = np.concatenate(self.train_x, axis=0)
            self.mean = np.mean(self.train_x_concate, axis=0)
            self.std = np.std(self.train_x_concate, axis=0)
            self.train_x_norm = (self.train_x - self.mean) / self.std

            self.train_eps = eps_n_list[:self.training_size]
            self.train_eps_norm = self.train_eps / self.std
            self.test_eps = eps_ab_list[self.training_size:]
            self.test_eps_norm = self.test_eps / self.std

            self.test_x = x_ab_list[self.training_size:]
            self.test_x_norm = (self.test_x - self.mean) / self.std

            self.test_label = label_list[self.training_size:]

        elif self.dataset_name == 'MSDS':
            self.p = 10
            self.save_parm = self.options['dataset_name']
            if self.preprocessing == 1:
                dataset_folder = 'data/MSDS'
                df_train = pd.read_csv(os.path.join(dataset_folder, 'train.csv'))
                df_test = pd.read_csv(os.path.join(dataset_folder, 'test.csv'))
                df_train, df_test = df_train.values[::5, 1:], df_test.values[::5, 1:]
                labels = pd.read_csv(os.path.join(dataset_folder, 'labels.csv'))
                labels = labels.values[::1, 1:]
                self.all_label = labels
                labels = np.max(labels, axis=1)
                self.train_x = df_train.astype(np.float32)
                self.mean = np.mean(self.train_x, axis=0)
                self.std = np.std(self.train_x, axis=0)
                self.train_x_norm = ((self.train_x - self.mean) / self.std)
                self.test_x = df_test.astype(np.float32)
                self.test_x_norm = (self.test_x - self.mean) / self.std
                self.test_label = labels.astype(np.float32)

                np.save('data/MSDS_train_x', self.train_x)
                np.save('data/MSDS_train_x_norm', self.train_x_norm)
                np.save('data/MSDS_test_x', self.test_x)
                np.save('data/MSDS_test_x_norm', self.test_x_norm)
                np.save('data/MSDS_test_label', self.test_label)
                np.save('data/MSDS_all_label', self.all_label)
            else:
                self.train_x = np.load('data/MSDS_train_x.npy')
                self.train_x_norm = np.load('data/MSDS_train_x_norm.npy')
                self.test_x = np.load('data/MSDS_test_x.npy')
                self.test_x_norm = np.load('data/MSDS_test_x_norm.npy')
                self.test_label = np.load('data/MSDS_test_label.npy')
                self.all_label = np.load('data/MSDS_all_label.npy')

        elif self.dataset_name == 'SWaT':
            self.p = 51
            if self.preprocessing == 1:
                normal = pd.read_excel('data/SWaT_Dataset_Normal_v1.xlsx', header=1)
                normal = normal.drop([" Timestamp", "Normal/Attack"], axis=1)
                index = pd.date_range('1/1/2000', periods=len(normal), freq='S')
                normal.index = index
                normal = normal.resample('5S').mean()
                normal.reset_index(drop=True, inplace=True)
                standard_scaler = preprocessing.StandardScaler()
                self.train_x = normal.values
                self.train_x_norm = standard_scaler.fit_transform(self.train_x)

                attack = pd.read_excel("data/SWaT_Dataset_Attack_v0.xlsx", header=1)
                labels = [int(label != 'Normal') for label in attack["Normal/Attack"].values]
                attack["Normal/Attack"] = np.array(labels).reshape(-1, 1)
                attack = attack.drop([" Timestamp"], axis=1)
                index = pd.date_range('1/1/2000', periods=len(attack), freq='S')
                attack.index = index
                attack = attack.resample('5S').mean()
                labels = [1 if i > 0 else 0 for i in attack["Normal/Attack"].values]
                attack.drop(["Normal/Attack"], axis=1, inplace=True)
                attack.reset_index(drop=True, inplace=True)
                self.test_x = attack.values
                self.test_x_norm = standard_scaler.transform(self.test_x)
                self.test_label = labels

                np.save('data/SWaT_train_x', self.train_x)
                np.save('data/SWaT_train_x_norm', self.train_x_norm)
                np.save('data/SWaT_test_x', self.test_x)
                np.save('data/SWaT_test_x_norm', self.test_x_norm)
                np.save('data/SWaT_test_label', self.test_label)
            else:
                self.train_x = np.load('data/SWaT_train_x.npy')
                self.train_x_norm = np.load('data/SWaT_train_x_norm.npy')
                self.test_x = np.load('data/SWaT_test_x.npy')
                self.test_x_norm = np.load('data/SWaT_test_x_norm.npy')
                self.test_label = np.load('data/SWaT_test_label.npy')

        elif self.dataset_name == 'lotka-volterra':
            self.p = 20
            self.adlength = self.options['adlength']
            self.adtype = self.options['adtype']
            self.save_parm = self.options['dataset_name'] + '_' + str(self.options['adlength']) + '_' + str(self.options[
                'adtype'])
            self.d = self.options['d']
            self.alpha_lv = self.options['alpha_lv']
            self.beta_lv = self.options['beta_lv']
            self.gamma_lv = self.options['gamma_lv']
            self.delta_lv = self.options['delta_lv']
            self.sigma_lv = self.options['sigma_lv']
            self.downsample_factor = self.options['downsample_factor']
            self.dt = self.options['dt']

            self.mlv = MultiLotkaVolterra(p=int(self.p / 2), d=self.d, alpha=self.alpha_lv, beta=self.beta_lv,
                                     gamma=self.gamma_lv,
                                     delta=self.delta_lv, sigma=self.sigma_lv, adlength=self.adlength, adtype=self.adtype)
            if self.preprocessing == 1:
                x_n_list, eps_n_list, x_ab_list, eps_ab_list, label_list, causal_struct, _ = self.mlv.simulate(n=int(self.training_size +
                                                                               self.testing_size),
                                                                         t=self.T, downsample_factor=self.downsample_factor,
                                                                         dt=self.dt, seed=self.seed)
                np.save(f'data/LV_{self.save_parm}_x_n_list', x_n_list)
                np.save(f'data/LV_{self.save_parm}_eps_n_list', eps_n_list)
                np.save(f'data/LV_{self.save_parm}_x_ab_list', x_ab_list)
                np.save(f'data/LV_{self.save_parm}_eps_ab_list', eps_ab_list)
                np.save(f'data/LV_{self.save_parm}_label_list', label_list)
                np.save(f'data/LV_{self.save_parm}_causal_struct', causal_struct)
            else:
                x_n_list = np.load(f'data/LV_{self.save_parm}_x_n_list.npy')
                eps_n_list = np.load(f'data/LV_{self.save_parm}_eps_n_list.npy')
                x_ab_list = np.load(f'data/LV_{self.save_parm}_x_ab_list.npy')
                eps_ab_list = np.load(f'data/LV_{self.save_parm}_eps_ab_list.npy')
                label_list = np.load(f'data/LV_{self.save_parm}_label_list.npy')
                causal_struct = np.load(f'data/LV_{self.save_parm}_causal_struct.npy')

            self.a = causal_struct
            self.train_x = x_n_list[:self.training_size]
            self.train_x_concate = np.concatenate(self.train_x, axis=0)
            self.mean = np.mean(self.train_x_concate, axis=0)
            self.std = np.std(self.train_x_concate, axis=0)
            self.train_x_norm = (self.train_x - self.mean) / self.std

            self.train_eps = eps_n_list[:self.training_size]
            self.train_eps_norm = self.train_eps / self.std
            self.test_eps = eps_ab_list[self.training_size:]
            self.test_eps_norm = self.test_eps / self.std

            self.test_x = x_ab_list[self.training_size:]
            self.test_x_norm = (self.test_x - self.mean) / self.std

            self.test_label = label_list[self.training_size:]
        else:
            NotImplementedError
        print('Finished preprocessing the dataset.')

    def _get_ad_model(self):
        w_size = self.ad_model_K * self.p
        z_size = self.ad_model_K * self.ad_model_hidden_size
        self.ad_model = usad.UsadModel(w_size, z_size, device=self.device)
        if self.training_ad_model:
            print('Training AD model.')
            self.ad_model.to(self.device)
            windows, _ = utils.sliding_window(self.train_x_norm, get_ys=0, y=None,
                                              size=self.ad_model_K, downsampling=self.ad_model_downsampling)
            windows_normal_train = windows[:int(np.floor(.8 * windows.shape[0]))]
            windows_normal_val = windows[int(np.floor(.8 * windows.shape[0])):]
            train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
                torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0], w_size]))
            ), batch_size=self.ad_model_batch_size, shuffle=False, num_workers=0)

            val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
                torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0], w_size]))
            ), batch_size=self.ad_model_batch_size, shuffle=False, num_workers=0)
            _ = usad.training(self.ad_model_n_epochs, self.ad_model, train_loader, val_loader, device=self.device)
            self.ad_model.eval()
            results = usad.testing(self.ad_model, val_loader, device=self.device)
            if len(results) > 1:
                self.y_pred = np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                                         results[-1].flatten().detach().cpu().numpy()])
            else:
                self.y_pred = results[0].flatten().detach().cpu().numpy()
            torch.save(self.ad_model.state_dict(), f'./saved_models/USAD_{self.save_parm}.pt')
            np.save(f'results/y_{self.save_parm}_pred', self.y_pred)
        else:
            print('Loading AD model.')
            self.ad_model.load_state_dict(torch.load(f'./saved_models/USAD_{self.save_parm}.pt', map_location=self.device))
            self.ad_model.to(self.device)
            self.ad_model.eval()
            self.y_pred = np.load(f'results/y_{self.save_parm}_pred.npy')

    def _get_mse(self):
        if self.quantile > 1:
            self.mse = np.float32(max(self.y_pred) * self.quantile)
        else:
            self.mse = np.float32(np.quantile(self.y_pred, self.quantile))

    def _anomaly_detection(self):
        print('Results for anomaly detection.')
        if self.dataset_name == 'linear':
            w_size = self.ad_model_K * self.p
            windows, y_true_ab = utils.sliding_window(self.test_x_norm, get_ys=0,
                                                   y=self.test_label,
                                                   size=self.ad_model_K, downsampling=1)
            test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
                torch.from_numpy(windows).float().view(([windows.shape[0], w_size]))
            ), batch_size=self.ad_model_batch_size, shuffle=False, num_workers=0)
            self.ad_model.eval()
            with torch.no_grad():
                results = usad.testing(self.ad_model, test_loader, alpha=self.ad_model_alpha, beta=self.ad_model_beta, device=self.device)
            y_pred = np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                                     results[-1].flatten().detach().cpu().numpy()])
            y_pred_label_ab = [int(i > self.mse) for i in y_pred]
            self.data = windows[np.array(y_pred_label_ab) == 1]
            self.data_ab = windows[np.array(y_true_ab) == 1]
            y_true = y_true_ab
            y_pred = y_pred_label_ab
            print(classification_report(y_true=y_true, y_pred=y_pred, digits=5))
            print(confusion_matrix(y_true=y_true, y_pred=y_pred))
            print('Anomaly Detection AUC-ROC: {:.5f}'.format(roc_auc_score(y_true, y_pred)))
            print('Anomaly Detection AUC-PR: {:.5f}'.format(average_precision_score(y_true, y_pred)))

        elif self.dataset_name == 'MSDS':
            w_size = self.ad_model_K * self.p
            windows, y_true = utils.sliding_window(self.test_x_norm, get_ys=0, y=self.test_label,
                                              size=self.ad_model_K, downsampling=1)
            test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
                torch.from_numpy(windows).float().view(([windows.shape[0], w_size]))
                ), batch_size=self.ad_model_batch_size, shuffle=False, num_workers=0)
            self.ad_model.eval()
            with torch.no_grad():
                results = usad.testing(self.ad_model, test_loader, alpha=self.ad_model_alpha, beta=self.ad_model_beta, device=self.device)
            y_pred = np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                                     results[-1].flatten().detach().cpu().numpy()])
            y_pred_label = [int(i > self.mse) for i in y_pred]
            self.data = windows[np.array(y_pred_label)==1]
            self.data_ab = windows[np.array(y_true)==1]
            print(classification_report(y_true=y_true, y_pred=y_pred_label, digits=5))
            print(confusion_matrix(y_true=y_true, y_pred=y_pred_label))
            print('Anomaly Detection AUC-ROC: {:.5f}'.format(roc_auc_score(y_true, y_pred)))
            print('Anomaly Detection AUC-PR: {:.5f}'.format(average_precision_score(y_true, y_pred)))

        elif self.dataset_name == 'lotka-volterra':
            w_size = self.ad_model_K * self.p
            # ab samples
            windows, y_true_ab = utils.sliding_window(self.test_x_norm, get_ys=0,
                                                      y=self.test_label,
                                                      size=self.ad_model_K, downsampling=1)
            test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
                torch.from_numpy(windows).float().view(([windows.shape[0], w_size]))
            ), batch_size=self.ad_model_batch_size, shuffle=False, num_workers=0)
            self.ad_model.eval()
            with torch.no_grad():
                results = usad.testing(self.ad_model, test_loader, alpha=self.ad_model_alpha, beta=self.ad_model_beta, device=self.device)
            y_pred = np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                                     results[-1].flatten().detach().cpu().numpy()])
            y_pred_label_ab = [int(i > self.mse) for i in y_pred]
            self.data = windows[np.array(y_pred_label_ab) == 1]
            self.data_ab = windows[np.array(y_true_ab) == 1]
            y_true = y_true_ab
            y_pred = y_pred_label_ab
            print(classification_report(y_true=y_true, y_pred=y_pred, digits=5))
            print(confusion_matrix(y_true=y_true, y_pred=y_pred))
            print('Anomaly Detection AUC-ROC: {:.5f}'.format(roc_auc_score(y_true, y_pred)))
            print('Anomaly Detection AUC-PR: {:.5f}'.format(average_precision_score(y_true, y_pred)))


    def _get_gvar(self):
        if self.training_gvar == 1:
            print('Training GVAR.')
            train_data = self.train_x_norm
            if len(train_data.shape) > 2:
                self.A_hat, self.A_value_hat, coeffs_full_l, self.senn_1, self.optim, self.senn_2, self.optim_2 = experimental_utils.training_procedure_trgc(
                    data=train_data[0], order=self.K,
                    hidden_layer_size=self.hidden_layer_size,
                    end_epoch=self.num_epochs, lmbd=self.lmbd,
                    gamma=self.gamma, batch_size=self.batch_size,
                    seed=self.seed,
                    num_hidden_layers=self.num_hidden_layers,
                    initial_learning_rate=self.initial_lr,
                    beta_1=self.beta_1, beta_2=self.beta_2,
                    verbose=False, use_cuda=self.device)
                for i in tqdm(range(1,len(train_data)), desc='GVAR training epoch:'):
                    self.A_hat, self.A_value_hat, coeffs_full_l, senn_1, optim_1, senn_2, optim_2 = experimental_utils.training_procedure_trgc(
                        data=train_data[i], order=self.K,
                        hidden_layer_size=self.hidden_layer_size,
                        end_epoch=self.num_epochs, lmbd=self.lmbd,
                        gamma=self.gamma, batch_size=self.batch_size,
                        seed=self.seed,
                        num_hidden_layers=self.num_hidden_layers,
                        initial_learning_rate=self.initial_lr,
                        beta_1=self.beta_1, beta_2=self.beta_2,
                        verbose=False, pretrained_senn=self.senn_1, pretrained_optim=self.optim,
                        pretrained_senn_2=self.senn_2, pretrained_optim_2=self.optim_2, use_cuda=self.device)
                    self.senn_1  = senn_1
                    self.optim = optim_1
                    self.senn_2 = senn_2
                    self.optim_2 = optim_2
            else:
                self.A_hat, self.A_value_hat, coeffs_full_l, self.senn_1, self.optim, self.senn_2, self.optim_2 = \
                    experimental_utils.training_procedure_trgc(data=train_data, order=self.K,
                                                              hidden_layer_size=self.hidden_layer_size,
                                                              end_epoch=self.num_epochs, lmbd=self.lmbd,
                                                              gamma=self.gamma, batch_size=self.batch_size,
                                                              seed=self.seed,
                                                              num_hidden_layers=self.num_hidden_layers,
                                                              initial_learning_rate=self.initial_lr,
                                                              beta_1=self.beta_1, beta_2=self.beta_2,
                                                              verbose=False, use_cuda=self.device)
            torch.save(self.senn_1.state_dict(), f'./saved_models/SENN_{self.save_parm}.pt')
            pd.DataFrame(self.A_value_hat).to_csv(f'./saved_models/A_{self.save_parm}.csv')
        else:
            print('Loading GVAR.')
            self.senn_1 = senn.SENNGC(num_vars=self.p, order=self.K, hidden_layer_size=self.hidden_layer_size,
                          num_hidden_layers=self.num_hidden_layers, device=self.device)
            self.senn_1.load_state_dict(torch.load(f'./saved_models/SENN_{self.save_parm}.pt', map_location=self.device))
            self.senn_1.to(self.device)
            self.A_value_hat = torch.Tensor(pd.read_csv(f'./saved_models/A_{self.save_parm}.csv', index_col=0).values)

            total_l2_gvar = []
            total_l2_our = []
            train_data = self.train_x_norm
            if len(train_data.shape) > 2:
                sample_l2_gvar = []
                sample_l2_our = []
                for i in range(len(train_data)):
                    sample_org = train_data[i].copy()
                    for j in range(self.ad_model_K, len(sample_org)):
                        window_org = torch.tensor(sample_org[np.newaxis, j-self.ad_model_K:j, :]).float().to(self.device)
                        # get delta
                        y_cf_gvar = self.senn_1(window_org[:, -self.K - 1:-1, :])[0]
                        delta_pred = window_org[:, -1, :] - y_cf_gvar.data
                        # generate cf world
                        y_cf_our = self.senn_1(window_org[:, -self.K - 1:-1, :])[0] + delta_pred
                        sample_l2_gvar.append(self.loss_mse_reduce(y_cf_gvar, window_org[:, -1, :]).item())
                        sample_l2_our.append(self.loss_mse_reduce(y_cf_our, window_org[:, -1, :]).item())
                total_l2_gvar.append(np.mean(sample_l2_gvar))
                total_l2_our.append(np.mean(sample_l2_our))
            else:
                for j in range(self.ad_model_K, len(train_data)):
                    window_org = torch.tensor(train_data[np.newaxis, j - self.ad_model_K:j, :]).float().to(self.device)

                    # get delta
                    y_cf_gvar = self.senn_1(window_org[:, -self.K - 1:-1, :])[0]
                    delta_pred = window_org[:, -1, :] - y_cf_gvar.data
                    # generate cf world
                    y_cf_our = self.senn_1(window_org[:, -self.K - 1:-1, :])[0] + delta_pred
                    total_l2_gvar.append(self.loss_mse_reduce(y_cf_gvar, window_org[:, -1, :]).item())
                    total_l2_gvar.append(self.loss_mse_reduce(y_cf_our, window_org[:, -1, :]).item())
            l2_gvar = np.mean(total_l2_gvar)
            l2_our = np.mean(total_l2_our)
            print(f'MSE for GVAR {l2_gvar}, MSE for our {l2_our}')

    def get_changed_data(self, window, delta_pred, eps_norm=None):
        if self.dataset_name == 'linear':
            if self.adtype == 'non_causal':
                ys_theta_next_gt = np.zeros((len(window), self.p))
                ys_theta_next_gt[:, 0] = self.a[0] * window[:, -1, 0]
                ys_theta_next_gt[:, 1] = self.a[1] * window[:, -1, 1] + self.a[2] * window[:, -1, 0]
                ys_theta_next_gt[:, 2] = self.a[3] * window[:, -1, 2] + self.a[4] * window[:, -1, 1]
                ys_theta_next_gt[:, 3] = self.a[5] * window[:, -1, 3] + self.a[6] * window[:, -1, 1] + self.a[7] * window[:, -1, 2]
                if eps_norm is None:
                    ys_theta_next_gt = ys_theta_next_gt + delta_pred.numpy()
                else:
                    ys_theta_next_gt = ys_theta_next_gt + eps_norm
            else:
                b = self.a.copy() * 3
                ys_theta_next_gt = np.zeros((len(window), self.p))
                ys_theta_next_gt[:, 3] = b[0] * window[:, -1, 3]
                ys_theta_next_gt[:, 2] = b[1] * window[:, -1, 2] + b[2] * window[:, -1, 0]
                ys_theta_next_gt[:, 1] = b[3] * window[:, -1, 1] + b[4] * window[:, -1, 1]
                ys_theta_next_gt[:, 0] = b[5] * window[:, -1, 0] + b[6] * window[:, -1, 1] + b[
                    7] * window[:, -1, 2]
                if eps_norm is None:
                    ys_theta_next_gt = ys_theta_next_gt + delta_pred.numpy()
                else:
                    ys_theta_next_gt = ys_theta_next_gt + eps_norm
        elif self.dataset_name == 'lotka-volterra':
            window = torch.tensor(window)
            if eps_norm is None:
                delta_pred = delta_pred[np.newaxis, :]
                ys_theta_next_gt = self.mlv.next_value(window[:, -1], delta_pred, dt=self.dt, downsample_factor=self.downsample_factor)
            else:
                eps_norm = torch.tensor(eps_norm[np.newaxis, :])
                ys_theta_next_gt = self.mlv.next_value(window[:, -1], eps_norm, dt=self.dt, downsample_factor=self.downsample_factor)
        else:
            NotImplementedError

        return ys_theta_next_gt


    def _get_recourse_model(self, cost_f=False):
        self.recourse_model_input_dim = self.p
        self.recourse_model_out_dim = self.p
        self.recourse_model_mse = nn.MSELoss()
        if self.dataset_name == 'MSDS':
            lst_test_x = []
            lst_test_x_norm = []
            lst_test_label = []
            lst_all_label = []
            count = 0
            temp_x = []
            temp_x_norm = []
            temp_label = []
            temp_all_label = []
            pas = 0
            for i in range(len(self.test_x) - 1):
                if pas == 0:
                    if count <= self.recourse_model_early_stop:
                        if count > 0:
                            count += 1
                        if (self.test_label[i] == 1) and (count == 0):
                            count = 1
                        temp_x.append(self.test_x[i])
                        temp_x_norm.append(self.test_x_norm[i])
                        temp_label.append(self.test_label[i])
                        temp_all_label.append(self.all_label[i])
                    else:
                        lst_test_x.append(np.array(temp_x))
                        lst_test_x_norm.append(np.array(temp_x_norm))
                        lst_test_label.append(np.array(temp_label))
                        lst_all_label.append(np.array(temp_all_label))
                        count = 0
                        temp_x = []
                        temp_x_norm = []
                        temp_label = []
                        temp_all_label = []
                        if self.test_label[i + 1] == 0:
                            pas = 0
                        else:
                            pas = 1
                else:
                    if self.test_label[i + 1] == 0:
                        pas = 0

            temp = list(zip(lst_test_x, lst_test_x_norm, lst_test_label, lst_all_label))
            random.shuffle(temp)
            res1, res2, res3, res4 = zip(*temp)
            lst_test_x, lst_test_x_norm, lst_test_label, lst_all_label = list(res1), list(res2), list(res3), list(res4)
            self.test_x = lst_test_x
            self.test_x_norm = lst_test_x_norm
            self.test_label = lst_test_label
            self.all_label = lst_all_label
            ind = int(len(self.test_x_norm)*0.8)
            self.rec_train_x = self.test_x_norm[:int(ind * 0.8)]
            self.rec_val_x = self.test_x_norm[int(ind * 0.8):ind]
            self.rec_test_x = np.array(self.test_x_norm[ind:])
            self.rec_train_label = self.test_label[:int(ind * 0.8)]
            self.rec_val_label = self.test_label[int(ind * 0.8):ind]
            self.rec_test_label = self.test_label[ind:]
            self.rec_all_label = self.all_label[ind:]
            # ind = int(len(self.test_x_norm)*0.9)
            # self.rec_train_x = self.test_x_norm[np.newaxis, :int(ind * 0.8), :]
            # self.rec_val_x = self.test_x_norm[np.newaxis, int(ind * 0.8):ind, :]
            # self.rec_test_x = self.test_x_norm[np.newaxis, ind:, :]
            # self.rec_train_label = self.test_label[np.newaxis, :int(ind * 0.8)]
            # self.rec_val_label = self.test_label[np.newaxis, int(ind * 0.8):ind]
            # self.rec_test_label = self.test_label[np.newaxis, ind:]
        else:
            ind = len(self.test_x_norm) // 2
            self.rec_train_x = self.test_x_norm[:int(ind*0.8)]
            self.rec_train_label = self.test_label[:int(ind * 0.8)]

            self.rec_val_x = self.test_x_norm[int(ind*0.8):ind]
            self.rec_val_eps_norm = self.test_eps_norm[int(ind*0.8):ind]
            self.rec_val_label = self.test_label[int(ind * 0.8):ind]

            self.rec_test_x = self.test_x_norm[ind:]
            self.rec_test_eps_norm = self.test_eps_norm[ind:]
            self.rec_test_label = self.test_label[ind:]

        if cost_f:
            self.cost_f = True
        else:
            self.cost_f = False
        self.ad_model.eval()
        self.senn_1.eval()
        # self.recourse_model = recourse.INet(num_vars=self.p, hidden_layer_size=self.hidden_layer_size,
        #               num_hidden_layers=self.num_hidden_layers, device=self.device).to(self.device)
        # self.recourse_model = nn.Sequential(
        #     nn.Dropout(p=0.1),
        #     nn.Linear(self.ad_model_K*self.p, self.recourse_model_hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.recourse_model_hidden_dim, self.recourse_model_hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.recourse_model_hidden_dim, self.recourse_model_hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.recourse_model_hidden_dim, self.p)
        # ).to(self.device)
        # self.recourse_model = adcarts.ADCARTS(input_size=self.p, hidden_size=self.recourse_model_hidden_dim, num_layers=1).to(self.device)
        self.recourse_model = lstm.MTLSTM(input_size=self.p, hidden_size=self.recourse_model_hidden_dim).to(self.device)
        self.recourse_model_optim = optim.Adam(self.recourse_model.parameters(), lr=self.recourse_model_lr)
        # self.recourse_model_optim = optim.SGD(self.recourse_model.parameters(), lr=self.recourse_model_lr)

        if self.recourse_model_training:
            self.recourse_model.train()
            best_eval_loss = float('inf')
            for _ in tqdm(range(self.recourse_model_max_epoch), desc="RecAD training epoch:"):
                # training step
                epoch_loss = np.float64(0.0)
                epoch_rec_loss = 0.0
                epoch_l2_loss = 0.0
                epoch_logit_loss = 0.0
                count = 0.00001
                for i in range(len(self.rec_train_x)):
                    sample_org = self.rec_train_x[i].copy()
                    sample_cf = sample_org.copy()
                    early_stop = 0
                    for j in range(self.ad_model_K, len(sample_cf)):
                        if early_stop >= self.recourse_model_early_stop:
                            break
                        window_org = torch.tensor(sample_org[np.newaxis, j-self.ad_model_K:j, :]).float().to(self.device)
                        window_cf = torch.tensor(sample_cf[np.newaxis, j-self.ad_model_K:j, :]).float().to(self.device)

                        # get delta
                        y_org_pred = self.senn_1(window_org[:, -self.K - 1:-1, :])[0]
                        delta_pred = window_org[:, -1, :] - y_org_pred.data
                        # generate cf world
                        y_cf_pred = self.senn_1(window_cf[:, -self.K - 1:-1, :])[0] + delta_pred
                        window_cf[:, -1, :] = y_cf_pred.data
                        sample_cf[j-1] = y_cf_pred.detach().cpu().data

                        window_cf_flattened = window_cf.reshape(1, -1)
                        w_recon = usad.testing_sample(self.ad_model, window_cf_flattened, alpha=self.ad_model_alpha, beta=self.ad_model_beta)
                        if w_recon > self.mse:
                            count += 1
                            if early_stop == 0:
                                early_stop = 1
                            # theta = self.recourse_model(delta_pred)
                            # theta = self.recourse_model(window_cf_flattened)
                            theta = self.recourse_model(window_cf[:, :-1, :])

                            # get recourse
                            window_rec = window_cf + F.pad(theta, (0, 0, self.ad_model_K-1, 0))
                            sample_cf[j-1] = window_rec.data.cpu()[0, -1]
                            window_rec_flattened = window_rec.reshape(1, -1)
                            rec_loss = usad.testing_sample(self.ad_model, window_rec_flattened, alpha=self.ad_model_alpha,
                                                          beta=self.ad_model_beta)
                            # get next time step loss
                            window_org_next = torch.tensor(
                                sample_org[np.newaxis, j - self.ad_model_K + 1:j + 1, :]).float().to(self.device)
                            window_cf_next = torch.tensor(
                                sample_cf[np.newaxis, j - self.ad_model_K + 1:j + 1, :]).float().to(self.device)
                            y_org_next_pred = self.senn_1(window_org_next[:, -self.K - 1:-1, :])[0]
                            delta_pred_next = window_org_next[:, -1, :] - y_org_next_pred.data
                            y_cf_next_pred = self.senn_1(window_cf_next[:, -self.K - 1:-1, :])[0] + delta_pred_next
                            window_cf_next[:, -1, :] = y_cf_next_pred.data
                            window_rec_next_flattened = window_cf_next.reshape(1, -1)
                            rec_loss_next = usad.testing_sample(self.ad_model, window_rec_next_flattened, alpha=self.ad_model_alpha,
                                                          beta=self.ad_model_beta)

                            # other loss
                            logit_loss = 1 if rec_loss > self.mse else 0
                            if self.cost_f:
                                l2_loss = self.recourse_model_mse(theta * self.scale, torch.zeros(theta.size()).to(self.device))
                            else:
                                l2_loss = self.recourse_model_mse(theta, torch.zeros(theta.size()).to(self.device))
                            loss = self.recourse_model_alpha * rec_loss  + self.recourse_model_beta * l2_loss + self.recourse_model_gamma * rec_loss_next

                            self.recourse_model_optim.zero_grad()
                            loss.backward()
                            self.recourse_model_optim.step()

                            epoch_loss += loss.item()
                            epoch_rec_loss += rec_loss.item()
                            epoch_l2_loss += l2_loss.item()
                            epoch_logit_loss += logit_loss
                        if early_stop > 0:
                            early_stop += 1

                # eval step
                eval_loss = 0.0
                eval_logit_loss = 0.0
                eval_count = 0.00001
                with torch.no_grad():
                    for i in range(len(self.rec_val_x)):
                        sample_org = self.rec_val_x[i].copy()
                        sample_cf = sample_org.copy()
                        sample_gt = sample_org.copy()
                        if self.dataset_name != 'MSDS':
                            sample_eps = self.rec_val_eps_norm[i]
                        else:
                            sample_eps = []
                        early_stop = 0
                        for j in range(self.ad_model_K, len(sample_cf)):
                            if early_stop >= self.recourse_model_early_stop:
                                break
                            if self.dataset_name != 'MSDS':
                                # generate cf gt world
                                window_gt = sample_gt[np.newaxis, j - self.ad_model_K:j, :]
                                eps_norm = sample_eps[j - 1, :]
                                y_gt_next_pred = self.get_changed_data(window_gt[:, -self.K - 1:-1, :], None,
                                                                       eps_norm)
                                sample_gt[j - 1] = y_gt_next_pred
                                sample_cf[j - 1] = y_gt_next_pred

                                window_gt = torch.tensor(sample_gt[np.newaxis, j - self.ad_model_K:j, :]).float().to(
                                    self.device)
                                window_cf = torch.tensor(sample_gt[np.newaxis, j - self.ad_model_K:j, :]).float().to(
                                    self.device)
                                y_gt_next_pred = self.senn_1(window_gt[:, -self.K - 1:-1, :])[0]
                                delta_pred = window_gt[:, -1, :] - y_gt_next_pred.data

                                window_gt_flattened = window_gt.reshape(1, -1)
                                w_recon = usad.testing_sample(self.ad_model, window_gt_flattened,
                                                              alpha=self.ad_model_alpha,
                                                              beta=self.ad_model_beta)
                            else:
                                window_org = torch.tensor(sample_org[np.newaxis, j - self.ad_model_K:j, :]).float().to(
                                    self.device)
                                window_cf = torch.tensor(sample_cf[np.newaxis, j - self.ad_model_K:j, :]).float().to(
                                    self.device)
                                # get delta
                                y_org_next_pred = self.senn_1(window_org[:, -self.K - 1:-1, :])[0]
                                delta_pred = window_org[:, -1, :] - y_org_next_pred.data
                                # generate cf world
                                y_cf_next_pred = self.senn_1(window_cf[:, -self.K - 1:-1, :])[0]
                                y_cf_next_pred += delta_pred
                                window_cf[:, -1, :] = y_cf_next_pred.data
                                sample_cf[j - 1] = y_cf_next_pred.detach().cpu().data

                                window_cf_flattened = window_cf.reshape(1, -1)
                                w_recon = usad.testing_sample(self.ad_model, window_cf_flattened,
                                                              alpha=self.ad_model_alpha,
                                                              beta=self.ad_model_beta)

                            if w_recon > self.mse:
                                if early_stop == 0:
                                    early_stop = 1
                                eval_count += 1
                                # theta = self.recourse_model(delta_pred)
                                # theta = self.recourse_model(window_cf_flattened)
                                theta = self.recourse_model(window_cf[:, :-1, :])
                                if self.dataset_name != 'MSDS':
                                    sample_gt[j - 1] = sample_gt[j - 1] + theta.detach().cpu().numpy()[0]
                                    window_rec = window_gt + F.pad(theta, (0, 0, self.ad_model_K - 1, 0))
                                else:
                                    window_rec = window_cf + F.pad(theta, (0, 0, self.ad_model_K - 1, 0))

                                sample_cf[j - 1] = window_rec.data.cpu()[0, -1]
                                window_rec_flattened = window_rec.reshape(1, -1)
                                rec_loss = usad.testing_sample(self.ad_model, window_rec_flattened,
                                                               alpha=self.ad_model_alpha,
                                                               beta=self.ad_model_beta)
                                logit_loss = 1 if rec_loss > self.mse else 0
                                if self.cost_f:
                                    l2_loss = self.recourse_model_mse(theta * self.scale, torch.zeros(theta.size()).to(self.device))
                                else:
                                    l2_loss = self.recourse_model_mse(theta, torch.zeros(theta.size()).to(self.device))
                                loss = self.recourse_model_alpha * rec_loss  + self.recourse_model_beta * l2_loss

                                eval_loss += loss.item()
                                eval_logit_loss += logit_loss
                            if early_stop > 0:
                                early_stop += 1

                if eval_logit_loss/eval_count < best_eval_loss:
                    best_eval_loss = eval_logit_loss/eval_count
                    torch.save(self.recourse_model.state_dict(), f'./saved_models/RC_{self.save_parm}.pt')

                print(f'Train reconstruction loss: {epoch_rec_loss/count}, l2 loss: {epoch_l2_loss/count}, logit loss: {epoch_logit_loss}, recourse wind_count: {count-1}')
                print(f'Val reconstruction loss: {eval_loss/eval_count}, logit loss: {eval_logit_loss}, recourse wind_count: {eval_count-1}')
            print('Finished training recourse model.')
        else:
            self.recourse_model.load_state_dict(
                torch.load(f'./saved_models/RC_{self.save_parm}.pt', map_location=self.device))
            self.recourse_model.eval()
            print('Loading recourse model.')
            # eval step
            eval_loss = 0.0
            eval_logit_loss = 0.0
            eval_count = 0.00001
            with torch.no_grad():
                for i in range(len(self.rec_val_x)):
                    sample_org = self.rec_val_x[i].copy()
                    sample_cf = sample_org.copy()
                    sample_gt = sample_org.copy()
                    if self.dataset_name != 'MSDS':
                        sample_eps = self.rec_val_eps_norm[i]
                    else:
                        sample_eps = []
                    early_stop = 0
                    for j in range(self.ad_model_K, len(sample_cf)):
                        if early_stop >= self.recourse_model_early_stop:
                            break
                        if self.dataset_name != 'MSDS':
                            # generate cf gt world
                            window_gt = sample_gt[np.newaxis, j - self.ad_model_K:j, :]
                            eps_norm = sample_eps[j - 1, :]
                            y_gt_next_pred = self.get_changed_data(window_gt[:, -self.K - 1:-1, :], None,
                                                                   eps_norm)
                            sample_gt[j - 1] = y_gt_next_pred
                            sample_cf[j - 1] = y_gt_next_pred

                            window_gt = torch.tensor(sample_gt[np.newaxis, j - self.ad_model_K:j, :]).float().to(
                                self.device)
                            window_cf = torch.tensor(sample_gt[np.newaxis, j - self.ad_model_K:j, :]).float().to(
                                self.device)
                            y_gt_next_pred = self.senn_1(window_gt[:, -self.K - 1:-1, :])[0]
                            delta_pred = window_gt[:, -1, :] - y_gt_next_pred.data

                            window_gt_flattened = window_gt.reshape(1, -1)
                            w_recon = usad.testing_sample(self.ad_model, window_gt_flattened, alpha=self.ad_model_alpha,
                                                          beta=self.ad_model_beta)
                        else:
                            window_org = torch.tensor(sample_org[np.newaxis, j - self.ad_model_K:j, :]).float().to(
                                self.device)
                            window_cf = torch.tensor(sample_cf[np.newaxis, j - self.ad_model_K:j, :]).float().to(
                                self.device)
                            # get delta
                            y_org_next_pred = self.senn_1(window_org[:, -self.K - 1:-1, :])[0]
                            delta_pred = window_org[:, -1, :] - y_org_next_pred.data
                            # generate cf world
                            y_cf_next_pred = self.senn_1(window_cf[:, -self.K - 1:-1, :])[0]
                            y_cf_next_pred += delta_pred
                            window_cf[:, -1, :] = y_cf_next_pred.data
                            sample_cf[j - 1] = y_cf_next_pred.detach().cpu().data

                            window_cf_flattened = window_cf.reshape(1, -1)
                            w_recon = usad.testing_sample(self.ad_model, window_cf_flattened, alpha=self.ad_model_alpha,
                                                          beta=self.ad_model_beta)

                        if w_recon > self.mse:
                            if early_stop == 0:
                                early_stop = 1
                            eval_count += 1
                            # theta = self.recourse_model(delta_pred)
                            # theta = self.recourse_model(window_cf_flattened)
                            theta = self.recourse_model(window_cf[:, :-1, :])
                            if self.dataset_name != 'MSDS':
                                sample_gt[j - 1] = sample_gt[j - 1] + theta.detach().cpu().numpy()[0]
                                window_rec = window_gt + F.pad(theta, (0, 0, self.ad_model_K - 1, 0))
                            else:
                                window_rec = window_cf + F.pad(theta, (0, 0, self.ad_model_K - 1, 0))

                            sample_cf[j - 1] = window_rec.data.cpu()[0, -1]
                            window_rec_flattened = window_rec.reshape(1, -1)
                            rec_loss = usad.testing_sample(self.ad_model, window_rec_flattened,
                                                           alpha=self.ad_model_alpha,
                                                           beta=self.ad_model_beta)
                            logit_loss = 1 if rec_loss > self.mse else 0
                            if self.cost_f:
                                l2_loss = self.recourse_model_mse(theta * self.scale,
                                                                  torch.zeros(theta.size()).to(self.device))
                            else:
                                l2_loss = self.recourse_model_mse(theta, torch.zeros(theta.size()).to(self.device))
                            loss = self.recourse_model_alpha * rec_loss + self.recourse_model_beta * l2_loss

                            eval_loss += loss.item()
                            eval_logit_loss += logit_loss
                        if early_stop > 0:
                            early_stop += 1
            print(
                f'Val reconstruction loss: {eval_loss / eval_count}, logit loss: {eval_logit_loss}, recourse wind_count: {eval_count - 1}')

    def _get_recourse_model_results(self):
        self.ad_model.eval()
        self.senn_1.eval()
        self.recourse_model.eval()
        total_loss = 0.0
        total_rec_loss = 0.0
        total_l2_loss = 0.0
        total_logit_loss = 0.0
        count = 0.00001
        final_sample_gt = []
        final_sample_cf = []
        final_sample_org = []
        final_sample_label = []
        final_sample_rec = []
        final_sample_all_label = []


        # RecAD
        ad_sample_count = 0
        for i in tqdm(range(len(self.rec_test_x)), desc='Testing sample:'):
        # for i in tqdm(range(1, 5), desc='Testing sample:'):
            sample_org = self.rec_test_x[i].copy()
            sample_cf = sample_org.copy()
            sample_gt = sample_org.copy()
            sample_rec = np.zeros(len(sample_org))
            if self.dataset_name != 'MSDS':
                sample_eps = self.rec_test_eps_norm[i]
            else:
                sample_eps = []
                sample_all_label = self.rec_all_label[i]
            sample_label = self.rec_test_label[i]
            early_stop = 0
            ad_sample = 10e-5
            with torch.no_grad():
                for j in range(self.ad_model_K, len(sample_cf)):
                    if early_stop >= self.recourse_model_early_stop:
                        break
                    if self.dataset_name != 'MSDS':
                        # generate cf gt world
                        window_gt = sample_gt[np.newaxis, j - self.ad_model_K:j, :]
                        eps_norm = sample_eps[j-1, :]
                        y_gt_next_pred = self.get_changed_data(window_gt[:, -self.K - 1:-1, :], None, eps_norm)
                        sample_gt[j - 1] = y_gt_next_pred
                        sample_cf[j - 1] = y_gt_next_pred

                        window_gt = torch.tensor(sample_gt[np.newaxis, j - self.ad_model_K:j, :]).float().to(
                            self.device)
                        window_cf = torch.tensor(sample_gt[np.newaxis, j - self.ad_model_K:j, :]).float().to(
                            self.device)
                        y_gt_next_pred = self.senn_1(window_gt[:, -self.K - 1:-1, :])[0]
                        delta_pred = window_gt[:, -1, :] - y_gt_next_pred.data

                        window_gt_flattened = window_gt.reshape(1, -1)
                        w_recon = usad.testing_sample(self.ad_model, window_gt_flattened, alpha=self.ad_model_alpha,
                                                      beta=self.ad_model_beta)
                    else:
                        window_org = torch.tensor(sample_org[np.newaxis, j - self.ad_model_K:j, :]).float().to(
                            self.device)
                        window_cf = torch.tensor(sample_cf[np.newaxis, j - self.ad_model_K:j, :]).float().to(
                            self.device)
                        # get delta
                        y_org_next_pred = self.senn_1(window_org[:, -self.K - 1:-1, :])[0]
                        delta_pred = window_org[:, -1, :] - y_org_next_pred.data
                        # generate cf world
                        y_cf_next_pred = self.senn_1(window_cf[:, -self.K - 1:-1, :])[0]
                        y_cf_next_pred += delta_pred
                        window_cf[:, -1, :] = y_cf_next_pred.data
                        sample_cf[j - 1] = y_cf_next_pred.detach().cpu().data

                        window_cf_flattened = window_cf.reshape(1, -1)
                        w_recon = usad.testing_sample(self.ad_model, window_cf_flattened, alpha=self.ad_model_alpha,
                                                      beta=self.ad_model_beta)

                    if w_recon > self.mse:
                        ad_sample = 1
                        if early_stop == 0:
                            early_stop = 1
                        count += 1
                        sample_rec[j-1] = 1
                        # theta = self.recourse_model(delta_pred)
                        # theta = self.recourse_model(window_cf_flattened)
                        theta = self.recourse_model(window_cf[:, :-1, :])
                        if self.dataset_name != 'MSDS':
                            sample_gt[j - 1] = sample_gt[j - 1] + theta.detach().cpu().numpy()[0]
                            window_rec = window_gt + F.pad(theta, (0, 0, self.ad_model_K - 1, 0))
                        else:
                            window_rec = window_cf + F.pad(theta, (0, 0, self.ad_model_K - 1, 0))

                        sample_cf[j - 1] = window_rec.data.cpu()[0, -1]
                        window_rec_flattened = window_rec.reshape(1, -1)
                        rec_loss = usad.testing_sample(self.ad_model, window_rec_flattened, alpha=self.ad_model_alpha,
                                                       beta=self.ad_model_beta)
                        logit_loss = 1 if rec_loss > self.mse else 0
                        if self.cost_f:
                            l2_loss = self.recourse_model_mse(theta * self.scale,
                                                              torch.zeros(theta.size()).to(self.device))
                        else:
                            l2_loss = self.recourse_model_mse(theta, torch.zeros(theta.size()).to(self.device))
                        loss = self.recourse_model_alpha * rec_loss + self.recourse_model_beta * l2_loss

                        total_loss += loss.item()
                        total_rec_loss += rec_loss.item()
                        total_l2_loss += l2_loss.item()
                        total_logit_loss += logit_loss
                    if early_stop > 0:
                        early_stop += 1
            ad_sample_count += ad_sample
            final_sample_cf.append(sample_cf)
            final_sample_gt.append(sample_gt)
            final_sample_org.append(sample_org)
            final_sample_label.append(sample_label)
            final_sample_rec.append(sample_rec)
            if self.dataset_name == 'MSDS':
                final_sample_all_label.append(sample_all_label)
        print(f'reconstruction loss: {total_rec_loss / count}, l2 loss: {total_l2_loss / count}, '
              f'logit loss: {total_logit_loss}, recourse wind_count: {count-1}, '
              f'flipping ratio: {1-total_logit_loss/count}, l2 loss per seq: {total_l2_loss / ad_sample_count}, '
              f'recourse wind_count per seq: {(count-1) / ad_sample_count}')
        print()
        print('=' * 20)
        if self.dataset_name == 'MSDS':
            return final_sample_org, final_sample_cf, final_sample_gt, final_sample_label, final_sample_rec, final_sample_all_label
        else:
            return final_sample_org, final_sample_cf, final_sample_gt, final_sample_label, final_sample_rec

