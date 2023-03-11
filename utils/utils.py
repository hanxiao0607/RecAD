# Some basic helper functions
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, balanced_accuracy_score, \
    precision_score, recall_score

from scipy.stats import entropy


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def construct_training_dataset(data, order):
    # Pack the data, if it is not in a list already
    if not isinstance(data, list):
        data = [data]

    data_out = None
    response = None
    time_idx = None
    # Iterate through time series replicates
    offset = 0
    for r in range(len(data)):
        data_r = data[r]
        # data: T x p
        T_r = data_r.shape[0]
        p_r = data_r.shape[1]
        inds_r = np.arange(order, T_r)
        data_out_r = np.zeros((T_r - order, order, p_r))
        response_r = np.zeros((T_r - order, p_r))
        time_idx_r = np.zeros((T_r - order, ))
        for i in range(T_r - order):
            j = inds_r[i]
            data_out_r[i, :, :] = data_r[(j - order):j, :]
            response_r[i] = data_r[j, :]
            time_idx_r[i] = j
        # TODO: just a hack, need a better solution...
        time_idx_r = time_idx_r + offset + 200 * (r >= 1)
        time_idx_r = time_idx_r.astype(int)
        if data_out is None:
            data_out = data_out_r
            response = response_r
            time_idx = time_idx_r
        else:
            data_out = np.concatenate((data_out, data_out_r), axis=0)
            response = np.concatenate((response, response_r), axis=0)
            time_idx = np.concatenate((time_idx, time_idx_r))
        offset = np.max(time_idx_r)
    return data_out, response, time_idx


def eval_causal_structure(a_true: np.ndarray, a_pred: np.ndarray, diagonal=False):
    if not diagonal:
        a_true_offdiag = a_true[np.logical_not(np.eye(a_true.shape[0]))]
        a_pred_offdiag = a_pred[np.logical_not(np.eye(a_true.shape[0]))]
        if np.max(a_true_offdiag) == np.min(a_true_offdiag):
            auroc = None
            auprc = None
        else:
            auroc = roc_auc_score(y_true=a_true_offdiag.flatten(), y_score=a_pred_offdiag.flatten())
            auprc = average_precision_score(y_true=a_true_offdiag.flatten(), y_score=a_pred_offdiag.flatten())
    else:
        auroc = roc_auc_score(y_true=a_true.flatten(), y_score=a_pred.flatten())
        auprc = average_precision_score(y_true=a_true.flatten(), y_score=a_pred.flatten())
    return auroc, auprc


def eval_causal_structure_binary(a_true: np.ndarray, a_pred: np.ndarray, diagonal=False):
    if not diagonal:
        a_true_offdiag = a_true[np.logical_not(np.eye(a_true.shape[0]))].flatten()
        a_pred_offdiag = a_pred[np.logical_not(np.eye(a_true.shape[0]))].flatten()
        precision = precision_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
        recall = recall_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
        accuracy = accuracy_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
        bal_accuracy = balanced_accuracy_score(y_true=a_true_offdiag, y_pred=a_pred_offdiag)
    else:
        precision = precision_score(y_true=a_true.flatten(), y_pred=a_pred.flatten())
        recall = recall_score(y_true=a_true.flatten(), y_pred=a_pred.flatten())
        accuracy = accuracy_score(y_true=a_true.flatten(), y_pred=a_pred.flatten())
        bal_accuracy = balanced_accuracy_score(y_true=a_true.flatten(), y_pred=a_pred.flatten())
    return accuracy, bal_accuracy, precision, recall


def eval_concordance(a_1: np.ndarray, a_2: np.ndarray):
    a_1 = a_1.flatten()
    a_2 = a_2.flatten()
    n_pairs = len(a_1) * (len(a_1) - 1)
    n_concordant_pairs = 0
    n_discordant_pairs = 0

    for i in range(len(a_1)):
        for j in range(len(a_1)):
            if i != j:
                if a_1[i] < a_1[j] and a_2[i] < a_2[j]:
                    n_concordant_pairs += 1
                elif a_1[i] > a_1[j] and a_2[i] > a_2[j]:
                    n_concordant_pairs += 1
                elif a_1[i] < a_1[j] and a_2[i] > a_2[j]:
                    n_discordant_pairs += 1
                elif a_1[i] > a_1[j] and a_2[i] < a_2[j]:
                    n_discordant_pairs += 1
    cindex = (n_concordant_pairs - n_discordant_pairs) / n_pairs

    return cindex


def kl_div_disc(x: np.ndarray, y: np.ndarray, n_bins=16):
    # NOTE: KL divergences are symmetrised!
    # Discretise and approximate using histograms
    h_y, bin_edges = np.histogram(a=y, bins=n_bins, density=False)
    # NOTE: Adding a small constant to avoid division by 0
    h_y = h_y + 1e-6
    h_y = h_y / np.sum(h_y)
    h_x, _ = np.histogram(a=x, bins=bin_edges, density=False)
    h_x = h_x + 1e-6
    h_x = h_x / np.sum(h_x)

    # Compute [D_{KL}(P || Q) + D_{KL}(Q || P)] / 2 for discrete distributions given by histograms
    return (entropy(pk=h_x, qk=h_y, axis=0) + entropy(pk=h_y, qk=h_x, axis=0)) / 2


def kl_div_normal(x: np.ndarray, y: np.ndarray):
    # NOTE: KL divergences are symmetrised!
    # Assumes normality
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    # Add a small positive constants to avoid division by 0
    sigma_x = np.std(x) + 1e-6
    sigma_y = np.std(x) + 1e-6

    kl_xy = np.log(sigma_y / sigma_x) + (sigma_x ** 2 + (mu_x - mu_y) ** 2) / (2 * sigma_y ** 2) - 0.5
    kl_yx = np.log(sigma_x / sigma_y) + (sigma_y ** 2 + (mu_y - mu_x) ** 2) / (2 * sigma_x ** 2) - 0.5

    return (kl_xy + kl_yx) / 2


def absolute_mean_deviation(x: np.ndarray, y: np.ndarray):
    return np.abs(np.mean(x) - np.mean(y))


def absolute_mean_relative_deviation(x: np.ndarray, y: np.ndarray):
    return np.abs(np.mean(x) - np.mean(y)) / (np.abs(np.mean(y)) + 1e-6)


def sliding_window(data, get_ys=1, y=None, size=5, step=1, downsampling=0.05):
    if get_ys == 1:
        windows = []
        ys = []
        if len(data.shape) > 2:
            for i in range(len(data)):
                for j in range(len(data[i])-size-step):
                    windows.append(data[i][j:j+size])
                    ys.append(data[i][j+size])
        else:
            for j in range(len(data) - size - step):
                windows.append(data[j:j + size])
                ys.append(data[j + size])
        windows = torch.tensor(windows).float()
        ys = torch.tensor(ys).float()
        return windows, ys
    else:
        windows = []
        lst_label = []
        if len(data.shape) > 2:
            if y is None:
                y = np.zeros((data.shape[0], data.shape[1]))
            for i in range(len(data)):
                for j in range(len(data[i]) - size):
                    if max(y[i][j:j + size]) == 1:
                        windows.append(data[i][j:j + size])
                        lst_label.append(1)
                    else:
                        windows.append(data[i][j:j + size])
                        lst_label.append(0)
        else:
            if y is None:
                y = np.zeros(len(data))
            for j in range(len(data) - size):
                if max(y[j:j + size]) == 1:
                    windows.append(data[j:j + size])
                    lst_label.append(1)
                else:
                    windows.append(data[j:j + size])
                    lst_label.append(0)

        ind = np.arange(len(windows))
        random.shuffle(ind)
        sel_ind = ind[:int(len(windows)*downsampling)]
        windows = np.array(windows)[sel_ind]
        lst_label = np.array(lst_label)[sel_ind]
        return windows, lst_label

def sliding_window_with_eps(data, label, eps, size=5):
    windows_n = []
    windows_ab = []
    ys_n = []
    ys_ab = []
    eps_n = []
    eps_ab = []
    if len(data.shape) > 2:
        for i in range(len(data)):
            for j in range(len(data[i]) - size):
                if label[i][j+size-1] == 1:
                    windows_ab.append(data[i][j:j + size])
                    ys_ab.append(data[i][j + size])
                    eps_ab.append(eps[i][j + size])
                elif max(label[i][j:j + size]) == 0:
                    windows_n.append(data[i][j:j + size])
                    ys_n.append(data[i][j + size])
                    eps_n.append(eps[i][j + size])
                else:
                    pass
    else:
        for i in range(len(data)):
            if label[i + size - 1] == 1:
                windows_ab.append(data[i:i + size])
                ys_ab.append(data[i + size])
                eps_ab.append(eps[i + size])
            elif max(label[i:i + size]) == 0:
                windows_n.append(data[i:i + size])
                ys_n.append(data[i + size])
                eps_n.append(eps[i + size])
            else:
                pass
    return windows_n, ys_n, eps_n, windows_ab, ys_ab, eps_ab


def get_labels(lst, mse, sample_size, size=5, step=1):
    lst = lst.reshape(sample_size, -1)
    lst = np.where(lst <= mse, 0, 1)
    label = []
    pos = []
    for i in range(len(lst)):
        temp_label = 0
        temp_pos = 0
        for j in range(len(lst[i])-4):
            if sum(lst[i][j:j+5]) >= 1:
                temp_label = 1
                temp_pos = j
                break
        label.append(temp_label)
        pos.append(temp_pos)
    return lst, label, pos

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def plot_history(history):
    losses1 = [x['val_loss1'] for x in history]
    losses2 = [x['val_loss2'] for x in history]
    plt.plot(losses1, '-x', label="loss1")
    plt.plot(losses2, '-x', label="loss2")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Losses vs. No. of epochs')
    plt.grid()
    plt.show()


def histogram(y_test, y_pred):
    plt.figure(figsize=(12, 6))
    plt.hist([y_pred[y_test == 0],
              y_pred[y_test == 1]],
             bins=20,
             color=['#82E0AA', '#EC7063'], stacked=True)
    plt.title("Results", size=20)
    plt.grid()
    plt.show()


def ROC(y_test, y_pred):
    fpr, tpr, tr = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    idx = np.argwhere(np.diff(np.sign(tpr - (1 - fpr)))).flatten()

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.plot(fpr, 1 - fpr, 'r:')
    plt.plot(fpr[idx], tpr[idx], 'ro')
    plt.legend(loc=4)
    plt.grid()
    plt.show()
    return tr[idx]


def confusion_matrix(target, predicted, perc=False):
    data = {'y_Actual': target,
            'y_Predicted': predicted
            }
    df = pd.DataFrame(data, columns=['y_Predicted', 'y_Actual'])
    confusion_matrix = pd.crosstab(df['y_Predicted'], df['y_Actual'], rownames=['Predicted'], colnames=['Actual'])

    if perc:
        sns.heatmap(confusion_matrix / np.sum(confusion_matrix), annot=True, fmt='.2%', cmap='Blues')
    else:
        sns.heatmap(confusion_matrix, annot=True, fmt='d')
    plt.show()

class SlidingWind(Dataset):
    def __init__(self, windows, ys):
        self.windows = windows
        self.ys = ys

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.windows[idx], self.ys[idx]

def normalize3(a, min_a = None, max_a = None):
	if min_a is None: min_a, max_a = np.min(a, axis = 0), np.max(a, axis = 0)
	return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a