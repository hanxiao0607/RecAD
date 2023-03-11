# Some synthetic datasets with linear dynamics
import numpy as np

def generate_linear_example_ab(n, t, mul=10, a=None, seed=0, adlength=1, adtype='non_causal'):
    """
    @param n: number of replicates.
    @param t: length of time series.
    @param seed: random generator seed.
    """
    if seed is not None:
        np.random.seed(seed)
    x_n_list = []
    x_ab_list = []
    eps_n_list = []
    eps_ab_list = []
    label_list = []
    if a is None:
        a = np.zeros((8,))
    for k in range(8):
        u_1 = np.random.uniform(0, 1, 1)
        if u_1 <= 0.5:
            a[k] = np.random.uniform(-0.8, -0.2, 1)
        else:
            a[k] = np.random.uniform(0.2, 0.8, 1)
    for i in range(n):
        eps_x = 0.4 * np.random.normal(0, 1, (t,))
        eps_y = 0.4 * np.random.normal(0, 1, (t,))
        eps_w = 0.4 * np.random.normal(0, 1, (t,))
        eps_z = 0.4 * np.random.normal(0, 1, (t,))
        x = np.zeros((t, 1))
        y = np.zeros((t, 1))
        w = np.zeros((t, 1))
        z = np.zeros((t, 1))

        for j in range(1, t):
            x[j, 0] = a[0] * x[j - 1, 0] + eps_x[j]
            w[j, 0] = a[1] * w[j - 1, 0] + a[2] * x[j - 1, 0] + eps_w[j]
            y[j, 0] = a[3] * y[j - 1, 0] + a[4] * w[j - 1, 0] + eps_y[j]
            z[j, 0] = a[5] * z[j - 1, 0] + a[6] * w[j - 1, 0] + a[7] * y[j - 1, 0] + eps_z[j]
        x_n_list.append(np.concatenate((x, w, y, z), axis=1))
        eps_n_list.append(np.concatenate((eps_x, eps_w, eps_y, eps_z), axis=0).reshape(4,-1).T)
        t_p = np.random.randint(int(0.2 * t), int(0.8 * t), size=1)
        if adlength > 1:
            temp_t_p = []
            for i in range(adlength):
                temp_t_p.append(t_p+i)
            t_p = np.array(temp_t_p)
        feature_p = np.random.randint(0, 4, size=np.random.randint(1,5))
        ab = np.zeros(4)
        ab[feature_p] += mul
        temp_label = np.zeros(t)
        temp_label[t_p] += 1

        x = np.zeros((t, 1))
        y = np.zeros((t, 1))
        w = np.zeros((t, 1))
        z = np.zeros((t, 1))

        for j in range(1, t):
            if j in t_p:
                if adtype == 'non_causal':
                    eps_x[j] += ab[0]
                    eps_w[j] += ab[1]
                    eps_y[j] += ab[2]
                    eps_z[j] += ab[3]
                    x[j, 0] = a[0] * x[j - 1, 0] + eps_x[j]
                    w[j, 0] = a[1] * w[j - 1, 0] + a[2] * x[j - 1, 0] + eps_w[j]
                    y[j, 0] = a[3] * y[j - 1, 0] + a[4] * w[j - 1, 0] + eps_y[j]
                    z[j, 0] = a[5] * z[j - 1, 0] + a[6] * w[j - 1, 0] + a[7] * y[j - 1, 0] + eps_z[j]
                elif adtype == 'causal':
                    # b = np.zeros((8,))
                    # for k in range(8):
                    #     b[k] = np.random.uniform(2, 3, 1)
                    b = a.copy()*3

                    z[j, 0] = b[0] * z[j - 1, 0] + eps_z[j]
                    y[j, 0] = b[1] * y[j - 1, 0] + b[2] * x[j - 1, 0] + eps_y[j]
                    w[j, 0] = b[3] * w[j - 1, 0] + b[4] * w[j - 1, 0] + eps_w[j]
                    x[j, 0] = b[5] * x[j - 1, 0] + b[6] * w[j - 1, 0] + b[7] * y[j - 1, 0] + eps_x[j]

                else:
                    NotImplementedError
            else:
                x[j, 0] = a[0] * x[j - 1, 0] + eps_x[j]
                w[j, 0] = a[1] * w[j - 1, 0] + a[2] * x[j - 1, 0] + eps_w[j]
                y[j, 0] = a[3] * y[j - 1, 0] + a[4] * w[j - 1, 0] + eps_y[j]
                z[j, 0] = a[5] * z[j - 1, 0] + a[6] * w[j - 1, 0] + a[7] * y[j - 1, 0] + eps_z[j]
        x_ab_list.append(np.concatenate((x, w, y, z), axis=1))
        eps_ab_list.append(np.concatenate((eps_x, eps_w, eps_y, eps_z), axis=0).reshape(4, -1).T)
        label_list.append(temp_label)
    causal_struct_value = np.array([[a[0], 0   , 0   , 0   ],
                                     [a[2], a[1], 0   , 0   ],
                                     [0   , a[4], a[3], 0   ],
                                     [0   , a[6], a[7], a[5]]])
    a_signed = np.sign(a)
    signed_causal_struct = np.array([[a_signed[0], 0, 0, 0],
                                     [a_signed[2], a_signed[1], 0, 0],
                                     [0, a_signed[4], a_signed[3], 0],
                                     [0, a_signed[6], a_signed[7], a_signed[5]]])
    causal_struct = np.array([[1, 0, 0, 0],
                              [1, 1, 0, 0],
                              [0, 1, 1, 0],
                              [0, 1, 1, 1]])
    return x_n_list, x_ab_list, eps_n_list, eps_ab_list, causal_struct, causal_struct_value, signed_causal_struct, label_list, a