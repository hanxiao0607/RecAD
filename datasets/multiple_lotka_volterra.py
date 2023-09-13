# Based on https://github.com/smkalami/lotka-volterra-in-python
import numpy as np
from tqdm import tqdm

class MultiLotkaVolterra:
    def __init__(self, p=10, d=2, alpha=1.2, beta=0.2, gamma=1.1, delta=0.05, sigma=0.1, n=10000, adlength=1, adtype='non_causal'):
        """
        Dynamical multi-species Lotka--Volterra system. The original two-species Lotka--Volterra is a special case
        with p = 1 , d = 1.
        @param p: number of predator/prey species. Total number of variables is 2*p.
        @param d: number of GC parents per variable.
        @param alpha: strength of interaction of a prey species with itself.
        @param beta: strength of predator -> prey interaction.
        @param gamma: strength of interaction of a predator species with itself.
        @param delta: strength of prey -> predator interaction.
        @param sigma: scale parameter for the noise.
        """

        assert p >= d and p % d == 0

        self.p = p
        self.d = d
        self.n = n

        # Coupling strengths
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.sigma = sigma
        self.adlength = adlength
        self.adtype = adtype

    def simulate(self, n:int, t: int, dt=0.01, downsample_factor=10, seed=None):
        if seed is not None:
            np.random.seed(seed)
        lst_n = []
        lst_ab = []
        eps_n = []
        eps_ab = []
        lst_labels = []
        lst_anomaly_term = []
        for _ in tqdm(range(n)):
            xs_0 = np.random.uniform(10, 20, size=(self.p, ))
            ys_0 = np.random.uniform(10, 20, size=(self.p, ))

            ts = np.arange(t) * dt

            # Simulation Loop
            xs = np.zeros((t, self.p))
            ys = np.zeros((t, self.p))
            eps_x = np.zeros((t, self.p))
            eps_y = np.zeros((t, self.p))
            xs[0, :] = xs_0
            ys[0, :] = ys_0

            xs_ab = np.zeros((t, self.p))
            ys_ab = np.zeros((t, self.p))
            eps_x_ab = np.zeros((t, self.p))
            eps_y_ab = np.zeros((t, self.p))
            label = np.zeros(t)
            anomaly_term = np.zeros((t, self.p*2))
            xs_ab[0, :] = xs_0
            ys_ab[0, :] = ys_0
            t_p = np.random.randint((0.5*t)//downsample_factor, t//downsample_factor, size=1)
            if self.adlength > 1:
                temp_t_p = []
                for i in range(self.adlength):
                    temp_t_p.append(t_p + i)
                t_p = np.array(temp_t_p)
            pp_p = np.random.randint(0, 2, size=1)
            feature_p = np.random.randint(0, self.p, size=np.random.randint(3, self.p+1))
            count = 0
            for k in range(t - 1):
                if k in (t_p*downsample_factor)-1:
                    xs[k + 1, :], ys[k + 1, :], eps_x[k + 1, :], eps_y[k + 1, :], xs_ab[k + 1, :], ys_ab[k + 1, :], \
                    eps_x_ab[k + 1, :], eps_y_ab[k + 1, :], label[k + 1] = self.next(xs[k, :], ys[k, :], xs_ab[k, :], ys_ab[k, :],
                                                                       dt, ab=1, pp_p=pp_p, feature_p=feature_p, adtype=self.adtype, seq_k=count)
                    anomaly_term[k+1, feature_p + (pp_p * self.p)] = 1
                    count += 1
                else:
                    xs[k + 1, :], ys[k + 1, :], eps_x[k + 1, :], eps_y[k + 1, :], xs_ab[k + 1, :], ys_ab[k + 1, :], \
                    eps_x_ab[k + 1, :], eps_y_ab[k + 1, :], label[k + 1] = self.next(xs[k, :], ys[k, :], xs_ab[k, :], ys_ab[k, :], dt)

            lst_n.extend([np.concatenate((xs[::downsample_factor, :], ys[::downsample_factor, :]), 1)])
            eps_n.extend([np.concatenate((eps_x[::downsample_factor, :], eps_y[::downsample_factor, :]), 1)])
            lst_ab.extend([np.concatenate((xs_ab[::downsample_factor, :], ys_ab[::downsample_factor, :]), 1)])
            eps_ab.extend([np.concatenate((eps_x_ab[::downsample_factor, :], eps_y_ab[::downsample_factor, :]), 1)])
            lst_labels.append(label[::downsample_factor])
            lst_anomaly_term.append(anomaly_term[::downsample_factor])

        causal_struct = np.zeros((self.p * 2, self.p * 2))
        signed_causal_struct = np.zeros((self.p * 2, self.p * 2))
        for j in range(self.p):
            # Self causation
            causal_struct[j, j] = 1
            causal_struct[j + self.p, j + self.p] = 1

            signed_causal_struct[j, j] = +1
            signed_causal_struct[j + self.p, j + self.p] = -1

            # Predator-prey relationships
            causal_struct[j, int(np.floor((j + self.d) / self.d) * self.d - 1 + self.p - self.d + 1):int(np.floor((j + self.d) / self.d) * self.d + self.p)] = 1
            causal_struct[j + self.p, int(np.floor((j + self.d) / self.d) * self.d - self.d + 1 - 1):int(np.floor((j + self.d) / self.d) * self.d)] = 1

            signed_causal_struct[j, int(np.floor((j + self.d) / self.d) * self.d - 1 + self.p - self.d + 1):int(np.floor((j + self.d) / self.d) * self.d + self.p)] = -1
            signed_causal_struct[j + self.p, int(np.floor((j + self.d) / self.d) * self.d - self.d + 1 - 1):int(np.floor((j + self.d) / self.d) * self.d)] = +1
        return np.array(lst_n)[:, 50:, :], np.array(eps_n)[:, 50:, :], np.array(lst_ab)[:, 50:, :],\
                np.array(eps_ab)[:, 50:, :], np.array(lst_labels)[:, 50:], causal_struct, signed_causal_struct,\
                np.array(lst_anomaly_term)[:, 50:, :]

    # Dynamics
    # State transitions using the Runge-Kutta method
    def next(self, x, y, x_ab, y_ab, dt, ab=0, pp_p=0, feature_p=None, adtype='non_causal', seq_k=0):
        if ab == 1:
            xdot1, ydot1 = self.f(x, y)
            xdot2, ydot2 = self.f(x + xdot1 * dt / 2, y + ydot1 * dt / 2)
            xdot3, ydot3 = self.f(x + xdot2 * dt / 2, y + ydot2 * dt / 2)
            xdot4, ydot4 = self.f(x + xdot3 * dt, y + ydot3 * dt)
            # Add noise to simulations
            eps_x = np.random.normal(scale=self.sigma, size=(self.p,))
            eps_y = np.random.normal(scale=self.sigma, size=(self.p,))
            eps_x_ab = eps_x.copy()
            eps_y_ab = eps_y.copy()
            xnew = x + (xdot1 + 2 * xdot2 + 2 * xdot3 + xdot4) * dt / 6 + \
                   eps_x
            ynew = y + (ydot1 + 2 * ydot2 + 2 * ydot3 + ydot4) * dt / 6 + \
                   eps_y
            if adtype == 'non_causal':
                xdot1_ab, ydot1_ab = self.f(x_ab, y_ab)
                xdot2_ab, ydot2_ab = self.f(x_ab + xdot1_ab * dt / 2, y + ydot1_ab * dt / 2)
                xdot3_ab, ydot3_ab = self.f(x_ab + xdot2_ab * dt / 2, y_ab + ydot2_ab * dt / 2)
                xdot4_ab, ydot4_ab = self.f(x_ab + xdot3_ab * dt, y + ydot3_ab * dt)

                # Add noise to simulations
                if pp_p == 0:
                    eps_x_ab[feature_p] += 100
                else:
                    eps_y_ab[feature_p] += 100

                xnew_ab = x_ab + (xdot1_ab + 2 * xdot2_ab + 2 * xdot3_ab + xdot4_ab) * dt / 6 + \
                          eps_x_ab
                ynew_ab = y_ab + (ydot1_ab + 2 * ydot2_ab + 2 * ydot3_ab + ydot4_ab) * dt / 6 + \
                          eps_y_ab
            else:
                xdot1_ab, ydot1_ab = self.f(x_ab, y_ab)
                xdot2_ab, ydot2_ab = self.f(x_ab + xdot1_ab * dt / 2, y + ydot1_ab * dt / 2)
                xdot3_ab, ydot3_ab = self.f(x_ab + xdot2_ab * dt / 2, y_ab + ydot2_ab * dt / 2)
                xdot4_ab, ydot4_ab = self.f(x_ab + xdot3_ab * dt, y + ydot3_ab * dt)
                lst_val = [10000, 100, 100]
                if pp_p == 0:
                    xnew_temp = x_ab + (xdot1_ab + 2 * xdot2_ab + 2 * xdot3_ab + xdot4_ab) * dt / 6 + \
                                eps_x_ab
                    for i in feature_p:
                        xdot1_ab[i] = xdot1_ab[i]*lst_val[seq_k]
                        if xdot1_ab[i] > 120000:
                            xdot1_ab[i] = 120000
                        if xdot1_ab[i] < 60000:
                            xdot1_ab[i] = 60000
                    xnew_ab = x_ab + xdot1_ab * dt / 6 + \
                              eps_x_ab
                    ynew_ab = y_ab + (ydot1_ab + 2 * ydot2_ab + 2 * ydot3_ab + ydot4_ab) * dt / 6 + \
                              eps_y_ab
                    eps_x_ab = eps_x_ab + xnew_ab - xnew_temp
                else:
                    ynew_temp = y_ab + (ydot1_ab + 2 * ydot2_ab + 2 * ydot3_ab + ydot4_ab) * dt / 6 + \
                                eps_y_ab
                    for i in feature_p:
                        ydot1_ab[i] = ydot1_ab[i]*lst_val[seq_k]
                        if ydot1_ab[i] > 120000:
                            ydot1_ab[i] = 120000
                        if ydot1_ab[i] < 60000:
                            ydot1_ab[i] = 60000
                    xnew_ab = x_ab + (xdot1_ab + 2 * xdot2_ab + 2 * xdot3_ab + xdot4_ab) * dt / 6 + \
                              eps_x_ab
                    ynew_ab = y_ab + ydot1_ab * dt / 6 + \
                              eps_y_ab
                    eps_y_ab = eps_y_ab + ynew_ab - ynew_temp
            # Clip from below to prevent populations from becoming negative
        else:
            xdot1, ydot1 = self.f(x, y)
            xdot2, ydot2 = self.f(x + xdot1 * dt / 2, y + ydot1 * dt / 2)
            xdot3, ydot3 = self.f(x + xdot2 * dt / 2, y + ydot2 * dt / 2)
            xdot4, ydot4 = self.f(x + xdot3 * dt, y + ydot3 * dt)
            # Add noise to simulations
            eps_x = np.random.normal(scale=self.sigma, size=(self.p, ))
            eps_y = np.random.normal(scale=self.sigma, size=(self.p, ))
            eps_x_ab = eps_x.copy()
            eps_y_ab = eps_y.copy()
            xnew = x + (xdot1 + 2 * xdot2 + 2 * xdot3 + xdot4) * dt / 6 + \
                   eps_x
            ynew = y + (ydot1 + 2 * ydot2 + 2 * ydot3 + ydot4) * dt / 6 + \
                   eps_y

            xdot1_ab, ydot1_ab = self.f(x_ab, y_ab)
            xdot2_ab, ydot2_ab = self.f(x_ab + xdot1_ab * dt / 2, y + ydot1_ab * dt / 2)
            xdot3_ab, ydot3_ab = self.f(x_ab + xdot2_ab * dt / 2, y_ab + ydot2_ab * dt / 2)
            xdot4_ab, ydot4_ab = self.f(x_ab + xdot3_ab * dt, y + ydot3_ab * dt)
            # Add noise to simulations
            xnew_ab = x_ab + (xdot1_ab + 2 * xdot2_ab + 2 * xdot3_ab + xdot4_ab) * dt / 6 + \
                      eps_x_ab
            ynew_ab = y_ab + (ydot1_ab + 2 * ydot2_ab + 2 * ydot3_ab + ydot4_ab) * dt / 6 + \
                      eps_y_ab
            # Clip from below to prevent populations from becoming negative
        return np.maximum(xnew, 0), np.maximum(ynew, 0), eps_x, eps_y, np.maximum(xnew_ab, 0), np.maximum(ynew_ab, 0), \
               eps_x_ab, eps_y_ab, ab

    def next_value(self, data, eps_norm, dt=0.01, downsample_factor=10):
        x_all = data[:, :self.p]
        y_all = data[:, self.p:]
        lst_results = []
        for k in range(len(data)):
            x = x_all[k].numpy()
            y = y_all[k].numpy()
            for i in range(downsample_factor):
                xdot1, ydot1 = self.f(x, y)
                xdot2, ydot2 = self.f(x + xdot1 * dt / 2, y + ydot1 * dt / 2)
                xdot3, ydot3 = self.f(x + xdot2 * dt / 2, y + ydot2 * dt / 2)
                xdot4, ydot4 = self.f(x + xdot3 * dt, y + ydot3 * dt)
                # Add noise to simulations
                if i == downsample_factor-1:
                    eps_x = eps_norm[k, :self.p].numpy()
                    eps_y = eps_norm[k, self.p:].numpy()
                else:
                    eps_x = np.zeros((self.p,))
                    eps_y = np.zeros((self.p,))
                xnew = x + (xdot1 + 2 * xdot2 + 2 * xdot3 + xdot4) * dt / 6 + \
                       eps_x
                ynew = y + (ydot1 + 2 * ydot2 + 2 * ydot3 + ydot4) * dt / 6 + \
                       eps_y
                x = np.maximum(xnew, 0).copy()
                y = np.maximum(ynew, 0).copy()
            lst_results.append(np.concatenate((x, y)))
        return np.array(lst_results)



    def f(self, x, y):
        xdot = np.zeros((self.p, ))
        ydot = np.zeros((self.p, ))

        for j in range(self.p):
            y_Nxj = y[int(np.floor((j + self.d) / self.d) * self.d - self.d + 1 - 1):int(np.floor((j + self.d) /
                                                                                                  self.d) * self.d)]
            x_Nyj = x[int(np.floor((j + self.d) / self.d) * self.d - self.d + 1 - 1):int(np.floor((j + self.d) /
                                                                                                  self.d) * self.d)]
            xdot[j] = self.alpha * x[j] - self.beta * x[j] * np.sum(y_Nxj) - 2.75 * 10e-5 * (x[j] / 200) ** 2
            ydot[j] = self.delta * np.sum(x_Nyj) * y[j] - self.gamma * y[j]
        return xdot, ydot
