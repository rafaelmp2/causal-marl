import torch
import numpy as np

from acd_model import model_loader, utils


# utility function to filter the data 
# taken from https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


# used to debug
def print_matrix_from_array(arr, arr2, x):
    k = 0
    print(arr.shape)
    last_pred_col = []
    mat = np.eye(x, x)
    print(arr)
    for i in range(x):
        for j in range(x):
            if i!=j:
                # save last column
                if j == x-1:
                    last_pred_col.append(int(arr2[k]))

                mat[i, j] = str(int(arr[k])) + "." + str(int(arr2[k]))
                k+=1
    np.fill_diagonal(mat, np.inf)
    print(*mat,sep='\n')
    print(last_pred_col)



class ACD_estimator:
    def __init__(self, args):
        self.n_agents = args.n_agents
        self.encoder, _, _, _, _ = model_loader.load_model(args, None, None, None, None)
        self.rel_rec, self.rel_send = utils.create_rel_rec_send(args, args.num_atoms)

        self.args = args


    def estimate_causality(self, batch):
        episode_num = batch['obs'].shape[0]  # gets number of episode batches in batch  # [b, ts, a, dim]

        #convert data in batch to tensor
        for key in batch.keys():  
            if key == 'actions':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)

        obs, reward = batch['obs'], batch['reward']

        reward_aux = reward.cuda()
        # passing the filter
        obs_smooth, r_smooth = [], []
        reward_np = reward.numpy()
        obs_np = obs.numpy()
        # to save the episodes where there is a reward
        r_idxs = []
        causal_values = None
        if (torch.sum(reward, dim=1) > 0.).any() and obs.shape[1] == self.args.timesteps:  # triggers if there is at least a reward and the episodes are padded(for consistency)
            for e in range(episode_num):
                # trigger if there is a reward in this episode
                if (reward[e] > 0.).any():    
                    r_idxs.append(e)
                    obs_agent_a = []
                    # filter reward
                    transf_reward = savitzky_golay(reward_np[e, :, 0], self.args.timesteps // 2 - 1, 10)  
                    # normalize reward
                    transf_reward = (transf_reward - min(transf_reward)) / (max(transf_reward) - min(transf_reward))
                    r_smooth.append(transf_reward)
                    for a in range(obs_np.shape[2]):
                        obs_agent_a_dim_d = []
                        for d in self.args.range_dims:
                            # filter observation
                            smooth_dim_d = savitzky_golay(obs_np[e, :, a, d], self.args.timesteps // 2 - 1, 10)  
                            obs_agent_a_dim_d.append(smooth_dim_d)
                        obs_agent_a.append(obs_agent_a_dim_d)
                    obs_smooth.append(obs_agent_a)
            # back to normal shapes
            obs_smooth = np.array(obs_smooth).transpose([0, 1, 3, 2]).transpose([0, 2, 1, 3])
            r_smooth = np.array(r_smooth).reshape(-1, 1, self.args.timesteps, 1)

            # now the acd model
            obs_smooth = obs_smooth.transpose([0, 2, 1, 3])
            r_smooth = np.pad(r_smooth, ((0, 0), (0, 0), (0, 0), (0, self.args.dims - 1)))  
            obs_smooth = torch.from_numpy(obs_smooth[:, :, :, :]).float()  # treat conversion numpy to torch 
            r_smooth = torch.from_numpy(r_smooth).float()

            acd_inputs = torch.cat((obs_smooth, r_smooth), dim=1)

            # forward pass inputs throught the acd model
            x = self.encoder(acd_inputs, self.rel_rec, self.rel_send)	
            
            _, acd_acc = x.max(-1)

            # gets last column (o->r)
            causal_values = torch.index_select(acd_acc, dim=-1, index=torch.arange(self.n_agents - 1, (self.n_agents) ** 2 + self.n_agents, self.n_agents).cuda())

        return causal_values, r_idxs

