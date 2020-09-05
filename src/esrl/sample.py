import pandas as pd
import numpy as np
import itertools
from collections import defaultdict


def normal_gamma_sample(m0: float, lamb0: float, alpha0: float, gamma0: float, n_samples: int,
                        r_bar: float = None, r_sq_bar: float = None, n_sa: int = None):

    if (r_bar and r_sq_bar and n_sa):
        m = (lamb0*m0+n_sa*r_bar)/(lamb0+n_sa)
        lamb = lamb0 + n_sa
        alpha = alpha0 + n_sa/2
        gamma = gamma0 + 0.5*n_sa*(r_sq_bar-r_bar**2) + \
            (n_sa*lamb0*(r_bar-m0)**2)/(2*(lamb0+n_sa))

    elif (r_bar or r_sq_bar or n_sa):
        raise ValueError(
            "Need to specify non-null values for r_bar, r_sq_bar and n_sa. ")

    else:
        m, lamb, alpha, gamma = m0, lamb0, alpha0, gamma0

    tautau = np.random.gamma(alpha, gamma, n_samples)
    sigma = 1/(lamb*tautau)
    mu_sa = np.random.normal(m, sigma, n_samples)

    return mu_sa


def sample_MDPs(df: pd.DataFrame, A_space: list, S_space: list, m0: float,
                lamb0: float, alpha0: float, gamma0: float, n_samples: int):
    R_samples, P_samples = [{} for _ in range(n_samples)], [
        {} for _ in range(n_samples)]

    # Dirichlet prior for next state
    dirich_conc = {(s, a): {ns: 1/len(S_space) for ns in S_space}
                   for (s, a) in itertools.product(S_space, A_space)}

    if df is not None:
        # For each state-action pair
        grouped_df = df.groupby(['state', 'action'])

        # Emprical freq of next state
        ns_data_freq = grouped_df['next_state'].agg(
            lambda x: defaultdict(lambda: 0, x.value_counts().items()))
        ns_data_freq = defaultdict(lambda: defaultdict(
            lambda: 0), ns_data_freq.to_dict())

        # Empirical statistics for rewards
        n_sa = defaultdict(
            lambda: None, grouped_df['reward'].agg(len).to_dict())
        r_bar = defaultdict(
            lambda: None, grouped_df['reward'].agg(pd.Series.mean).to_dict())
        r_sq_bar = defaultdict(
            lambda: None, (grouped_df['reward'].agg(lambda s: (s**2).mean())).to_dict())

    # Sample
    for (s, a) in itertools.product(S_space, A_space):
        mean_reward_samples = normal_gamma_sample(m0=m0, lamb0=lamb0, alpha0=alpha0, gamma0=gamma0, n_samples=n_samples,
                                                  r_bar=r_bar[(s, a)], r_sq_bar=r_sq_bar[(s, a)], n_sa=n_sa[(s, a)])
        trans_probs_samples = np.random.dirichlet(
            [dirich_conc[(s, a)][ns] + ns_data_freq[(s, a)][ns] for ns in S_space], n_samples)

        for k in range(n_samples):
            R_samples[k][(s, a)] = mean_reward_samples[k]
            P_samples[k][(s, a)] = trans_probs_samples[k]

    return R_samples, P_samples


# def sample_Vs(seed: int = 2020):
#     np.random.seed(seed)
