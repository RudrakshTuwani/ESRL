from typing import List, Dict

import itertools
from collections import defaultdict
import pandas as pd
import numpy as np
import scipy.stats as stats
from .sample import sample_MDPs


def compute_propensity_scores(df: pd.DataFrame, A_space: List, S_space: List, tau: int) -> pd.DataFrame:
    """ Takes as input a pandas dataframe with columns timestep, state, and action
        and computes propensity scores. """

    for column in ['timestep', 'state', 'action']:
        if column not in df:
            raise KeyError(
                f"Did not find {column} column in the supplied dataframe.")

    if (type(tau) != int) or (tau <= 0):
        raise ValueError("tau needs to be an integer greater than 0.")

    # Number of times action a was selected at time step t in state s
    n_tsa = df.groupby(['timestep', 'state', 'action']
                       ).agg(len).iloc[:, 0].to_dict()
    n_tsa = defaultdict(lambda: 0, n_tsa)

    # Number of times state s was observed at time step t
    n_ts = df.groupby(['timestep', 'state']).agg(len).iloc[:, 0].to_dict()
    n_ts = defaultdict(lambda: 0, n_ts)

    # Probability of taking action a at time step t in state s
    pi_tsa = list()
    for a, s, t in itertools.product(A_space, S_space, np.arange(1, tau+1, 1)):
        num = n_tsa[(t, s, a)]
        denom = n_ts[(t, s)]
        prob = None
        if denom != 0:
            if num != 0:
                prob = num/denom
            else:
                prob = 1e-3
        else:
            pass
            # print(f"Warning: the state {s} hasn't been visited at the timestep {t}.")
            # raise ValueError(f"The state {s} hasn't been visited at the timestep {t}.")

        pi_tsa.append({
            'action': a,
            'state': s,
            'timestep': t,
            'probability': prob
        })
    pi_tsa = pd.DataFrame(pi_tsa)

    return pi_tsa


def compute_behavior_policy(df: pd.DataFrame, A_space: List, S_space: List, tau: int) -> pd.Series:
    """ Computes the behavior policy based on the most likely action """

    def max_prob_action(action_prob: pd.Series):
        max_prob = action_prob['probability'].fillna(0).max()
        if max_prob == 0:
            return np.random.choice(A_space)
        else:
            most_likely_actions = action_prob.loc[action_prob['probability']
                                                  == max_prob, 'action'].tolist()
            return np.random.choice(most_likely_actions)

    pi_tsa = compute_propensity_scores(df, A_space, S_space, tau)

    # Find the action w
    pi_st = pi_tsa.groupby(['state', 'timestep']).apply(max_prob_action)

    assert pi_st.isna().sum(
    ) == 0, "Found null actions in computed behavior policy"
    # TODO: In case of null actions, randomly sample action

    return pi_st


def P_H0_MV(s: int, t: int, a_behavior: int, a_mu: int, Mk_R_sa: Dict, Mk_P_sas: Dict,
            kset: np.array, V_st: Dict, S_space: List, A_space: List):
    Qs = np.zeros((len(kset), len(A_space)))
    for i, k in enumerate(kset):
        # compute Q values for current state of interest
        R_sa, P_sas = Mk_R_sa[k], Mk_P_sas[k]
        Qs[i, :] = [(R_sa[(s, a)] + sum([P_sas[(s, a)][int(nxt_s)]*V_st[k]
                                         [(nxt_s, t+1)] for nxt_s in S_space])) for a in A_space]

    return np.mean(Qs[:, a_mu] < Qs[:, a_behavior]), Qs


def compute_SOMVPRSL(df: pd.DataFrame, A_space: List, S_space: List, tau: int,
                     priors: Dict[str, float], K: int, alpha: float):
    pi_st = compute_behavior_policy(df, A_space, S_space, tau)

    assert K//2 == K/2, "K needs to be an even number"
    assert (len(set(priors.keys()) & set(['m0', 'lamb0', 'alpha0', 'gamma0'])) == 4) & (len(set(priors.keys())) == 4),\
        "Found invalid keys in the priors dict. Need to specify 'm0','lamb0','alpha0','gamma0'."

    I_1, I_2 = np.arange(0, K//2), np.arange(K//2, K)

    # Sample K MDPs from the posterior
    Mk_R_sa, Mk_P_sas = sample_MDPs(
        df=df, A_space=A_space, S_space=S_space, n_samples=K, **priors)

    # Initialize value Vtau(S) and policy function dictionaries
    V_st = {k: {(s, tau+1): 0 for s in S_space} for k in range(K)}
    mu_st_alpha, mu_st = {k: {} for k in range(K)}, {k: {} for k in range(K)}
    maj_vote_mu, maj_vote_mu_alpha, maj_vote_set_alpha = {}, {}, {}
    Qs_st = {}

    # Iterate over timestep and states
    for t in range(tau, 0, -1):
        for s in S_space:

            # For each sample
            for k in range(K):

                # Compute the Q value
                R_sa, P_sas = Mk_R_sa[k], Mk_P_sas[k]
                q_vals = [(R_sa[(s, a)] + sum([P_sas[(s, a)][int(nxt_s)]*V_st[k]
                                               [(nxt_s, t+1)] for nxt_s in S_space])) for a in A_space]

                # Compute mu_k
                mu_st[k][(s, t)] = np.argmax(q_vals)

            # Compute policy based on majority vote:
            maj_vote_mu[(s, t)] = int(stats.mode(
                [mu_st[k][(s, t)] for k in I_1])[0])

            # Compute P(H_0|s,d,H_T)
            P_0, Qs_st[(s, t)] = P_H0_MV(s, t, a_behavior=pi_st[(s, t)], a_mu=maj_vote_mu[(s, t)],
                                         Mk_R_sa=Mk_R_sa, Mk_P_sas=Mk_P_sas, kset=I_2, V_st=V_st,
                                         S_space=S_space, A_space=A_space)

            for k in range(K):
                # Compute policy based on P-value rule
                mu_st_alpha[k][(s, t)] = mu_st[k][(
                    s, t)] if P_0 < alpha else pi_st[(s, t)]

                # Compute value function based on chosen policy
                V_st[k][(s, t)] = float(*[(R_sa[(s, a)] + sum([P_sas[(s, a)][int(nxt_s)]*V_st[k][(nxt_s, t+1)]
                                                               for nxt_s in S_space])) for a in [mu_st_alpha[k][(s, t)]]])

            # Compute policy based on majority vote, and set of k's which chose the most common action:
            maj_vote_mu_alpha[(s, t)] = int(stats.mode(
                [mu_st_alpha[k][(s, t)] for k in I_1])[0])
            maj_vote_set_alpha[(s, t)] = [
                k for k in I_1 if maj_vote_mu_alpha[(s, t)] == mu_st_alpha[k][(s, t)]]

    # Define majority voting set and check if there are models in all:
    MV_set = set(k for k in range(K))
    for key in maj_vote_set_alpha.keys():
        MV_set = MV_set.intersection(maj_vote_set_alpha[key])
    if len(MV_set) > 0:
        chosen_k = np.random.choice(list(MV_set))
    else:
        chosen_k = int(stats.mode([k for key in list(
            maj_vote_set_alpha.keys()) for k in maj_vote_set_alpha[key]])[0])

    # return mu_st_alpha[chosen_k], Mk_R_sa[chosen_k], Mk_P_sas[chosen_k], Qs_st
    return mu_st_alpha[chosen_k]


def sample_Vs(df: pd.DataFrame, A_space: List, S_space: List, tau: int, priors: Dict,
              n_samples: int, mu_st: Dict, random_starting_state: bool, seed: int = 2020):

    np.random.seed(seed)

    V_samps = np.zeros((n_samples))
    starting_state_samps = np.zeros((n_samples))

    # Sample MDP from posterior based on IPW repo:
    R_sa_dict, P_sas_dict = sample_MDPs(
        df=df, A_space=A_space, S_space=S_space, n_samples=n_samples, **priors)

    for k in range(n_samples):

        # Sample MDP from posterior based on IPW repo:
        R_sa, P_sas = R_sa_dict[k], P_sas_dict[k]

        # initialize value V and policy function dictionaries:
        if random_starting_state:
            s = int(np.random.choice(S_space, 1))
        else:
            s = 0  # all episodes start at state = 0 in riverswim
        starting_state_samps[k] = s

        V_s0 = 0
        weights = [1]

        for t in range(1, tau+1):
            a = mu_st[(s, t)]
            sum_pi = 1
            V_s0 += R_sa[(s, a)]*np.product(weights)/sum_pi

            # Sample a state according to the sampled MDP and action taken
            s = int(np.random.choice(len(S_space), 1, p=P_sas[(s, a)]))

        V_samps[k] = V_s0

    return V_samps, starting_state_samps
