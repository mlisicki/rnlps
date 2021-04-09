import random
import sys
sys.path.insert(0, '/Volumes/DATA/neural_combinatorial_optimization/mab/rnlps')
import numpy as np
from multiprocessing.connection import Listener
from rnlps.policies.contextual_policies import ThompsonRecurrentNetwork

class IPCBandit(object):
    def __init__(self, connection):
        self.connection = connection
        conn.send({'query': 'num_arms'})
        self.n_arms = int(conn.recv())
        conn.send({'query': 'context_dims'})
        self.context_dims = int(conn.recv())
        self.step = 0

    def reset(self):
        self.step = 0
        # get first context from server
        conn.send({'query': 'context'})
        context = conn.recv()
        print("Received context: {}".format(context))
        return context

    def pull(self, arm):
        conn.send({'arm': arm, 'query': 'reward'})
        msg = conn.recv()
        if msg == "close":
            return None, None, None
        reward = msg
        print("Received reward: {}".format(reward))
        self.step += 1
        conn.send({'query': 'context'})
        msg = conn.recv()
        if msg == "close":
            return None, None, None
        context = msg
        print("Received context: {}".format(context))
        regret = None
        return reward, context, regret

    def best_arms(self):
        # Many of the bandits I have in my library don't have access to an oracle
        raise NotImplementedError()

    def expected_cumulative_rewards(self, trial_length):
        return np.cumsum(np.zeros(trial_length))

    def __repr__(self):
        return "IPC Bandit"

if __name__=="__main__":
    address = ('localhost', 6000)  # family is deduced to be 'AF_INET'
    listener = Listener(address, authkey=b'rnlps')
    conn = listener.accept()
    print('connection accepted from', listener.last_accepted)

    bandit = IPCBandit(conn)
    policy_parameters = {"n_units": [32, 32, 32],
                         "learning_rate": 0.01,
                         "regularise_lambda": 0.001,
                         "epochs": 3,
                         "train_every": 1,
                         "std_targets": 0.3,
                         "std_weights": 1.0,
                         "verbose": True,
                         "seed": random.randint(10000, 99999)}
    policy = ThompsonRecurrentNetwork(bandit, **policy_parameters)

    trial_length = np.inf
    trial = policy.interact(trial_length)
    conn.close()
    listener.close()
