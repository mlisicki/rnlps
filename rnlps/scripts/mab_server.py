import random
import argparse
import sys
sys.path.insert(0, '../../')
import numpy as np
import multiprocessing
from multiprocessing.connection import Listener
from rnlps.policies.contextual_policies import ThompsonRecurrentNetwork


class IPCBandit(object):
    def __init__(self, connection):
        self.connection = connection
        conn.send({'query': 'num_arms'})
        self.n_arms = int(conn.recv())
        print("Number of arms: {}".format(self.n_arms))
        conn.send({'query': 'context_dims'})
        self.context_dims = int(conn.recv())
        print("Context dimensions: {}".format(self.context_dims))
        self.step = 0

    def reset(self):
        self.step = 0
        # get first context from server
        conn.send({'query': 'context'})
        context = conn.recv()
        print("Received context: {}".format(context))
        return context

    def pull(self, arm):
        conn.send({'arm': arm})
        conn.send({'query': 'reward'})
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

def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="RNLPS bandit server")

    parser.add_argument('--n_units', type=int, nargs='+', default=[32, 32, 32], help='Three arguments describing the number of units at each layer')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='')
    parser.add_argument('--regularise_lambda', type=float, default=0.001, help='')
    parser.add_argument('--epochs', type=int, default=10, help='')
    parser.add_argument('--train_every', type=int, default=32, help='')
    parser.add_argument('--std_targets', type=float, default=0.3, help='')
    parser.add_argument('--std_weights', type=float, default=1.0, help='')
    parser.add_argument('--ipc_port', type=int, default=6000, help='Port to use for IPC.')
    opts, unknown = parser.parse_known_args(args)
    if unknown:
        print("Unknown args: {}".format(unknown))

    return opts

if __name__=="__main__":
    opts = get_options()
    print("Parameters: {}".format(opts))
    address = ('localhost', opts.ipc_port)  # family is deduced to be 'AF_INET'
    listener = Listener(address, authkey=b'rnlps')
    conn = listener.accept()
    multiprocessing.current_process().authkey = b'rnlps'
    print('connection accepted from', listener.last_accepted)

    bandit = IPCBandit(conn)
    policy_parameters = {"n_units": opts.n_units,
                         "learning_rate": opts.learning_rate,
                         "regularise_lambda": opts.regularise_lambda,
                         "epochs": opts.epochs,
                         "train_every": opts.train_every,
                         "std_targets": opts.std_targets,
                         "std_weights": opts.std_weights,
                         "verbose": True,
                         "seed": random.randint(10000, 99999)}
    policy = ThompsonRecurrentNetwork(bandit, **policy_parameters)

    trial_length = np.inf
    trial = policy.interact(trial_length)
    conn.close()
    listener.close()
