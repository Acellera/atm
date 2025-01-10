from contextlib import contextmanager
import time
from signal import signal, SIGINT, SIGTERM
from pathlib import Path
import sys
import os
import warnings


class TerminationGuard:
    def __enter__(self):
        self.terminate = False
        self.sigint = signal(SIGINT, self)
        self.sigterm = signal(SIGTERM, self)

    def __call__(self, *_):
        self.terminate = True
        warnings.warn(
            "The process termination is delayed until a critical operation is completed"
        )

    def __exit__(self, *_):
        signal(SIGINT, self.sigint)
        signal(SIGTERM, self.sigterm)
        if self.terminate:
            sys.exit(0)


@contextmanager
def Timer(logger, message):
    logger(f"Started: {message}")
    start = time.monotonic()
    yield
    duration = time.monotonic() - start
    logger(f"Finished: {message} (duration: {duration} s)")


def _exit(message):
    """Print and flush a message to stdout and then exit."""
    print(message)
    sys.stdout.flush()
    print("exiting...")
    sys.exit(1)


def weighted_choice(choices):
    import numpy as np

    """Return a discrete outcome given a set of outcome/weight pairs."""
    r = np.random.random() * np.sum(w for c, w in list(choices))
    for c, w in choices:
        r -= w
        if r < 0:
            return c
    # You should never get here.
    return None


def pairwise_independence_sampling(repl_i, sid_i, replicas, states, U):
    """
    Return a replica "j" to exchange with the given replica "i" based on
    independent sampling from the discrete Metropolis transition matrix, T:

    T_rs = alpha_rs min[1,exp(-du_rs)]  r != s
    T_rr = 1 - sum_(s != r) T_rs        otherwise

    Here r and s are state index permutations, r being the current permutation
    and s the new permutation. alpha_rs = 0 unless r and s differ by a single
    replica swap and alpha_rs = 1/(n-1) otherwise, n being the number of
    replicas/states and (n-1) is the number of permutations, s, differing by
    permutation r by a single swap. du_rs is the change in reduced potential
    energy of the replica exchange ensemble in going from permutation r to
    permutation s (that is, due to a replica swap). Based on the above we have:

    du_rs = u_a(j)+u_b(i)-[u_a(i)+u_b(j)]

    where i and j are the replicas being swapped and a and b are, respectively,
    the states they occupy in r and b and a, respectively, those in s.

    The energies u_a(i), i=1,n and a=1,n, are assumed stored in the input
    "swap matrix," U[a][i].

    In general, the set of replicas across which exchanges are considered is a
    subset of the n replicas. This list is passed in the 'replicas' list.
    Replica "i" ('repl_i') is assumed to be in this list.
    """
    # Evaluate all i-j swap probabilities.
    #
    import numpy as np

    nreplicas = len(replicas)
    ps = np.zeros(nreplicas)  # probability of swap i <-> j
    du = np.zeros(nreplicas)  # Boltzmann exponent, ps ~ exp(-du)

    for j, repl_j, sid_j in zip(range(nreplicas), replicas, states):
        du[j] = (
            U[sid_i][repl_j] + U[sid_j][repl_i] - U[sid_i][repl_i] - U[sid_j][repl_j]
        )
    eu = np.exp(-du)

    pii = 1.0
    i = -1
    f = 1.0 / (float(nreplicas) - 1.0)
    for j in range(nreplicas):
        repl_j = replicas[j]
        if repl_j == repl_i:
            i = j
        else:
            if eu[j] > 1.0:
                ps[j] = f
            else:
                ps[j] = f * eu[j]
            pii -= ps[j]
    try:
        ps[i] = pii
    except IndexError:
        _exit(
            "gibbs_re_j(): unrecoverable error: replica %d not in the "
            "list of waiting replicas?" % i
        )

    return replicas[weighted_choice(list(zip(range(nreplicas), ps)))]


@contextmanager
def set_directory(path: Path):
    """Sets the cwd within the context

    Args:
        path (Path): The path to the cwd

    Yields:
        None
    """

    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)
