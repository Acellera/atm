import logging
import math
from openmm.unit import kelvin, kilocalories_per_mole
from atm.utils import pairwise_independence_sampling
from atm.ommreplica import OMMReplicaATM
from atm.ommsystem import OMMSystemAmberRBFE
from atm.worker import OMMWorkerATM
from atm.utils import TerminationGuard, Timer
import sys

logger = logging.getLogger(__name__)


class OpenmmJobAmberRBFE:

    def __init__(self, config_file):
        import yaml
        import json

        self.logger = logger

        self.logger.info("Configuration:")
        if config_file.endswith(".yaml"):
            with open(config_file, "r") as f:
                self.config = yaml.safe_load(f)
        elif config_file.endswith(".json"):
            with open(config_file, "r") as f:
                self.config = json.load(f)
        else:
            raise ValueError("Invalid configuration file format")

        for key, value in self.config.items():
            self.logger.info(f"{key}: {value}")

        if self.config.get("VERBOSE"):
            self.logger.setLevel(logging.DEBUG)

        self.basename = self.config["BASENAME"]
        self.state_params = self._getStateParams()
        self.nreplicas = len(self.state_params)

    def _getStateParams(self):
        lambdas = self.config["LAMBDAS"]
        directions = self.config["DIRECTION"]
        intermediates = self.config["INTERMEDIATE"]
        lambda1s = self.config["LAMBDA1"]
        lambda2s = self.config["LAMBDA2"]
        alphas = self.config["ALPHA"]
        uhs = self.config["U0"]
        w0s = self.config["W0COEFF"]
        temperatures = self.config["TEMPERATURES"]

        assert len(directions) == len(lambdas)
        assert len(intermediates) == len(lambdas)
        assert len(lambda1s) == len(lambdas)
        assert len(lambda2s) == len(lambdas)
        assert len(alphas) == len(lambdas)
        assert len(uhs) == len(lambdas)
        assert len(w0s) == len(lambdas)
        assert len(temperatures) == 1

        self.logger.info("State parameters")
        state_params = []
        for lambda_, direction, intermediate, lambda1, lambda2, alpha, uh, w0 in zip(
            lambdas, directions, intermediates, lambda1s, lambda2s, alphas, uhs, w0s
        ):
            par = {}
            par["lambda"] = float(lambda_)
            par["atmdirection"] = float(direction)
            par["atmintermediate"] = float(intermediate)
            par["lambda1"] = float(lambda1)
            par["lambda2"] = float(lambda2)
            par["alpha"] = float(alpha) / kilocalories_per_mole
            par["uh"] = float(uh) * kilocalories_per_mole
            par["Umax"] = float(self.config.get("UMAX")) * kilocalories_per_mole
            par["Ubcore"] = float(self.config.get("UBCORE")) * kilocalories_per_mole
            par["Acore"] = float(self.config.get("ACORE"))
            par["w0"] = float(w0) * kilocalories_per_mole
            par["temperature"] = float(temperatures[0]) * kelvin
            state_params.append(par)
            self.logger.info(f"    State: {par}")

        return state_params

    def setupJob(self):
        with Timer(self.logger.info, "ATM setup"):

            with Timer(self.logger.info, "create system"):
                prmtopfile = self.basename + ".prmtop"
                crdfile = self.basename + ".inpcrd"
                ommsystem = OMMSystemAmberRBFE(
                    self.basename, self.config, prmtopfile, crdfile, self.logger
                )
                ommsystem.create_system()

            with Timer(self.logger.info, "create worker"):
                self.worker = OMMWorkerATM(ommsystem, self.config, self.logger)

            with Timer(self.logger.info, "create replicas"):
                self.replicas = []
                for i in range(self.nreplicas):
                    replica = OMMReplicaATM(i, self.basename, self.worker, self.logger)
                    if not replica.get_stateid():
                        replica.set_state(i, self.state_params[i])
                    self.replicas.append(replica)

                self.replica_states = [
                    replica.get_stateid() for replica in self.replicas
                ]
                for i, replica in enumerate(self.replicas):
                    self.logger.info(
                        f"Replica {i}: cycle {replica.get_cycle()}, state {replica.get_stateid()}"
                    )

            with Timer(self.logger.info, "update replicas"):
                self._updateReplicas()

    def scheduleJobs(self):
        with Timer(self.logger.info, "ATM simulations"):

            last_sample = self.replicas[0].get_cycle()
            num_samples = str(self.config["MAX_SAMPLES"])
            if num_samples.startswith("+"):
                num_extra_samples = int(num_samples[1:])
                num_samples = num_extra_samples + last_sample - 1
                self.logger.info(f"Additional number of samples: {num_extra_samples}")
            else:
                num_samples = int(num_samples)
                self.logger.info(f"Target number of samples: {num_samples}")

            # in memory storage for output data
            output_data = [[] for _ in range(len(self.replicas))]

            for isample in range(last_sample, num_samples + 1):

                with Timer(self.logger.info, f"sample {isample}"):

                    for irepl, replica in enumerate(self.replicas):
                        with Timer(
                            self.logger.info, f"sample {isample}, replica {irepl}"
                        ):
                            assert replica.get_cycle() == isample
                            self.worker.run(replica)

                    with Timer(self.logger.info, "exchange replicas"):
                        self._exhangeReplicas()

                    with Timer(self.logger.info, "update replicas"):
                        self._updateReplicas()

                    with Timer(self.logger.info, "save replica samples"):
                        for irepl, replica in enumerate(self.replicas):
                            # save output data to write at same freq at checkpoint
                            data = replica.save_out()
                            output_data[irepl].append(data)

                    if (
                        isample % int(self.config["CHECKPOINT_FREQUENCY"]) == 0
                        or isample == num_samples
                    ):
                        # at every checkpoint_frequency samples write the the .out files and the checkpoint xml.
                        # also write the output if this is the last sample

                        with TerminationGuard():
                            with Timer(
                                self.logger.info,
                                "write replicas samples and trajectories",
                            ):

                                if not all(
                                    len(lst) == len(output_data[0])
                                    for lst in output_data
                                ):
                                    self.logger.error(
                                        "Inconsistent output data. Did one of the replicas crash?"
                                    )
                                    raise ValueError

                                for irepl, replica in enumerate(self.replicas):
                                    # write multiple lines to the out file
                                    replica.write_out(output_data[irepl])
                                    # clear the saved lines
                                    output_data[irepl] = []

                                    if (
                                        replica.get_mdsteps()
                                        % int(self.config["TRJ_FREQUENCY"])
                                        == 0
                                    ):
                                        replica.save_dcd()

                            with Timer(self.logger.info, "checkpointing"):
                                for replica in self.replicas:
                                    replica.save_checkpoint()

                    # Report progress on GPUGRID
                    progress = float(isample - last_sample + 1) / float(
                        num_samples - last_sample + 1
                    )
                    open("progress", "w").write(str(progress))

    def _updateReplicas(self):
        for replica, stateid in zip(self.replicas, self.replica_states):
            replica.set_state(stateid, self.state_params[stateid])

    def _exhangeReplicas(self):

        # Matrix of replica energies in each state.
        swap_matrix = self._computeSwapMatrix(
            range(self.nreplicas), self.replica_states
        )
        self.logger.debug("Swap matrix")
        for row in swap_matrix:
            self.logger.debug(f"    {row}")

        self.logger.debug(f"Replica states before: {self.replica_states}")
        for repl_i in range(self.nreplicas):
            sid_i = self.replica_states[repl_i]
            repl_j = pairwise_independence_sampling(
                repl_i, sid_i, range(self.nreplicas), self.replica_states, swap_matrix
            )
            if repl_j != repl_i:
                sid_i = self.replica_states[repl_i]
                sid_j = self.replica_states[repl_j]
                self.replica_states[repl_i] = sid_j
                self.replica_states[repl_j] = sid_i
                self.logger.info(f"Replica {repl_i}: {sid_i} --> {sid_j}")
                self.logger.info(f"Replica {repl_j}: {sid_j} --> {sid_i}")

        self.logger.debug(f"Replica states after: {self.replica_states}")

    def _computeSwapMatrix(self, repls, states):
        """
        Compute matrix of dimension-less energies: each column is a replica
        and each row is a state so U[i][j] is the energy of replica j in state
        i.
        """
        U = [[0.0 for _ in range(self.nreplicas)] for _ in range(self.nreplicas)]

        n = len(repls)

        # collect replica parameters and potentials
        par = [self._getPar(k) for k in repls]
        pot = [self._getPot(k) for k in repls]

        for i in range(n):
            repl_i = repls[i]
            for j in range(n):
                sid_j = states[j]
                # energy of replica i in state j
                U[sid_j][repl_i] = _reduced_energy(par[j], pot[i])
        return U

    def _getPar(self, repl):
        _, par = self.replicas[repl].get_state()
        return par

    # customized getPot to return the unperturbed potential energy
    # of the replica U0 = U - W_lambda(u)
    def _getPot(self, repl):
        replica = self.replicas[repl]
        pot = replica.get_energy()
        epot = pot["potential_energy"]
        pertpot = pot["perturbation_energy"]
        _, par = replica.get_state()
        lambda1 = par["lambda1"]
        lambda2 = par["lambda2"]
        alpha = par["alpha"]
        uh = par["uh"]
        w0 = par["w0"]
        ebias = _softplus(lambda1, lambda2, alpha, uh, w0, pertpot)
        pot["unbiased_potential_energy"] = epot - ebias
        pot["direction"] = par["atmdirection"]
        pot["intermediate"] = par["atmintermediate"]
        return pot


def _reduced_energy(par, pot):
    lambda1 = par["lambda1"]
    lambda2 = par["lambda2"]
    alpha = par["alpha"]
    uh = par["uh"]
    w0 = par["w0"]
    state_direction = par["atmdirection"]
    state_intermediate = par["atmintermediate"]
    epot0 = pot["unbiased_potential_energy"]
    pertpot = pot["perturbation_energy"]
    replica_direction = pot["direction"]
    replica_intermediate = pot["intermediate"]

    kb = 0.0019872041 * kilocalories_per_mole / kelvin
    beta = 1.0 / (kb * par["temperature"])

    if (replica_direction == state_direction) or (
        state_intermediate > 0 and replica_intermediate > 0
    ):
        ebias = _softplus(lambda1, lambda2, alpha, uh, w0, pertpot)
        return beta * (epot0 + ebias)
    else:
        # prevent exchange
        large_energy = 1.0e12
        return large_energy


# evaluates the softplus function
def _softplus(lambda1, lambda2, alpha, uh, w0, uf):
    ee = 1.0 + math.exp(-alpha * (uf - uh))
    softplusf = lambda2 * uf + w0
    if alpha._value > 0.0:
        softplusf += ((lambda2 - lambda1) / alpha) * math.log(ee)
    return softplusf


def rbfe_production(config_file=None):
    from atm.utils import set_directory
    from pathlib import Path
    import os

    if config_file is None:
        config_file = sys.argv[1]

    with set_directory(Path(config_file).parent):
        rx = OpenmmJobAmberRBFE(os.path.basename(os.path.abspath(config_file)))
        rx.setupJob()
        rx.scheduleJobs()


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Specify ONE input file"

    rbfe_production(sys.argv[1])
