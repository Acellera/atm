import os
from openmm import OpenMMException, Platform
from openmm.app import Simulation, StateDataReporter
from openmm.unit import kelvin, kilojoules_per_mole
from atm.utils import Timer


class OMMWorkerATM:

    def __init__(self, ommsystem, config, logger):
        self.ommsystem = ommsystem
        self.config = config
        self.logger = logger

        self.topology = self.ommsystem.topology
        self.integrator = self.ommsystem.integrator

        nodefile = self.config.get("NODEFILE")
        assert nodefile, "NODEFILE needs to be specified"
        device = (
            open(nodefile, "r").readline().split(",")[1].strip().split(":")[1].strip()
        )

        if Platform.getNumPlatforms() == 1:
            if conda_prefix := os.environ.get("CONDA_PREFIX"):
                plugin_dir = os.path.join(conda_prefix, "lib", "plugins")
                Platform.loadPluginsFromDirectory(plugin_dir)

        platform_name = self.config.get("OPENMM_PLATFORM", "CUDA")

        platform = Platform.getPlatformByName(platform_name)
        properties = {}
        if platform_name in ("CUDA", "OpenCL", "HIP"):
            properties = {"DeviceIndex": device, "Precision": "mixed"}

        self.logger.info(f"Platform: {platform_name} {properties}")

        self.simulation = Simulation(
            self.topology, self.ommsystem.system, self.integrator, platform, properties
        )
        self.context = self.simulation.context
        self.context.setPositions(self.ommsystem.positions)
        self.context.setPeriodicBoxVectors(*self.ommsystem.boxvectors)

        # one preliminary energy evaluation seems to be required to init the energy routines
        self.context.getState(getEnergy=True).getPotentialEnergy()

        # load initial state/coordinates
        basename = self.config["BASENAME"]
        self.simulation.loadState(basename + "_0.xml")

        # replace parameters loaded from the initial xml file with the values in the system
        for key, value in self.ommsystem.cparams.items():
            self.context.setParameter(key, value)

        wdir = f"cntxt_{device}"
        if not os.path.isdir(wdir):
            os.mkdir(wdir)
        self.logfile = open(os.path.join(wdir, basename), "a+")
        nprnt = int(self.config.get("PRNT_FREQUENCY"))
        self.simulation.reporters.append(
            StateDataReporter(
                self.logfile, nprnt, step=True, temperature=True, speed=True
            )
        )

    def set_state(self, par):
        self.logger.debug("ommworker.set_state")

        self.integrator.setTemperature(par["temperature"])
        self.context.setParameter(
            self.ommsystem.parameter["temperature"], par["temperature"] / kelvin
        )

        atmforce = self.ommsystem.atmforce
        self.context.setParameter(atmforce.Lambda1(), par["lambda1"])
        self.context.setParameter(atmforce.Lambda2(), par["lambda2"])
        self.context.setParameter(atmforce.Alpha(), par["alpha"] * kilojoules_per_mole)
        self.context.setParameter(atmforce.Uh(), par["uh"] / kilojoules_per_mole)
        self.context.setParameter(atmforce.W0(), par["w0"] / kilojoules_per_mole)
        self.context.setParameter(atmforce.Direction(), par["atmdirection"])

    def set_posvel(self, positions, velocities):
        self.logger.debug("ommworker.set_posvel")
        self.context.setPositions(positions)
        self.context.setVelocities(velocities)

    def _perturbation_u0_u1(self):
        # get the perturbation energies u1, u0 without ATMForce.getPerturbationEnergy(),
        # which hangs on the CUDA platform when a PythonForce (our NNP) is wrapped in the
        # ATMForce: getPerturbationEnergy() re-evaluates the wrapped forces without
        # releasing the GIL, so the python force callback deadlocks. getState() does
        # release the GIL, so instead we read u0 and u1 off the ATMForce energy expression
        #   select(step(Direction),u0,u1) + ((L2-L1)/Alpha)*log(...) + L2*usc + W0
        # with Lambda1=Lambda2=0 and W0=0 only the leading select() is left: it is u0 for
        # Direction>0 and u1 for Direction<0 (Alpha=1 just avoids a 0/0 in the now
        # zero-weighted middle term). the sampling state is restored before returning.
        atm = self.ommsystem.atmforce
        ctx = self.context
        grp = {self.ommsystem.atmforcegroup}
        names = [atm.Lambda1(), atm.Lambda2(), atm.Alpha(), atm.Uh(), atm.W0(), atm.Direction()]
        saved = {n: ctx.getParameter(n) for n in names}
        try:
            ctx.setParameter(atm.Lambda1(), 0.0)
            ctx.setParameter(atm.Lambda2(), 0.0)
            ctx.setParameter(atm.Alpha(), 1.0)
            ctx.setParameter(atm.Uh(), 0.0)
            ctx.setParameter(atm.W0(), 0.0)
            ctx.setParameter(atm.Direction(), 1.0)
            u0 = ctx.getState(getEnergy=True, groups=grp).getPotentialEnergy()
            ctx.setParameter(atm.Direction(), -1.0)
            u1 = ctx.getState(getEnergy=True, groups=grp).getPotentialEnergy()
        finally:
            for n, v in saved.items():   # restore the sampling state
                ctx.setParameter(n, v)
        return u1, u0

    def get_energy(self, par):
        self.logger.debug("ommworker.get_energy")
        fgroups = {0, self.ommsystem.atmforcegroup}
        state = self.context.getState(getEnergy=True, groups=fgroups)
        pot = {}
        pot["potential_energy"] = state.getPotentialEnergy()
        if self.ommsystem.nnpforcegroup is not None:
            # PythonForce wrapped in ATMForce -> getPerturbationEnergy hangs on CUDA
            (u1, u0) = self._perturbation_u0_u1()
        else:
            (u1, u0, _) = self.ommsystem.atmforce.getPerturbationEnergy(self.context)
        umcore = (
            self.context.getParameter(self.ommsystem.atmforce.Umax())
            * kilojoules_per_mole
        )
        ubcore = (
            self.context.getParameter(self.ommsystem.atmforce.Ubcore())
            * kilojoules_per_mole
        )
        acore = self.context.getParameter(self.ommsystem.atmforce.Acore())
        if par["atmdirection"] > 0:
            pot["perturbation_energy"] = self.ommsystem.atm_utils.softCorePertE(
                u1 - u0, umcore, ubcore, acore
            )
        else:
            pot["perturbation_energy"] = self.ommsystem.atm_utils.softCorePertE(
                u0 - u1, umcore, ubcore, acore
            )
        if self.ommsystem.doMetaD:
            state = self.simulation.context.getState(
                getEnergy=True, groups={self.ommsystem.metaDforcegroup}
            )
            pot["bias_energy"] = state.getPotentialEnergy()
        else:
            pot["bias_energy"] = 0.0 * kilojoules_per_mole

        return pot

    def get_posvel(self):
        self.logger.debug("ommworker.get_posvel")
        state = self.context.getState(getPositions=True, getVelocities=True)
        return state.getPositions(asNumpy=True), state.getVelocities(asNumpy=True)

    def run(self, replica):
        assert replica.worker is self

        with Timer(self.logger.debug, "set replica state"):
            _, par = replica.get_state()
            self.set_state(par)
            self.set_posvel(replica.positions, replica.velocities)

        with Timer(self.logger.debug, "run replica"):
            nsteps = int(self.config["PRODUCTION_STEPS"])
            ntry = 5
            for _ in range(ntry):
                try:
                    self.simulation.step(nsteps)
                    break
                except OpenMMException as e:
                    self.logger.warning(f"Simulation failed: {e}")
            else:
                self.logger.error(f"Simulation failed {ntry} times!")
                raise RuntimeError(f"Simulation failed {ntry} times!")

        with Timer(self.logger.debug, "get replica state"):
            pos, vel = self.get_posvel()
            _, par = replica.get_state()
            pot = self.get_energy(par)

            replica.set_posvel(pos, vel)
            replica.set_energy(pot)
            replica.set_cycle(replica.get_cycle() + 1)
            replica.set_mdsteps(replica.get_mdsteps() + nsteps)
