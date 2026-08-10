"""
torchmdnetpotential.py: Implements the TorchMD-Net potential function.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2021-2026 Stanford University and the Authors.
Authors: Peter Eastman
Contributors: Stephen Farr

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import math
import os
import openmm
from openmm import unit
from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory
from typing import Iterable, Optional
import numpy as np
import torch

# Hartree*Bohr in eV*Angstrom: Coulomb prefactor for q(e)*q(e)/r(Angstrom) -> eV.
# Used by the electrostatic-embedding short-range correction in _ComputeTorchMDNet.
_COULOMB_FACTOR_eV_ANG = 27.211386024367243 * 0.5291772105638411

# Atomic polarizabilities (Angstrom^3) for the induction/polarization term, keyed by
# atomic number (ANI-MBIS table; bohr^3 * 0.1481847113 -> Angstrom^3).
_BOHR3_TO_ANG3 = 0.1481847113
_POLARIZABILITIES_ANG3 = {1: 3.08, 6: 11.30, 7: 7.40, 8: 5.30, 9: 3.74, 16: 19.40, 17: 14.60}
_THOLE_A = 1.3   # Thole linear-damping parameter for the induction field


def _electrostatic_terms(ml_A, mm_A, q_ml, ml_ff, mm_q, pol, box_diag, pme_alpha, thole_a, epsilon):
    """Short-range ML<->MM Coulomb delta + Thole-damped induction (eV).
    Pure-torch so it can be torch.compile'd (the sum over MM atoms is otherwise a
    big eager op). Angstrom units; box_diag=diag of the box (huge for non-periodic)."""
    dr_vec = ml_A.unsqueeze(1) - mm_A.unsqueeze(0)            # [N_ml, N_mm, 3]
    dr_vec = dr_vec - box_diag * torch.round(dr_vec / box_diag)   # minimum image
    dr = torch.sqrt(torch.clamp((dr_vec ** 2).sum(-1), min=1e-7))
    rhat = dr_vec / dr.unsqueeze(-1)
    # (a) Coulomb delta (q_ML - q_MM)*q_env, PME direct-space erfc kernel
    qq = (q_ml - ml_ff).unsqueeze(1) * mm_q.unsqueeze(0)
    e_coul = (_COULOMB_FACTOR_eV_ANG * qq * torch.erfc(pme_alpha * dr) / dr).sum()
    # (b) Thole-damped induction of ML atoms in the MM field, screened by epsilon
    s = thole_a * (dr / pol.unsqueeze(1) ** (1.0/3.0))
    thole = 1.0 - (1.0 + s + 0.5*s**2) * torch.exp(-s)
    term1 = torch.erfc(pme_alpha * dr) / dr**2
    term2 = (2.0*pme_alpha/math.sqrt(math.pi)) * torch.exp(-(pme_alpha**2)*dr**2) / dr
    field = ((mm_q.unsqueeze(0) * (term1 + term2) * thole).unsqueeze(-1) * rhat).sum(1)
    e_ind = -0.5 * _COULOMB_FACTOR_eV_ANG / epsilon * (pol * (field**2).sum(-1)).sum()
    return e_coul, e_ind


class _NNPElectrostaticEnergy(torch.nn.Module):
    """Single module = ligand NNP intra-energy + short-range ML<->MM electrostatic
    (Coulomb delta + Thole induction), returning the TOTAL energy (kJ/mol).

    Combining the model and the electrostatic sum into ONE module and taking backward()
    on its scalar energy output keeps the whole forward+backward in a single CUDA graph.
    (Two separate compiled regions with a combined eager backward re-record the graph
    every step -- "static input data pointer changed" -- which is ~50 ms/step wasted.)

    forward(pos, box): pos = full-system positions (nm, requires_grad set by caller),
    box = box vectors in Angstrom. Returns (E_total, E_intra, E_coul, E_ind) in kJ/mol;
    caller does E_total.backward() so grad lands on all atoms (ligand = MLIP+elec,
    environment = elec)."""

    def __init__(self, model, numbers, batch, charge, indices, mm_indices,
                 ml_ff_charges, mm_charges, pol_ml, pme_alpha, thole_a, epsilon,
                 lengthScale, energyScale):
        super().__init__()
        self.model = model
        self.register_buffer("numbers", numbers)
        self.register_buffer("batch", batch)
        self.register_buffer("charge", charge)
        self.register_buffer("indices", indices)
        self.register_buffer("mm_indices", mm_indices)
        self.register_buffer("ml_ff_charges", ml_ff_charges)
        self.register_buffer("mm_charges", mm_charges)
        self.register_buffer("pol_ml", pol_ml)
        self.pme_alpha = float(pme_alpha)
        self.thole_a = float(thole_a)
        self.epsilon = float(epsilon)
        self.lengthScale = float(lengthScale)
        self.energyScale = float(energyScale)

    def forward(self, pos, box):
        ml = pos.index_select(0, self.indices) / self.lengthScale          # ligand, Angstrom
        out = self.model(z=self.numbers, pos=ml, batch=self.batch, q=self.charge, box=box)
        e_intra = out[0].sum()
        q_ml = out[2].reshape(-1)
        mm = pos.index_select(0, self.mm_indices) / self.lengthScale       # environment, Angstrom
        e_coul, e_ind = _electrostatic_terms(
            ml, mm, q_ml, self.ml_ff_charges, self.mm_charges, self.pol_ml,
            torch.diagonal(box), self.pme_alpha, self.thole_a, self.epsilon)
        # single scalar output (kJ/mol): cudagraph trees re-record if extra outputs
        # are kept alive across calls, so return only the total energy.
        return self.energyScale * (e_intra + e_coul + e_ind)


class TorchMDNetPotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates TorchMDNetPotentialImpl objects."""

    def createImpl(
        self, 
        name: str, 
        modelPath: Optional[str] = None,
        lengthScale: float = 0.1, # angstrom -> nm
        energyScale: float = 96.4916,  # eV -> kJ/mol
    ) -> MLPotentialImpl:
        return TorchMDNetPotentialImpl(name, modelPath, lengthScale, energyScale)

class TorchMDNetPotentialImpl(MLPotentialImpl):
    """This is the MLPotentialImpl implementing the TorchMDNet potential.

    The TorchMDNet potential is constructed using `torchmdnet` to build a PyTorch model,
    and then integrated into the OpenMM System using a TorchForce.  To use it, specify the model by name
    and provide the path to a model.

    >>> potential = MLPotential('torchmdnet', modelPath=<model_file_path>)

    The default energy and length scales assume a model is trained with positions in angstroms and energies in eV.
    If this is not the case you can specify the length and energy scales by passing the factors that convert the model
    distance to nm and the energy to kJ/mol, for example:

    >>> potential = MLPotential('torchmdnet', modelPath=<model_file_path>, 
                                lengthScale=0.1 # angstrom to nm, 
                                energyScale=4.184 # kcal/mol to kJ/mol)

    During system creation you can enable CUDA graphs for a speed-up for small molecules:

    >>>  system = potential.createSystem(pdb.topology, cudaGraphs=True)

    The default is to enable this for TensorNet models.

    You can also specify the molecule's total charge:

    >>>  system = potential.createSystem(pdb.topology, charge=0)

    Pretained AceFF models can be used directly:

    >>> potential = MLPotential('aceff-2.0')

    >>> potential = MLPotential('aceff-1.1')

    >>> potential = MLPotential('aceff-1.0')

    Coulomb cutoff behavior
    ------------------------
    The Coulomb cutoff in TorchMD-Net uses a reaction-field approximation. Applying it to a
    non-periodic system introduces errors, so by default the cutoff is only used when the
    system uses periodic boundary conditions.

    You can override this with the ``useCoulombCutoff`` argument if you know which behavior
    you want, for example:

    >>>  system = potential.createSystem(pdb.topology, useCoulombCutoff=False)

    """

    # (Repository ID, filename, long-range)
    KNOWN_MODELS = {
        'aceff-1.0': ('Acellera/AceFF-1.0', 'aceff_v1.0.ckpt', False),
        'aceff-1.1': ('Acellera/AceFF-1.1', 'aceff_v1.1.ckpt', False),
        'aceff-2.0': ('Acellera/AceFF-2.0', 'aceff_v2.0.ckpt', False),
    }

    def __init__(self, 
                 name: str,
                 modelPath: str,
                 lengthScale: float,
                 energyScale: float
    ) -> None:
        """
        Initialize the TorchMDNetPotentialImpl.

        Parameters
        ----------
        name : str
            The name of the model.
            'torchmdnet' for a local model file, or pretrained models are available: 'aceff-1.0' or 'aceff-1.1'. 
        modelPath : str, optional
            The path to the locally trained torchmdnet model if ``name`` is 'torchmdnet'.
        lengthScale : float
            The length conversion factor from the model units to nanometers. 
            If not specified the default is 0.1 which corresponds to a model in angstrom
        energyScale : float
            The energy conversion factor from the model units to kJ/mol.
            If not specified the default is 96.4916 which corresponds to a model in eV.
   
        """
        self.name = name
        self.modelPath = modelPath
        self.lengthScale = lengthScale
        self.energyScale = energyScale

    def addForces(self,
                  topology: openmm.app.Topology,
                  system: openmm.System,
                  atoms: Optional[Iterable[int]],
                  forceGroup: int,
                  **args):
        # Load the TorchMDNet model.
        try:
            import torchmdnet
            from torchmdnet.models.model import load_model
        except ImportError as e:
            raise ImportError(f"Failed to import torchmdnet please install from https://torchmd-net.readthedocs.io/en/latest/installation.html")
        import torch

        includedAtoms = list(topology.atoms())
        if atoms is not None:
            includedAtoms = [includedAtoms[i] for i in atoms]
        device = self._getTorchDevice(args)
        numbers = torch.tensor([atom.element.atomic_number for atom in includedAtoms], device=device, requires_grad=False)
        # charge may be a scalar (one molecule) or a per-ligand list (RBFE: two ligands)
        charge = torch.tensor(np.atleast_1d(args.get('charge', 0)), dtype=torch.float32, device=device, requires_grad=False)
        cutoff = 10*args.get('coulomb_cutoff', 1.2)
        if unit.is_quantity(cutoff):
            cutoff = cutoff.value_in_unit(unit.angstrom)

        if self.modelPath is not None:
            # a local path to a torchmdnet checkpoint was provided
            model_file_path = self.modelPath
        else:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as e:
                raise ImportError(f"Failed to import huggingface_hub please install from https://huggingface.co/docs/huggingface_hub/en/installation")

            if self.name in TorchMDNetPotentialImpl.KNOWN_MODELS:
                repo_id, filename, _ = TorchMDNetPotentialImpl.KNOWN_MODELS[self.name]
            else:
                raise ValueError(f'Model name {self.name} does not exist.')

            model_file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
            )

        periodic = (topology.getPeriodicBoxVectors() is not None) or system.usesPeriodicBoundaryConditions()
        use_coulomb_cutoff = args.get('useCoulombCutoff', periodic)
        model = load_model(
            model_file_path,
            derivative=False,
            remove_ref_energy = args.get('remove_ref_energy', True),
            max_num_neighbors = min(args.get('max_num_neighbors', 64), numbers.shape[0]),
            coulomb_cutoff = cutoff if use_coulomb_cutoff else None,
            static_shapes = True,
            check_errors = False
        ).to(device)
        for parameter in model.parameters():
            parameter.requires_grad = False
        batch = args.get('batch', None)
        if batch is None:
            batch = torch.zeros_like(numbers, requires_grad=False)
        else:
            batch = torch.tensor(batch, dtype=torch.long, device=device, requires_grad=False)
        if atoms is None:
            indices = None
        else:
            indices = np.array(atoms)

        # Electrostatic embedding: an *additive* short-range correction on top of the
        # (unchanged) mechanical mixed system produced by createMixedSystem. The classical
        # force field keeps the ligand<->environment Coulomb with the fixed FF ligand
        # charges q_MM; here we add the delta (q_ML - q_MM)*q_env using the model's
        # predicted partial charges q_ML, so the net ligand<->environment electrostatics
        # use q_ML. Needs the environment (MM) charges q_env, the ligand FF charges q_MM,
        # and the PME screening parameter (for the erfc direct-space kernel).
        # Our ML embedding choice (mechanical | electrostatic). Passed as `mlEmbedding`
        # to avoid clashing with openmm-ml's own `embedding` arg (which selects the MM-side
        # mixed-system surgery and stays 'mechanical' — our electrostatic term is additive).
        embedding = args.get('mlEmbedding', 'mechanical')
        if embedding not in ('mechanical', 'electrostatic'):
            raise ValueError(f"mlEmbedding must be 'mechanical' or 'electrostatic', got {embedding!r}")
        mm_indices = mm_charges = ml_ff_charges = pme_alpha = None
        if embedding == 'electrostatic':
            if atoms is None:
                raise ValueError("electrostatic embedding requires the ML atom subset (atoms)")
            # FF charges: use the original charges passed in (before the mechanical mixed-system
            # surgery) if available, otherwise read them from the system here.
            all_q = args.get('all_charges')
            if all_q is None:
                nb = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
                all_q = [nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                         for i in range(system.getNumParticles())]
            all_q = np.asarray(all_q, dtype=np.float32)
            ml_set = set(atoms)
            mm_indices = np.array([i for i in range(len(all_q)) if i not in ml_set])
            mm_charges = all_q[mm_indices]                        # q_env
            ml_ff_charges = all_q[np.asarray(list(atoms))]        # q_MM, in `atoms` order
            pme_alpha = args.get('pme_alpha')
            if pme_alpha is None:
                nb = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
                tol = nb.getEwaldErrorTolerance()
                alpha = (1.0 / nb.getCutoffDistance()) * np.sqrt(-np.log(2.0 * tol))
                pme_alpha = float(alpha.value_in_unit(unit.angstrom ** -1))

        # Create the PythonForce and add it to the System.

        compute = _ComputeTorchMDNet(model=model,
                                     numbers=numbers,
                                     charge=charge,
                                     batch=batch,
                                     lengthScale=self.lengthScale,
                                     energyScale=self.energyScale,
                                     indices=indices,
                                     periodic=periodic,
                                     embedding=embedding,
                                     mm_indices=mm_indices,
                                     mm_charges=mm_charges,
                                     ml_ff_charges=ml_ff_charges,
                                     pme_alpha=pme_alpha,
                                     epsilon=float(args.get('epsilon', 2.0)),
                                     n_atoms=system.getNumParticles())
        force = openmm.PythonForce(compute)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(periodic)
        system.addForce(force)

    def getMLLongRange(self) -> bool | None:
        if self.name in TorchMDNetPotentialImpl.KNOWN_MODELS:
            _, _, longRange = TorchMDNetPotentialImpl.KNOWN_MODELS[self.name]
            return longRange
        return None

class _ComputeTorchMDNet(object):
    def __init__(self, model, numbers, charge, batch, lengthScale, energyScale, indices, periodic,
                 embedding='mechanical', mm_indices=None, mm_charges=None, ml_ff_charges=None,
                 pme_alpha=None, epsilon=2.0, n_atoms=None):
        import torch
        self.model = model
        self.compiled_model = None
        self.numbers = numbers
        self.charge = charge
        self.batch = batch
        self.lengthScale = lengthScale
        self.energyScale = energyScale
        self.indices = indices
        self.periodic = periodic
        self.has_recompiled = False
        self.embedding = embedding
        if embedding == 'electrostatic':
            device = numbers.device
            mm_indices_t = torch.tensor(mm_indices, dtype=torch.long, device=device)
            mm_charges_t = torch.tensor(mm_charges, dtype=torch.float32, device=device)   # q_env
            ml_ff_t = torch.tensor(ml_ff_charges, dtype=torch.float32, device=device)     # q_MM
            if indices is None:
                indices_t = torch.arange(numbers.shape[0], dtype=torch.long, device=device)
            else:
                indices_t = torch.as_tensor(indices, dtype=torch.long, device=device)
            # per-ML-atom polarizability (Angstrom^3), looked up by atomic number
            pol = torch.zeros(128, dtype=torch.float32, device=device)
            for z, a3 in _POLARIZABILITIES_ANG3.items():
                pol[z] = a3 * _BOHR3_TO_ANG3
            pol_ml = pol[numbers.to(torch.long)]
            self._graph = None          # manual torch.cuda.CUDAGraph (captured at setup)
            self._static_pos = None
            self._static_box = None
            self._static_E = None
            self._n_atoms = int(n_atoms) if n_atoms is not None else (len(mm_indices) + len(indices))
            self._n_calls = 0
            self._print_every = 100   # print the MLIP / MLIP_elec split once per this many evals
            # Prepare the model for fast inference (precompute Zij_map etc.); match the
            # reference sequence: build the lookup on CPU, then move the model to device.
            try:
                self.model.to("cpu")
                self.model.representation_model.setup_for_inference(numbers.cpu(), batch.cpu())
                self.model.to(device)
            except Exception as e:
                print(f"[atm] setup_for_inference skipped: {e}", flush=True)
            # single module (model + electrostatic) -> one compiled CUDA graph
            self._energy_module = _NNPElectrostaticEnergy(
                self.model, numbers, batch, charge, indices_t, mm_indices_t,
                ml_ff_t, mm_charges_t, pol_ml, pme_alpha, float(_THOLE_A), epsilon,
                lengthScale, energyScale)
            # Capture the CUDA graph NOW (during system build, before any openmm Context
            # exists). Capturing lazily inside the PythonForce callback fails when that
            # callback runs nested inside the ATMForce's CUDA work
            # (cudaErrorStreamCaptureInvalidated). Dummy positions of the right shape are
            # fine -- the graph is captured for shapes; real positions are copied in at
            # replay. Requires a CUDA device (n_atoms known from the System).
            dummy_pos = torch.rand(int(n_atoms), 3, dtype=torch.float32, device=device) * 30.0
            dummy_box = torch.eye(3, dtype=torch.float32, device=device) * 50.0
            self._capture(dummy_pos, dummy_box)

    # The CUDA graph / compiled module / static tensors can't be pickled. OpenMM
    # serializes this force (via copy.copy -> XmlSerializer.clone) when set_atmforce
    # puts it inside the ATMForce, so exclude the un-picklable runtime state and
    # re-capture on the clone. The clone is built during set_atmforce (before any
    # openmm Context exists) so its capture is in a clean CUDA stream.
    def __getstate__(self):
        state = dict(self.__dict__)
        for k in ("_graph", "_compiled_module", "_static_pos", "_static_box", "_static_E"):
            state.pop(k, None)
        return state

    def __setstate__(self, state):
        import torch
        self.__dict__.update(state)
        self._graph = None
        self._compiled_module = None
        self._static_pos = None
        self._static_box = None
        self._static_E = None
        if self.embedding == 'electrostatic':
            device = self.numbers.device
            dummy_pos = torch.rand(int(self._n_atoms), 3, dtype=torch.float32, device=device) * 30.0
            dummy_box = torch.eye(3, dtype=torch.float32, device=device) * 50.0
            self._capture(dummy_pos, dummy_box)

    def __call__(self, state):
        if self.embedding == 'electrostatic':
            return self._call_electrostatic(state)
        # --- mechanical (stock) path: torch.compile'd model over the ligand only ---
        import torch
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        numAtoms = positions.shape[0]
        positions = torch.tensor(positions, dtype=torch.float32, device=self.numbers.device)
        if self.indices is not None:
            positions = positions[self.indices]
        positions.requires_grad_(True)
        if self.periodic:
            cell = torch.tensor(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer), dtype=torch.float32, device=self.numbers.device)/self.lengthScale
        else:
            cell = None
        if self.compiled_model is None:
            # The model can't be compiled until after it has been invoked once.
            energy = self.model(z=self.numbers, pos=positions/self.lengthScale, batch=self.batch, q=self.charge, box=cell)[0].sum()*self.energyScale
            self.compiled_model = torch.compile(self.model, backend="inductor", dynamic=False, fullgraph=True, mode="reduce-overhead")
        else:
            # reduce-overhead + fullgraph, no fallback: if it can't graph, it must raise.
            energy = self.compiled_model(z=self.numbers, pos=positions/self.lengthScale, batch=self.batch, q=self.charge, box=cell)[0].sum()*self.energyScale
        energy.backward()
        forces = (-positions.grad).detach().cpu().numpy()
        if self.indices is not None:
            f = np.zeros((numAtoms, 3), dtype=np.float32)
            f[self.indices] = forces
            forces = f
        return energy, forces

    def _capture(self, pos, box):
        """Capture the combined module's forward+backward into a CUDA graph with static
        input/grad buffers. torch.compile's cudagraph trees can't replay inside OpenMM's
        PythonForce (OpenMM's CUDA allocations between calls invalidate the pool -> it
        re-records every step). A manual graph uses a private pool + fixed buffers, so
        replay is immune to OpenMM's activity."""
        import torch
        self._static_pos = pos.detach().clone().requires_grad_(True)
        self._static_box = box.detach().clone()
        # one eager pass first: initialises lazy model attributes (e.g. Qeq.dim_size)
        # that torch.compile(fullgraph=True) would otherwise graph-break on.
        self._static_pos.grad = None
        self._energy_module(self._static_pos, self._static_box).backward()
        # inductor-fuse the module forward/backward (mode="default" -> fusion, NO
        # cudagraphs) so the electrostatic sum runs as a couple of fused kernels
        # (~6x vs eager). The manual CUDA graph below then captures those fused kernels.
        mod = torch.compile(self._energy_module, backend="inductor",
                            dynamic=False, fullgraph=True, mode="default")
        self._compiled_module = mod
        # warm up on a side stream (triggers compilation + inits lazy model state) before capture
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                self._static_pos.grad = None
                mod(self._static_pos, self._static_box).backward()
        torch.cuda.current_stream().wait_stream(s)
        self._static_pos.grad = None
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static_E = self._compiled_module(self._static_pos, self._static_box)
            self._static_E.backward()

    def _call_electrostatic(self, state):
        """Ligand MLIP intra-energy + short-range ML<->MM electrostatic (Coulomb delta +
        Thole induction), from a single module captured into a manual CUDA graph. Replay
        gives grad on all atoms: ligand -> MLIP + electrostatic force, environment ->
        electrostatic force."""
        import torch
        device = self.numbers.device
        pos_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        pos = torch.tensor(pos_nm, dtype=torch.float32, device=device)
        if self.periodic:
            box = torch.tensor(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer),
                               dtype=torch.float32, device=device) / self.lengthScale     # Angstrom
        else:
            box = torch.eye(3, dtype=torch.float32, device=device) * 1e30                 # no PBC wrap

        # graph was captured at setup (in __init__); here we only replay.
        # copy this step's inputs into the static buffers, replay, read outputs
        self._static_pos.data.copy_(pos)
        self._static_box.data.copy_(box)
        if os.environ.get("ATM_NO_CUDAGRAPH"):
            # fallback: run the compiled module eagerly (fusion, no CUDA-graph replay)
            self._static_pos.grad = None
            E = self._compiled_module(self._static_pos, self._static_box)
            E.backward()
            energy = float(E)
        else:
            self._static_pos.grad.zero_()
            self._graph.replay()
            energy = float(self._static_E)
        forces = (-self._static_pos.grad).detach().cpu().numpy()

        self._n_calls += 1
        if (self._n_calls - 1) % self._print_every == 0:
            print(f"[MLIP eval {self._n_calls}] E = {energy:.3f} kJ/mol", flush=True)
        return energy, forces


# Register this potential with openmm-ml (stock openmm-ml is unmodified; atm owns
# the TorchMD-NET potential + its electrostatic embedding). Importing atm registers it.
MLPotential.registerImplFactory("TorchMD-NET", TorchMDNetPotentialImplFactory())
