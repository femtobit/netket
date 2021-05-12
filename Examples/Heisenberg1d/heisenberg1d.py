# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import netket as nk
# import numpy as np
import jax
import flax

# 1D Lattice
L = 20

g = nk.graph.Chain(length=L, pbc=True)
# Lattice translation operations

# Hilbert space of spins on the graph
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

# Ising spin hamiltonian
ha = nk.operator.Heisenberg(hilbert=hi, graph=g)

# RBM Spin Machine
ma = nk.models.RBM(
    alpha=4,
    use_visible_bias=True,
    use_hidden_bias=False,
    dtype=float,
    kernel_init=nk.nn.initializers.normal(0.0001),
)

# Metropolis Local Sampling
sa = nk.sampler.MetropolisLocal(hi, n_chains=16)

# Optimizer
op = nk.optim.Sgd(learning_rate=0.01)
sr = None  # nk.optim.SR(0.1)

def local_loss_fn(logpsi_fn, w, s):
    log_predict = logpsi_fn({"params": w}, s)
    log_targets = 1.0
    return jax.numpy.abs(log_predict - log_targets)**2

# Variational monte carlo driver
gs = nk.VMC((local_loss_fn, hi), op, sa, ma, n_samples=120000, n_discard=100, sr=sr)


if Path("test.mpack").exists():
    with open("test.mpack", "rb") as f:
        gs.state.variables = flax.serialization.from_bytes(gs.state.variables, f.read())

print(gs.state._expect_fn(local_loss_fn))

# Print parameter structure
print(f"# variational parameters: {gs.state.n_parameters}")


# Run the optimization for 300 iterations
for i in gs.iter(n_steps=300):
    if nk.utils.mpi.rank == 0: print(i)
    jax.profiler.save_device_memory_profile(f"trace_memory_{nk.utils.mpi.rank}_{i}.prof")
