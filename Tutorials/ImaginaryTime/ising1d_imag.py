# Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import print_function
import json

L = 20
pars = {}

# defining the lattice
pars['Graph'] = {
    'Name': 'Hypercube',
    'L': L,
    'Dimension': 1,
    'Pbc': True,
}

# defining the hamiltonian
pars['Hamiltonian'] = {
    'Name': 'Ising',
    'h': 1.0,
}

sigmaxop = []
sites = []
for i in range(L):
    # \sum_i sigma^x(i)
    sigmaxop.append([[0, 1], [1, 0]])
    sites.append([i])

pars['Observables'] = {
    'Operators': sigmaxop,
    'ActingOn': sites,
    'Name': 'SigmaX',
}


# defining the GroundState method
# here we use exact imaginary time propagation
pars['GroundState'] = {
    'Method': 'ImaginaryTimePropagation',
    'StartTime': 0,
    'EndTime': 20,
    'TimeStep': 0.1,
    'OutputFile': "test-itp",
}

json_file = "ising1d_imag.json"
with open(json_file, 'w') as outfile:
    json.dump(pars, outfile)

print("\nGenerated Json input file: ", json_file)
print("\nNow you have two options to run NetKet: ")
print("\n1) Serial mode: netket " + json_file)
print("\n2) Parallel mode: mpirun -n N_proc netket " + json_file)
