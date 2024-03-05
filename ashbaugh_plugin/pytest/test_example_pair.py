# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

# Import the plugin module.
from hoomd import ashbaugh_plugin
# Import the hoomd Python package.
import hoomd

import itertools
import pytest
import numpy as np


# Python implementation of the pair force and energy.
def ashbaugh_force_and_energy(dx, epsilon, sigma, lam, r_cut, shift=False):

    dr = np.linalg.norm(dx)

    if dr >= r_cut:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64), 0.0

    sigma_6 = sigma * sigma * sigma * sigma * sigma * sigma
    lj1 = 4.0 * epsilon * sigma_6 * sigma_6
    lj2 = 4.0 * epsilon * sigma_6
    inv_dr_6 = 1.0 / (dr*dr*dr*dr*dr*dr)
    inv_rcut_6 = 1.0 / (r_cut*r_cut*r_cut*r_cut*r_cut*r_cut)
    f = (12.0 * lj1 * inv_dr_6 * inv_dr_6 - 6.0 * lj2 * inv_dr_6) / dr * np.array(dx, dtype=np.float64) / dr
    e = inv_dr_6 * (lj1 * inv_dr_6 - lj2) 
    e_shift = (1.0 - lam) * epsilon

    rmin = 2.0**(1/6) * sigma
    if dr <= rmin:
        e += e_shift
    else:
        f *= lam
        e *= lam

    if shift:
        e -=  inv_rcut_6 * (lj1 * inv_rcut_6 - lj2) 

    return f, e


# Build up list of parameters.
distances = np.linspace(0.1, 2.0, 3)
epsilon = [0.5, 2.0, 5.0]
sigmas = [0.5, 1.0, 1.5]
lams = [0.0, 0.5, 1.0]
# No need to test "xplor", as that is handled outside of the plugin impl.
modes = ["none", "shift"]

testdata = list(itertools.product(distances, epsilon, sigmas, lams, modes))


@pytest.mark.parametrize("distance, k, sigma, mode", testdata)
def test_force_and_energy_eval(simulation_factory,
                               two_particle_snapshot_factory, distance, epsilon,
                               sigma, lam, mode):

    # Build the simulation from the factory fixtures defined in
    # hoomd/conftest.py.
    sim = simulation_factory(two_particle_snapshot_factory(d=distance))

    # Setup integrator and force.
    integrator = hoomd.md.Integrator(dt=0.001)
    nve = hoomd.md.methods.NVE(hoomd.filter.All())

    cell = hoomd.md.nlist.Cell(buffer=0.4)
    example_pair: hoomd.md.pair.Pair = ashbaugh_plugin.pair.AshbaughPair(
        cell, default_r_cut=2.0, mode=mode)
    example_pair.params[("A", "A")] = dict(epsilon=epsilon, sigma=sigma, lam=lam)
    integrator.forces = [example_pair]
    integrator.methods = [nve]

    sim.operations.integrator = integrator

    sim.run(0)
    snap = sim.state.get_snapshot()
    if snap.communicator.rank == 0:
        vec_dist = snap.particles.position[1] - snap.particles.position[0]

        # Compute force and energy from Python
        shift = mode == "shift"
        f, e = ashbaugh_force_and_energy(vec_dist, epsilon, sigma, lam, 2.0, shift)
        e /= 2.0

    # Test that the forces and energies match that predicted by the Python
    # implementation.
    forces = example_pair.forces
    if snap.communicator.rank == 0:
        np.testing.assert_array_almost_equal(forces, [-f, f], decimal=6)

    energies = example_pair.energies
    if snap.communicator.rank == 0:
        np.testing.assert_array_almost_equal(energies, [e, e], decimal=6)
