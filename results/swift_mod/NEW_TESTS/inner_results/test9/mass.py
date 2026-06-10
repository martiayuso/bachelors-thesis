from pNbody import Nbody
import numpy as np

nb = Nbody("chunk_0100_0104.hdf5")

mass = nb.Mass(units="Msol")
mass = mass[mass > 0]

# find unique masses and their counts
unique_masses, counts = np.unique(mass, return_counts=True)

print(f"Total particles: {len(mass)}\n")

print("Mass values and particle counts:")
for m, c in zip(unique_masses, counts):
    print(f"  {m:.6e} Msol  ->  {c} particles")
