import numpy as np
import matplotlib.pyplot as plt
from pNbody import Nbody

import numpy as np
nb = Nbody("snap_50/centered/snapshot_0227.hdf5")

r1 = nb.rxyz()
phi1 = nb.pot


plt.figure(figsize=(6,4))
plt.scatter(r1, phi1, s=1, alpha=0.3)
plt.xlabel("Radius R")
plt.ylabel("Potential Φ")
plt.title("Particle Potential vs Radius")
plt.grid(True)
plt.show()


nb = Nbody("snap_50/centered/snapshot_0100.hdf5")

r2 = nb.rxyz()
phi2 = nb.pot


plt.figure(figsize=(6,4))
plt.scatter(r2, phi2, s=1, alpha=0.3)
plt.xlabel("Radius R")
plt.ylabel("Potential Φ")
plt.title("Particle Potential vs Radius")
plt.grid(True)
plt.show()




plt.figure(figsize=(6,4))

plt.scatter(r1, phi1, s=1, alpha=0.3, label="snap_50")
plt.scatter(r2, phi2, s=1, alpha=0.3, label="snap_70")

plt.xlabel("Radius R")
plt.ylabel("Potential Φ")
plt.title("Particle Potential vs Radius (Combined)")
plt.legend()
plt.grid(True)

plt.show()
