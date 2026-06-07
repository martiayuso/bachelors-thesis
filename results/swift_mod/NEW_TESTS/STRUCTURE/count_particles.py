#!/usr/bin/env python3

import argparse
from pNbody import Nbody

# ========================================================
# ARGUMENT PARSER
# ========================================================

parser = argparse.ArgumentParser(
    description="Print number of particles in a snapshot"
)

parser.add_argument(
    "snapshot",
    type=str,
    help="Snapshot file"
)

args = parser.parse_args()

# ========================================================
# LOAD SNAPSHOT
# ========================================================

nb = Nbody(args.snapshot)

# ========================================================
# OUTPUT
# ========================================================

npart = len(nb.pos)

print(f"Total particles: {npart:,}")
