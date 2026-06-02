#!/usr/bin/env python3
"""
Scan a set of HDF5 snapshots, find particles that move from r<Rcut to r>Rcut,
and extract time-series for those particles into a CSV.

Usage:
  tools/find_crossers.py --snap-glob "snapshots/snapshot_*.hdf5" --rcut 10.0 --halo-center 100 100 100 \
    --ptype 1 --out-cross crossings.csv --out-ts particles_timeseries.csv

This script expects HDF5 snapshots with particle groups like PartType0..5
and datasets 'ParticleIDs','Coordinates','Velocities'. It will also try
to read a 'Potential' dataset if present.
"""
import argparse
import glob
import h5py
import numpy as np
import csv
import os


def read_header_time(f):
    # Try common locations for snapshot time
    for key in ('Time',):
        if key in f.attrs:
            return float(f.attrs['Time'])
    # Try Header group
    if 'Header' in f:
        h = f['Header']
        for k in ('Time','time'):
            if k in h.attrs:
                return float(h.attrs[k])
    # fallback: use file mtime
    return os.path.getmtime(f.filename)


def get_group_name(ptype):
    return f'PartType{ptype}'


def read_snapshot(path, ptype):
    with h5py.File(path, 'r') as f:
        t = read_header_time(f)
        gname = get_group_name(ptype)
        if gname not in f:
            raise RuntimeError(f"Particle group {gname} not found in {path}")
        g = f[gname]
        # common dataset names
        ids = None
        for name in ('ParticleIDs','ID','ParticleIDs/ID'):
            if name in g:
                ids = g[name][()]
                break
        if ids is None:
            # try generic 'ParticleIDs' at root
            if 'ParticleIDs' in f:
                ids = f['ParticleIDs'][()]
        if ids is None:
            raise RuntimeError('Could not find particle ids dataset')
        coords = None
        for name in ('Coordinates','Position','Positions'):
            if name in g:
                coords = g[name][()]
                break
        if coords is None:
            raise RuntimeError('Could not find coordinates dataset')
        vels = None
        for name in ('Velocities','Velocity','Vel'):
            if name in g:
                vels = g[name][()]
                break
        # potential optional
        pot = None
        for name in ('Potential','PotentialEnergy','Phi'):
            if name in g:
                pot = g[name][()]
                break
        return t, ids, coords, vels, pot


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--snap-glob', required=True)
    p.add_argument('--rcut', required=True, type=float)
    p.add_argument('--halo-center', nargs=3, type=float, default=(0.0,0.0,0.0))
    p.add_argument('--ptype', type=int, default=1, help='particle type for DM')
    p.add_argument('--alpha', type=float, default=0.1,
                   help='window width for near-Rcut selection; only particles with r in [Rcut-alpha, Rcut] at the previous snapshot are considered')
    p.add_argument('--out-cross', default='crossings.csv')
    p.add_argument('--out-ts', default='particles_timeseries.csv')
    p.add_argument('--max-cross', type=int, default=1000, help='max crossings to record')
    args = p.parse_args()

    files = sorted(glob.glob(args.snap_glob))
    if not files:
        raise SystemExit('No snapshots found for pattern')

    prev_inside = {}
    prev_r = {}
    crossings = []  # list of (id, snap_idx_prev, snap_idx_curr, time_prev, time_curr)

    print(f'Reading {len(files)} snapshots...')
    # First pass: detect outward crossings
    for i, fp in enumerate(files):
        t, ids, coords, vels, pot = read_snapshot(fp, args.ptype)
        cen = np.array(args.halo_center)
        rel = coords - cen
        r = np.sqrt((rel**2).sum(axis=1))
        inside = r < args.rcut
        # On first snapshot, seed prev_inside
        if i == 0:
            for idx, val in zip(ids, inside):
                prev_inside[int(idx)] = bool(val)
            prev_time = t
            continue
        # subsequent snapshots: compare
        for idx, cur_in, rr in zip(ids, inside, r):
            idx = int(idx)
            was_in = prev_inside.get(idx, False)
            was_r = prev_r.get(idx, None)
            consider = True
            if args.alpha > 0.0:
                # only consider if previous r was within [Rcut-alpha, Rcut]
                if was_r is None:
                    consider = False
                else:
                    consider = (was_r >= (args.rcut - args.alpha)) and (was_r < args.rcut)
            if was_in and (not bool(cur_in)) and consider:
                crossings.append((idx, i-1, i, prev_time, t))
                if len(crossings) >= args.max_cross:
                    break
            prev_inside[idx] = bool(cur_in)
            prev_r[idx] = float(rr)
        prev_time = t
        if len(crossings) >= args.max_cross:
            break

    print(f'Found {len(crossings)} crossing events')
    # write crossings
    with open(args.out_cross, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id','snap_prev_idx','snap_curr_idx','time_prev','time_curr'])
        for row in crossings:
            w.writerow(row)

    if not crossings:
        print('No crossings found; exiting')
        return

    cross_ids = sorted({c[0] for c in crossings})

    # Second pass: extract timeseries for cross_ids
    header = ['id','time','x','y','z','r','vx','vy','vz','v_mag','potential']
    out_f = open(args.out_ts, 'w', newline='')
    writer = csv.writer(out_f)
    writer.writerow(header)

    for fp in files:
        t, ids, coords, vels, pot = read_snapshot(fp, args.ptype)
        id_to_idx = {int(i): j for j,i in enumerate(ids)}
        cen = np.array(args.halo_center)
        for pid in cross_ids:
            j = id_to_idx.get(pid, None)
            if j is None:
                continue
            pos = coords[j]
            vel = vels[j]
            r = np.linalg.norm(pos - cen)
            vmag = np.linalg.norm(vel)
            pv = pot[j] if pot is not None else np.nan
            writer.writerow([pid, t, pos[0], pos[1], pos[2], r,
                             vel[0], vel[1], vel[2], vmag, pv])

    out_f.close()
    print('Wrote timeseries to', args.out_ts)


if __name__ == '__main__':
    main()
    
    
    
'''
I need you to write a postprocessing script to check a couple of things of swift simulation snapshots. I've developped a new implementation in swift where past a certain Rcut particles are massless and an analytical potential is applied. I need to test this is working correclty, therefore I want to observe particles that are close by Rcut: they start inside but end up crossing. I then want to plot position, velocity and potential of these particles.



This is how I imagine we could do this: we can determine the positions of particles, so we can loop through a couple of snapshots and check particles positions close to the Rcut, we can then know if a particle has exited Rcut, after that we can note the particles ID and plot its potential, position and velocity across multiple snapshots.



Here's an example of a script that uses pNbody to plot the position and potential of particles. So you can use as an example to know how to treat the snapshots:
'''
