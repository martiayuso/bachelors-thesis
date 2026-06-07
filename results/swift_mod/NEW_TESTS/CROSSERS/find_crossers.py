#!/usr/bin/env python3
"""
Scan a set of HDF5 snapshots, find particles that move from r<Rcut to r>Rcut,
and extract time-series for those particles into a CSV.

Usage:
  find_crossers.py --snap-glob "snap_deep/snapshot_*.hdf5" --rcut 10.0 --halo-center 100 100 100 \
    --ptype 1 --out-cross deep_crossings.csv --out-ts deep_particles_timeseries.csv

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
        masses = None
        for name in ('Masses', 'Mass', 'ParticleMass'):
            if name in g:
                masses = g[name][()]
                break
        return t, ids, coords, vels, pot, masses

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--snap-glob', required=True)
    p.add_argument('--rcut', required=True, type=float)
    p.add_argument('--halo-center', nargs=3, type=float, default=(0.0,0.0,0.0))
    p.add_argument('--ptype', type=int, default=1, help='particle type for DM')
    p.add_argument('--alpha', type=float, default=1,
                   help='window width for near-Rcut selection; only particles with r in [Rcut-alpha, Rcut] at the previous snapshot are considered')
    p.add_argument('--min-mass', type=float, default=1e-8,
                   help='minimum particle mass for selection')
    p.add_argument('--out-cross', default='crossings.csv')
    p.add_argument('--out-ts', default='particles_timeseries.csv')
    p.add_argument('--max-cross', type=int, default=100, help='max crossings to record')
    args = p.parse_args()

    files = sorted(glob.glob(args.snap_glob))
    if not files:
        raise SystemExit('No snapshots found for pattern')

    prev_inside = {}
    prev_r = {}
    prev_mass = {}

    already_crossed = set()
    crossings = []

    print(f'Reading {len(files)} snapshots...')

    # First pass: detect outward crossings
    for i, fp in enumerate(files):
        t, ids, coords, vels, pot, masses = read_snapshot(fp, args.ptype)

        cen = np.array(args.halo_center)
        rel = coords - cen
        r = np.sqrt((rel**2).sum(axis=1))
        inside = r < args.rcut

        m_arr = masses if masses is not None else np.full(len(ids), np.inf)

        if i == 0:
            for idx, val, rr, m in zip(ids, inside, r, m_arr):
                idx = int(idx)
                prev_inside[idx] = bool(val)
                prev_r[idx] = float(rr)
                prev_mass[idx] = float(m)
            prev_time = t
            continue

        for idx, cur_in, rr, m in zip(ids, inside, r, m_arr):
            idx = int(idx)

            if idx in already_crossed:
                continue

            was_in = prev_inside.get(idx, False)
            was_r = prev_r.get(idx, None)
            was_mass = prev_mass.get(idx, 0.0)

            consider = True
            if args.alpha > 0.0:
                if was_r is None:
                    consider = False
                else:
                    consider = (was_r >= (args.rcut - args.alpha)) and (was_r < args.rcut)

            mass_ok = (was_mass > args.min_mass) and (m > args.min_mass)

            if was_in and (not bool(cur_in)) and consider and mass_ok:
                crossings.append((idx, i-1, i, prev_time, t))
                already_crossed.add(idx)

                if len(crossings) >= args.max_cross:
                    break

            prev_inside[idx] = bool(cur_in)
            prev_r[idx] = float(rr)
            prev_mass[idx] = float(m)

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
        t, ids, coords, vels, pot, masses = read_snapshot(fp, args.ptype)

        id_to_idx = {int(i): j for j, i in enumerate(ids)}
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

            writer.writerow([pid, t,
                             pos[0], pos[1], pos[2],
                             r,
                             vel[0], vel[1], vel[2],
                             vmag,
                             pv])

    out_f.close()
    print('Wrote timeseries to', args.out_ts)


if __name__ == '__main__':
    main()
    
    
