import argparse
import array
import os
import sys
from pathlib import Path

import numpy as np
import uproot

# Add source folder to path
fastmtt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source"))
if fastmtt_path not in sys.path:
    sys.path.insert(0, fastmtt_path)

from FastMTT_IO import load_input_file, process_FastMTT


def _first_existing(keys, preferred, fallbacks=()):
    for name in (preferred, *fallbacks):
        if name in keys:
            return name
    return None


def _load_root_inputs(file_path, tree_name="ntuple", entry_stop=None, channel="tt"):
    with uproot.open(file_path) as f:
        tree = f[tree_name]
        keys = set(tree.keys())

        # Prefer branch naming as in test_kinfit.py, but keep fallbacks for older ntuples
        b_pt1 = _first_existing(keys, "pt_1")
        b_eta1 = _first_existing(keys, "eta_1")
        b_phi1 = _first_existing(keys, "phi_1")
        b_m1 = _first_existing(keys, "mass_1", ("m_1",))

        b_pt2 = _first_existing(keys, "pt_2")
        b_eta2 = _first_existing(keys, "eta_2")
        b_phi2 = _first_existing(keys, "phi_2")
        b_m2 = _first_existing(keys, "mass_2", ("m_2",))

        b_met = _first_existing(keys, "met_pt", ("met",))
        b_metphi = _first_existing(keys, "met_phi", ("metphi",))

        b_covxx = _first_existing(keys, "met_covXX", ("metcov00",))
        b_covxy = _first_existing(keys, "met_covXY", ("metcov01",))
        b_covyy = _first_existing(keys, "met_covYY", ("metcov11",))

        # Optional decay modes/types
        b_dm1 = _first_existing(keys, "decayModePNet_1", ("decayMode_1", "dm_1", "dm1"))
        b_dm2 = _first_existing(keys, "decayModePNet_2", ("decayMode_2", "dm_2", "dm2"))

        b_type1 = _first_existing(keys, "decay_type_1", ("type_1", "type1"))
        b_type2 = _first_existing(keys, "decay_type_2", ("type_2", "type2"))

        required = [
            b_pt1,
            b_eta1,
            b_phi1,
            b_m1,
            b_pt2,
            b_eta2,
            b_phi2,
            b_m2,
            b_met,
            b_metphi,
            b_covxx,
            b_covxy,
            b_covyy,
        ]

        missing = [b for b in required if b is None]
        if missing:
            raise ValueError(
                "Missing required branches for FastMTT input. "
                "Expected names like in test_kinfit.py (pt_*, eta_*, phi_*, mass_*, met_pt, met_phi, met_covXX/XY/YY)."
            )

        branches = set(required)
        if b_dm1:
            branches.add(b_dm1)
        if b_dm2:
            branches.add(b_dm2)
        if b_type1:
            branches.add(b_type1)
        if b_type2:
            branches.add(b_type2)

        data = tree.arrays(sorted(branches), library="np", entry_stop=entry_stop)

    pt1 = data[b_pt1].astype(np.float64)
    eta1 = data[b_eta1].astype(np.float64)
    phi1 = data[b_phi1].astype(np.float64)
    m1 = data[b_m1].astype(np.float64)

    pt2 = data[b_pt2].astype(np.float64)
    eta2 = data[b_eta2].astype(np.float64)
    phi2 = data[b_phi2].astype(np.float64)
    m2 = data[b_m2].astype(np.float64)

    met = data[b_met].astype(np.float64)
    metphi = data[b_metphi].astype(np.float64)

    covxx = data[b_covxx].astype(np.float64)
    covxy = data[b_covxy].astype(np.float64)
    covyy = data[b_covyy].astype(np.float64)

    n = len(pt1)
    if not (len(pt2) == len(met) == len(covxx) == n):
        raise ValueError("Input branch lengths are inconsistent")

    if b_type1 and b_type2:
        type1 = data[b_type1].astype(np.int32)
        type2 = data[b_type2].astype(np.int32)
    else:
        # FastMTT convention (see FastMTT.py): 1=TauToHad, 2=TauToElec, 3=TauToMu
        if channel == "mt":
            type1 = np.full(n, 3, dtype=np.int32)
            type2 = np.full(n, 1, dtype=np.int32)
        elif channel == "tt":
            type1 = np.full(n, 1, dtype=np.int32)
            type2 = np.full(n, 1, dtype=np.int32)
        else:
            raise ValueError(f"Unsupported channel: {channel}")

    dm1 = data[b_dm1].astype(np.int32) if b_dm1 else np.full(n, -1, dtype=np.int32)
    dm2 = data[b_dm2].astype(np.int32) if b_dm2 else np.full(n, -1, dtype=np.int32)

    # For leptonic legs decay mode is irrelevant; keep -1
    dm1 = np.where(type1 == 1, dm1, -1).astype(np.int32)
    dm2 = np.where(type2 == 1, dm2, -1).astype(np.int32)

    measuredTauLeptons = np.stack(
        [
            np.stack([type1, pt1, eta1, phi1, m1, dm1], axis=1),
            np.stack([type2, pt2, eta2, phi2, m2, dm2], axis=1),
        ],
        axis=1,
    ).astype(np.float64)

    covMET = np.stack(
        [
            np.stack([covxx, covxy], axis=1),
            np.stack([covxy, covyy], axis=1),
        ],
        axis=1,
    ).astype(np.float64)

    metx = met * np.cos(metphi)
    mety = met * np.sin(metphi)

    return {
        "measuredTauLeptons": measuredTauLeptons,
        "measuredMETx": metx,
        "measuredMETy": mety,
        "covMET": covMET,
        "n": n,
    }


def _write_merged_root_overwrite_branches(
    input_root,
    tree_name,
    output_root,
    overwrite_columns,
    n_entries_to_write=None,
):
    import ROOT

    fin = ROOT.TFile.Open(str(input_root), "READ")
    if not fin or fin.IsZombie():
        raise RuntimeError(f"Failed to open input ROOT file: {input_root}")

    intree = fin.Get(tree_name)
    if not intree:
        raise RuntimeError(f"Tree '{tree_name}' not found in {input_root}")

    n_tree_entries = int(intree.GetEntries())
    if n_entries_to_write is None:
        n_entries = n_tree_entries
    else:
        n_entries = int(n_entries_to_write)
        if n_entries < 0:
            raise ValueError(f"n_entries_to_write must be >= 0, got {n_entries}")
        if n_entries > n_tree_entries:
            raise ValueError(
                f"n_entries_to_write ({n_entries}) exceeds tree entries ({n_tree_entries})"
            )

    for name, arr in overwrite_columns.items():
        if len(arr) != n_entries:
            raise ValueError(
                f"Length mismatch for column '{name}': {len(arr)} vs entries to write {n_entries}"
            )

    fout = ROOT.TFile(str(output_root), "RECREATE")
    fout.cd()

    # Remember existing leaf types before we disable branches for cloning
    leaf_typecode = {}
    for name in overwrite_columns.keys():
        leaf = intree.GetLeaf(name)
        if leaf:
            tname = leaf.GetTypeName()
            if tname in ("Double_t", "double"):
                leaf_typecode[name] = ("d", "D")
            elif tname in ("Float_t", "float"):
                leaf_typecode[name] = ("f", "F")
            else:
                # default for any other numeric type
                leaf_typecode[name] = ("f", "F")
        else:
            leaf_typecode[name] = ("f", "F")

    intree.SetBranchStatus("*", 1)
    for name in overwrite_columns.keys():
        if intree.GetBranch(name):
            intree.SetBranchStatus(name, 0)

    outtree = intree.CloneTree(0)

    buffers = {}
    for name in overwrite_columns.keys():
        arr_code, leaf_code = leaf_typecode.get(name, ("f", "F"))
        buffers[name] = array.array(arr_code, [0.0])
        outtree.Branch(name, buffers[name], f"{name}/{leaf_code}")

    for i in range(n_entries):
        intree.GetEntry(i)
        for name, arr in overwrite_columns.items():
            buffers[name][0] = float(arr[i])
        outtree.Fill()

    outtree.Write()
    fout.Close()
    fin.Close()


def run_fastmtt_root(input_root, tree_name, channel, batch_size, num_workers, entry_stop=None):
    parsed = _load_root_inputs(
        str(input_root), tree_name=tree_name, entry_stop=entry_stop, channel=channel
    )

    measuredTauLeptons = parsed["measuredTauLeptons"]
    metx = parsed["measuredMETx"]
    mety = parsed["measuredMETy"]
    covMET = parsed["covMET"]

    print("Input shapes:", measuredTauLeptons.shape, covMET.shape, metx.shape, mety.shape)

    # Run FastMTT in two configurations:
    # - window=False: without constraint
    # - window=True : with window constraint
    _m0, _pt0, tau1pt0, tau2pt0 = process_FastMTT(
        measuredTauLeptons,
        metx,
        mety,
        covMET,
        batch_size=batch_size,
        num_workers=num_workers,
        window=False,
    )
    _m1, _pt1, tau1pt1, tau2pt1 = process_FastMTT(
        measuredTauLeptons,
        metx,
        mety,
        covMET,
        batch_size=batch_size,
        num_workers=num_workers,
        window=True,
    )

    out_cols = {
        "FastMTT_pt_1": np.asarray(tau1pt0, dtype=np.float32),
        "FastMTT_pt_2": np.asarray(tau2pt0, dtype=np.float32),
        "FastMTT_pt_1_constraint": np.asarray(tau1pt1, dtype=np.float32),
        "FastMTT_pt_2_constraint": np.asarray(tau2pt1, dtype=np.float32),
    }

    output_root = Path.cwd() / "merged.root"
    _write_merged_root_overwrite_branches(
        input_root=input_root,
        tree_name=tree_name,
        output_root=output_root,
        overwrite_columns=out_cols,
        n_entries_to_write=parsed.get("n"),
    )

    print(f"Wrote: {output_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", help="Input file (.root or .csv)")
    parser.add_argument("--tree", dest="tree_name", default="ntuple", help="ROOT tree name (default: ntuple)")
    parser.add_argument("--channel", default="tt", choices=["mt", "tt"], help="Needed to infer leg types if not present")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--entry-stop", type=int, default=None, help="Optional uproot entry_stop for quick tests")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"

    if args.input is None:
        # Keep the old behaviour for CSV example runs
        file_path = data_dir / "Higgs.csv"
        print(f"No input provided. Using default CSV: {file_path}")
    else:
        file_path = Path(args.input)
        if not file_path.is_absolute() and file_path.suffix.lower() == ".csv":
            file_path = data_dir / file_path

    if str(file_path).endswith(".root"):
        run_fastmtt_root(
            input_root=file_path,
            tree_name=args.tree_name,
            channel=args.channel,
            batch_size=args.batch_size,
            num_workers=args.workers,
            entry_stop=args.entry_stop,
        )
        return

    # CSV path (no merged.root writing here)
    parsed = load_input_file(str(file_path))
    measuredTauLeptons = parsed["measuredTauLeptons"]
    metx = parsed["measuredMETx"]
    mety = parsed["measuredMETy"]
    covMET = parsed["covMET"]
    print("Input shapes:", measuredTauLeptons.shape, covMET.shape, metx.shape, mety.shape)
    mFast, ptFast, _tau1pt, _tau2pt = process_FastMTT(
        measuredTauLeptons,
        metx,
        mety,
        covMET,
        batch_size=args.batch_size,
        num_workers=args.workers,
        window=False,
    )
    print("---Output shape---\nmass: ", mFast.shape, "\npT: ", ptFast.shape)
    print("---Output means---\nmass: ", np.mean(mFast), "\npT: ", np.mean(ptFast))


if __name__ == "__main__":
    main()
