import numpy as np
import sys
import os
from pathlib import Path

# Add source folder to path
fastmtt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'source'))
if fastmtt_path not in sys.path:
    sys.path.insert(0, fastmtt_path)

from FastMTT_IO import load_input_file, process_FastMTT


def run_fastmtt(file_path):
    tree_name = "tree;1;1"
    branches = [
        'pt_1', 'eta_1', 'phi_1', 'm_1',
        'pt_2', 'eta_2', 'phi_2', 'm_2', 'dm_2',
        'met', 'metphi',
        'metcov00', 'metcov01', 'metcov11'
    ]

    # Load data
    parsed = load_input_file(str(file_path), tree_name=tree_name, branches=branches)

    measuredTauLeptons = parsed["measuredTauLeptons"]
    METx = parsed["measuredMETx"]
    METy = parsed["measuredMETy"]
    covMET = parsed["covMET"]

    print("Input shapes:", measuredTauLeptons.shape, covMET.shape, METx.shape, METy.shape)

    # Run FastMTT
    mFast, ptFast, tau1pt, tau2pt = process_FastMTT(
        measuredTauLeptons, METx, METy, covMET,
        batch_size=10,
        num_workers=8
    )

    print("---Output shape---\nmass: ", mFast.shape, "\npT: ", ptFast.shape)
    print("---Output means---\nmass: ", np.mean(mFast), "\npT: ", np.mean(ptFast))


def main():
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"

    if len(sys.argv) < 2:
        file_path = data_dir / "Higgs.csv" 
        print(f"No input provided. Using default: {file_path}")
    else:
        arg_path = Path(sys.argv[1])
        if not arg_path.is_absolute():
            file_path = data_dir / arg_path
        else:
            file_path = arg_path

    run_fastmtt(str(file_path))


if __name__ == "__main__":
    main()
