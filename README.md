# FastMTT

Author: Wiktor Matyszkiewicz
Last update: 28.08.2025

Following repository contains a python implementation of the FastMTT algorithm, used to reconstruct invariant mass of di-tau system at high speed (around 1s for 1000 events on modern computer). For C++ version refer to: https://github.com/SVfit/ClassicSVfit/tree/fastMTT_2024.

Project structure:  
- **FastMTT.py** – high-level implementation of the reconstruction algorithm (grid scan, maximum likelihood selection, reconstruction of tau four-vectors).  
- **Likelihood.py** – implementation of the likelihood function, with different components: MET, tau-mass constraint, optionals.

## Installation

Clone the repository:
```
bash
git clone https://github.com/<your-username>/FastMTT.git
cd FastMTT
```

No special installation is required apart from standard Python libraries (see [requirements.txt](./requirements.txt)).

## Usage

Basic usage in Python:
```
import FastMTT

fMTT = FastMTT.FastMTT()
fMTT.run(measuredTauLeptons, measuredMETx, measuredMETy, covMET)

masses = fMTT.mass
```

Input and output should be contained in numpy arrays with the structure (N, ...), where N is the number of events. For inputs one should need:

1) measuredTauLeptons -- (N, 2, 6), with the structure (number_of_events, taon for reconstruction, kinematic parameters)

Kinematic parameters should be:

lepton[0]: decay_type:
1 - TauToHad
2 - TauToElec
3 - TauToMu

lepton[1]: pt
lepton[2]: eta
lepton[3]: phi
lepton[4]: mass
lepton[5]: hadron decay mode (-1 for non-hadrons)

2) measuredMETx -- (N,) array with the information of x component of reconstructed MET.
3) measuredMETy -- (N,) array with the information of y component of reconstructed MET.
4) covMET -- (N, 2, 2) array with the covariance matrix, containting information of transfer function (Gaussian) between true and reconstructed MET. Event-by-event information will work at best.

For the output one will obtain one array of the size (N,), containing estimated invariant masses.

## Quick Start

Minimal examples are contained in the batch_processing.py and visualisation.py:

```
python3 examples/batch_processing.py examples/example_data.csv
python3 examples/visualisation.py examples/example_data.csv
```

visualisation.py will produce example histograms of reconstructed mass and pt resolution, that will be stored in the ./images directory.

batch_processing.py will calculate batch of events using multiprocessing regime.

## Documentation

For full details on inputs, batching, additional options, and likelihood components, see [DOCUMENTATION.md](./DOCUMENTATION.md).
