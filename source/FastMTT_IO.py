import numpy as np
import uproot
import time
import FastMTT
import multiprocessing as mp
import pandas as pd

# Globalna instancja dla każdego procesu
global_fMTT = None  

def init_worker(window=False):
    # One FastMTT object for each core
    global global_fMTT
    global_fMTT = FastMTT.FastMTT()
    # Configure likelihood components per worker
    global_fMTT.myLikelihood.enableLikelihoodComponents(window=window)

def process_batches_for_worker(args):
    worker_id, worker_batches = args
    # Each core processes its own batches
    global global_fMTT
    results = []
    for batch_data in worker_batches:
        measuredTau, METx, METy, covMET = batch_data
        global_fMTT.run(measuredTau, METx, METy, covMET)
        results.append((global_fMTT.mass, global_fMTT.pt, global_fMTT.tau1pt, global_fMTT.tau2pt))
    return results


def process_single_batch(args):
    batch_id, measuredTau, METx, METy, covMET = args
    global global_fMTT
    global_fMTT.run(measuredTau, METx, METy, covMET)
    return batch_id, (global_fMTT.mass, global_fMTT.pt, global_fMTT.tau1pt, global_fMTT.tau2pt)

def process_FastMTT(measuredTauLeptons, xMETs, yMETs, covMETs, batch_size=100, num_workers=4, window=False):
    num_total = len(measuredTauLeptons)
    num_batches = int(np.ceil(num_total / batch_size))

    if num_total == 0:
        return (
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    print(
        f"FastMTT: n_events={num_total}, batch_size={batch_size}, n_batches={num_batches}, "
        f"workers={num_workers}, window={window}"
    )

    # Build batch payloads (batch_id keeps original ordering)
    batch_args = []
    for batch_id, start in enumerate(range(0, num_total, batch_size)):
        stop = min(start + batch_size, num_total)
        batch_args.append(
            (
                batch_id,
                measuredTauLeptons[start:stop],
                xMETs[start:stop],
                yMETs[start:stop],
                covMETs[start:stop],
            )
        )
    
    start_time = time.time()
    
    # Multiprocessing: process per-batch so the parent can report progress.
    results_by_batch = [None] * num_batches
    processed = 0
    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(window,)) as pool:
        for batch_id, out in pool.imap_unordered(process_single_batch, batch_args, chunksize=1):
            results_by_batch[batch_id] = out
            # out[0] is mass array for this batch
            processed += len(out[0])
            print(f"  processed {processed}/{num_total} events (batch {batch_id + 1}/{num_batches})")

    # Concatenate in the correct original order
    mFast, ptFast, tau1pt, tau2pt = zip(*results_by_batch)
    
    end_time = time.time()
    print(f"Processing FastMTT took {end_time - start_time:.2f} seconds")
    
    return np.concatenate(mFast, axis=0), np.concatenate(ptFast, axis=0), np.concatenate(tau1pt, axis = 0), np.concatenate(tau2pt, axis = 0)

def read_root_file(file_path, tree_name, branches, entry_stop=None):
    with uproot.open(file_path) as file:
        tree = file[tree_name]
        
        data = tree.arrays(branches, library="np", entry_stop=entry_stop)
    
    return data

def load_root_events(file_path, tree_name, branches, entry_stop=None):
    data = read_root_file(file_path, tree_name, branches, entry_stop)

    shape = data["pt_1"].shape

    measuredTauLeptons = np.array([
        [np.full(shape, 3),  data["pt_1"], data["eta_1"], data["phi_1"], data["m_1"], data["dm_1"]],
        [np.full(shape, 1),  data["pt_2"], data["eta_2"], data["phi_2"], data["m_2"], data["dm_2"]]
    ])
    measuredTauLeptons = np.transpose(measuredTauLeptons, (2, 0, 1))

    covMET = np.array([
        [data["metcov00"], data["metcov01"]],
        [data["metcov01"], data["metcov11"]]
    ])
    covMET = np.transpose(covMET, (2, 0, 1))

    METx = data["met"] * np.cos(data["metphi"])
    METy = data["met"] * np.sin(data["metphi"])

    return {
        "measuredTauLeptons": measuredTauLeptons,
        "measuredMETx": METx,
        "measuredMETy": METy,
        "covMET": covMET
    }

def load_events_csv(csv_data):

    df = pd.read_csv(csv_data)

    event_df = df[['H.m', 'H.pt', 'METx', 'METy', 'covXX', 'covXY', 'covYY', 'dm1', 'pt1', 'eta1', 'phi1', 'mass1', 'type1', 'dm2', 'pt2', 'eta2', 'phi2', 'mass2', 'type2']].copy()

    Higgs_mass = event_df.pop('H.m').to_numpy()
    Higgs_pt = event_df.pop('H.pt').to_numpy()
    METx = event_df.pop('METx').to_numpy()
    METy = event_df.pop('METy').to_numpy()
    metcov = event_df[['covXX', 'covXY', 'covXY', 'covYY']].to_numpy()
    event_df.drop(columns=['covXX', 'covXY', 'covYY'], inplace=True)
    metcov = np.reshape(metcov, (len(metcov), 2, 2))

    print('pandas dataframe:\n', event_df)

    events = event_df.to_numpy()
    events = np.reshape(events, (len(events), 2, 6))

    return {"measuredTauLeptons": events, "measuredMETx": METx, "measuredMETy": METy, "covMET": metcov, "Higgs_mass": Higgs_mass, "Higgs_pt": Higgs_pt}

def load_input_file(file_path, tree_name=None, branches=None):
    if file_path.endswith(".root"):
        if tree_name is None or branches is None:
            raise ValueError("ROOT file should have tree_name and branches")
        return load_root_events(file_path, tree_name, branches)

    elif file_path.endswith(".csv"):
        return load_events_csv(file_path)

    else:
        raise ValueError(f"Unsupported file format: {file_path}")
