import numpy as np
import uproot
import time
import FastMTT
import multiprocessing as mp
import pandas as pd

# Globalna instancja dla każdego procesu
global_fMTT = None  

def init_worker():
    #One FastMTT object for each core
    global global_fMTT
    global_fMTT = FastMTT.FastMTT()

def process_batches_for_worker(args):
    worker_id, worker_batches = args
    #Each core processes its own batches
    global global_fMTT
    results = []
    for batch_data in worker_batches:
        measuredTau, METx, METy, covMET = batch_data
        global_fMTT.run(measuredTau, METx, METy, covMET)
        results.append((global_fMTT.mass, global_fMTT.pt, global_fMTT.tau1pt, global_fMTT.tau2pt))
    return results

def process_FastMTT(measuredTauLeptons, xMETs, yMETs, covMETs, batch_size=5_000, num_workers=4):
    num_total = len(measuredTauLeptons)
    num_batches = int(np.ceil(num_total / batch_size))
    
    # Split to cores
    worker_data_splits = np.array_split(range(num_total), num_workers)
    worker_batches = []
    
    for worker_id, worker_indices in enumerate(worker_data_splits):
        batches = [
            (measuredTauLeptons[idxs],
            xMETs[idxs],
            yMETs[idxs],
            covMETs[idxs])
            for idxs in np.array_split(worker_indices, int(np.ceil(len(worker_indices) / batch_size)))
        ]
        worker_batches.append((worker_id, batches))
    
    start_time = time.time()
    
    # Multiprocessing
    with mp.Pool(processes=num_workers, initializer=init_worker) as pool:
        results = pool.map(process_batches_for_worker, worker_batches)
    
    # Calculating results
    mFast, ptFast, tau1pt, tau2pt = zip(*[item for sublist in results for item in sublist])
    
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

    df = pd.read_csv(csv_data, nrows = 1000)

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
