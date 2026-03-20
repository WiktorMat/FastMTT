import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Add source folder to path
fastmtt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "source"))
if fastmtt_path not in sys.path:
    sys.path.insert(0, fastmtt_path)

from FastMTT_IO import process_FastMTT


def _first_existing(keys, preferred, fallbacks=()):
    for name in (preferred, *fallbacks):
        if name in keys:
            return name
    return None


def _resolution(pt_reco, pt_gen):
    pt_reco = np.asarray(pt_reco, dtype=np.float64)
    pt_gen = np.asarray(pt_gen, dtype=np.float64)
    mask = np.isfinite(pt_reco) & np.isfinite(pt_gen) & (pt_gen > 0)
    if not np.any(mask):
        return np.empty((0,), dtype=np.float64)
    return (pt_reco[mask] - pt_gen[mask]) / pt_gen[mask]


def plot_pt_resolution_hist(
    resolution,
    *,
    title,
    output_path,
    xmin=-1.0,
    xmax=1.0,
    bin_width=0.05,
):
    """Histogram of (pt_reco - pt_gen) / pt_gen."""

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise ImportError("Plotting requires matplotlib") from e

    res = np.asarray(resolution, dtype=np.float64)
    res = res[np.isfinite(res)]
    if res.size == 0:
        print(f"No points for histogram '{title}'; skipping.")
        return

    res = res[(res >= float(xmin)) & (res <= float(xmax))]
    if res.size == 0:
        print(f"No points in [{xmin}, {xmax}] for histogram '{title}'; skipping.")
        return

    mean = float(np.mean(res))
    std = float(np.std(res))
    n_points = int(res.size)

    bins = np.arange(float(xmin), float(xmax) + float(bin_width), float(bin_width))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    ax = plt.gca()
    ax.hist(res, bins=bins, color="blue", alpha=0.75, edgecolor="black", density=True)
    plt.xlabel(r"$(p_T^{reco} - p_T^{gen}) / p_T^{gen}$")
    plt.ylabel("Density")
    plt.title(title)

    stats_text = "\n".join(
        [
            rf"$N = {n_points}$",
            rf"$\mu = {mean:.4f}$",
            rf"$\sigma = {std:.4f}$",
        ]
    )
    ax.text(
        0.97,
        0.97,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.85),
    )

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved plot: {out}")


def _load_parquet_inputs(file_path, entry_stop=None, channel="tt"):
    try:
        import pandas as pd
    except Exception as e:
        raise ImportError(
            "Reading .parquet requires 'pandas' plus a parquet engine (recommended: 'pyarrow')."
        ) from e

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to read parquet file '{file_path}'. "
            "This typically requires 'pyarrow' (recommended) or 'fastparquet'."
        ) from e

    if entry_stop is not None:
        df = df.iloc[: int(entry_stop)]

    parsed = _parquet_df_to_fastmtt_inputs(df, channel=channel)
    parsed["_df"] = df
    return parsed


def _parquet_df_to_fastmtt_inputs(df, channel="tt"):
    cols = set(df.columns)

    b_pt1 = _first_existing(cols, "pt_1")
    b_eta1 = _first_existing(cols, "eta_1")
    b_phi1 = _first_existing(cols, "phi_1")
    b_m1 = _first_existing(cols, "mass_1", ("m_1",))

    b_pt2 = _first_existing(cols, "pt_2")
    b_eta2 = _first_existing(cols, "eta_2")
    b_phi2 = _first_existing(cols, "phi_2")
    b_m2 = _first_existing(cols, "mass_2", ("m_2",))

    b_met = _first_existing(cols, "met_pt", ("met",))
    b_metphi = _first_existing(cols, "met_phi", ("metphi",))

    b_covxx = _first_existing(cols, "met_covXX", ("metcov00",))
    b_covxy = _first_existing(cols, "met_covXY", ("metcov01",))
    b_covyy = _first_existing(cols, "met_covYY", ("metcov11",))

    b_dm1 = _first_existing(cols, "decayModePNet_1", ("decayMode_1", "dm_1", "dm1"))
    b_dm2 = _first_existing(cols, "decayModePNet_2", ("decayMode_2", "dm_2", "dm2"))

    b_type1 = _first_existing(cols, "decay_type_1", ("type_1", "type1"))
    b_type2 = _first_existing(cols, "decay_type_2", ("type_2", "type2"))

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
    if any(x is None for x in required):
        raise ValueError(
            "Missing required columns for FastMTT input in parquet. "
            "Expected names like pt_*, eta_*, phi_*, mass_*/m_*, met_pt/met, met_phi/metphi, met_covXX/XY/YY."
        )

    pt1 = df[b_pt1].to_numpy(dtype=np.float64)
    eta1 = df[b_eta1].to_numpy(dtype=np.float64)
    phi1 = df[b_phi1].to_numpy(dtype=np.float64)
    m1 = df[b_m1].to_numpy(dtype=np.float64)

    pt2 = df[b_pt2].to_numpy(dtype=np.float64)
    eta2 = df[b_eta2].to_numpy(dtype=np.float64)
    phi2 = df[b_phi2].to_numpy(dtype=np.float64)
    m2 = df[b_m2].to_numpy(dtype=np.float64)

    met = df[b_met].to_numpy(dtype=np.float64)
    metphi = df[b_metphi].to_numpy(dtype=np.float64)

    covxx = df[b_covxx].to_numpy(dtype=np.float64)
    covxy = df[b_covxy].to_numpy(dtype=np.float64)
    covyy = df[b_covyy].to_numpy(dtype=np.float64)

    n = len(pt1)

    if b_type1 and b_type2:
        type1 = df[b_type1].to_numpy(dtype=np.int32)
        type2 = df[b_type2].to_numpy(dtype=np.int32)
    else:
        if channel == "mt":
            type1 = np.full(n, 3, dtype=np.int32)
            type2 = np.full(n, 1, dtype=np.int32)
        elif channel == "tt":
            type1 = np.full(n, 1, dtype=np.int32)
            type2 = np.full(n, 1, dtype=np.int32)
        else:
            raise ValueError(f"Unsupported channel: {channel}")

    dm1 = df[b_dm1].to_numpy(dtype=np.int32) if b_dm1 else np.full(n, -1, dtype=np.int32)
    dm2 = df[b_dm2].to_numpy(dtype=np.int32) if b_dm2 else np.full(n, -1, dtype=np.int32)

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


def _iter_parquet_batches(parquet_path, batch_rows=200_000, columns=None):
    """Yield pandas DataFrames from a single parquet file in row batches.

    This avoids loading the full dataset into memory.
    Requires pyarrow.
    """

    try:
        import pyarrow.parquet as pq
    except Exception as e:
        raise ImportError(
            "Streaming .parquet requires 'pyarrow'. Install it or set --chunk-rows 0 to load whole file."
        ) from e

    parquet_path = Path(parquet_path)
    if not parquet_path.is_file():
        raise ValueError(f"Expected a parquet file path, got: {parquet_path}")

    pf = pq.ParquetFile(parquet_path)
    for record_batch in pf.iter_batches(batch_size=int(batch_rows), columns=columns):
        yield record_batch.to_pandas()


def run_fastmtt_parquet(
    input_parquet,
    channel,
    batch_size,
    num_workers,
    entry_stop=None,
    chunk_rows=200_000,
    window=False,
):
    output_parquet = Path.cwd() / "merged.parquet"

    # If chunk_rows <= 0, keep the old behaviour (load whole file)
    if chunk_rows is None or int(chunk_rows) <= 0:
        parsed = _load_parquet_inputs(str(input_parquet), entry_stop=entry_stop, channel=channel)

        measuredTauLeptons = parsed["measuredTauLeptons"]
        metx = parsed["measuredMETx"]
        mety = parsed["measuredMETy"]
        covMET = parsed["covMET"]

        print("Input shapes:", measuredTauLeptons.shape, covMET.shape, metx.shape, mety.shape)

        _m0, pt0, tau1pt0, tau2pt0 = process_FastMTT(
            measuredTauLeptons,
            metx,
            mety,
            covMET,
            batch_size=batch_size,
            num_workers=num_workers,
            window=bool(window),
        )

        df = parsed.get("_df")
        if df is None:
            raise RuntimeError("Internal error: parquet dataframe not available")

        df = df.copy()
        df["FastMTT_pt"] = np.asarray(pt0, dtype=np.float32)
        df["FastMTT_pt_1"] = np.asarray(tau1pt0, dtype=np.float32)
        df["FastMTT_pt_2"] = np.asarray(tau2pt0, dtype=np.float32)

        try:
            df.to_parquet(output_parquet, index=False)
        except Exception as e:
            raise RuntimeError(
                f"Failed to write parquet '{output_parquet}'. "
                "This typically requires 'pyarrow' (recommended) or 'fastparquet'."
            ) from e

        print(f"Wrote: {output_parquet}")

        # Required histograms: resolution using FastMTT leg pT vs genPart_pt
        missing_cols = [c for c in ("genPart_pt_1", "genPart_pt_2") if c not in df.columns]
        if missing_cols:
            print(f"Resolution plots skipped; missing columns: {missing_cols}")
        else:
            try:
                res1 = _resolution(pt_reco=tau1pt0, pt_gen=df["genPart_pt_1"].to_numpy())
                res2 = _resolution(pt_reco=tau2pt0, pt_gen=df["genPart_pt_2"].to_numpy())
                plot_pt_resolution_hist(
                    res1,
                    title="Resolution leg1: (FastMTT_pt_1 - genPart_pt_1) / genPart_pt_1",
                    output_path="images/fastMTT/resolution_leg1.png",
                )
                plot_pt_resolution_hist(
                    res2,
                    title="Resolution leg2: (FastMTT_pt_2 - genPart_pt_2) / genPart_pt_2",
                    output_path="images/fastMTT/resolution_leg2.png",
                )
            except Exception as e:
                print(f"Resolution plotting skipped (parquet): {e}")
        return

    # Streaming mode: read and write incrementally.
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as e:
        raise ImportError(
            "Streaming .parquet requires 'pyarrow'. Install it or set --chunk-rows 0."
        ) from e

    if output_parquet.exists():
        output_parquet.unlink()

    # For plotting without blowing memory: sample resolution values during streaming
    pf_in = pq.ParquetFile(Path(input_parquet))
    n_total_rows = pf_in.metadata.num_rows if pf_in.metadata is not None else None
    max_plot_points = 200_000
    if n_total_rows and n_total_rows > 0:
        p_keep = min(1.0, max_plot_points / float(n_total_rows))
    else:
        p_keep = 0.01

    sample_res1 = []
    sample_res2 = []

    writer = None
    processed_rows = 0

    for df_chunk in _iter_parquet_batches(input_parquet, batch_rows=int(chunk_rows)):
        if entry_stop is not None and processed_rows >= int(entry_stop):
            break

        if entry_stop is not None:
            remaining = int(entry_stop) - processed_rows
            if remaining <= 0:
                break
            if len(df_chunk) > remaining:
                df_chunk = df_chunk.iloc[:remaining]

        chunk_parsed = _parquet_df_to_fastmtt_inputs(df_chunk, channel=channel)
        measuredTauLeptons = chunk_parsed["measuredTauLeptons"]
        metx = chunk_parsed["measuredMETx"]
        mety = chunk_parsed["measuredMETy"]
        covMET = chunk_parsed["covMET"]

        print("Chunk shapes:", measuredTauLeptons.shape, covMET.shape, metx.shape, mety.shape)

        _m0, pt0, tau1pt0, tau2pt0 = process_FastMTT(
            measuredTauLeptons,
            metx,
            mety,
            covMET,
            batch_size=batch_size,
            num_workers=num_workers,
            window=bool(window),
        )

        df_out = df_chunk.copy()
        df_out["FastMTT_pt"] = np.asarray(pt0, dtype=np.float32)
        df_out["FastMTT_pt_1"] = np.asarray(tau1pt0, dtype=np.float32)
        df_out["FastMTT_pt_2"] = np.asarray(tau2pt0, dtype=np.float32)

        # Sample required resolution values using FastMTT leg pT (tau1pt0/tau2pt0)
        try:
            if "genPart_pt_1" in df_chunk.columns:
                res1 = _resolution(pt_reco=tau1pt0, pt_gen=df_chunk["genPart_pt_1"].to_numpy())
                keep1 = np.random.random(res1.size) < p_keep
                if np.any(keep1):
                    sample_res1.append(res1[keep1])
            if "genPart_pt_2" in df_chunk.columns:
                res2 = _resolution(pt_reco=tau2pt0, pt_gen=df_chunk["genPart_pt_2"].to_numpy())
                keep2 = np.random.random(res2.size) < p_keep
                if np.any(keep2):
                    sample_res2.append(res2[keep2])
        except Exception:
            pass

        table = pa.Table.from_pandas(df_out, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_parquet, table.schema)
        writer.write_table(table)

        processed_rows += len(df_chunk)
        print(f"  wrote {processed_rows} rows -> {output_parquet}")

    if writer is not None:
        writer.close()

    print(f"Wrote: {output_parquet}")

    # Finalize resolution histograms from samples
    try:
        if sample_res1:
            res1 = np.concatenate(sample_res1, axis=0)
            if res1.size > max_plot_points:
                idx = np.random.choice(res1.size, size=max_plot_points, replace=False)
                res1 = res1[idx]
            plot_pt_resolution_hist(
                res1,
                title="Resolution leg1: (FastMTT_pt_1 - genPart_pt_1) / genPart_pt_1",
                output_path="images/fastMTT/resolution_leg1.png",
            )
        else:
            print("Resolution leg1 skipped (streaming): missing genPart_pt_1 or no samples")

        if sample_res2:
            res2 = np.concatenate(sample_res2, axis=0)
            if res2.size > max_plot_points:
                idx = np.random.choice(res2.size, size=max_plot_points, replace=False)
                res2 = res2[idx]
            plot_pt_resolution_hist(
                res2,
                title="Resolution leg2: (FastMTT_pt_2 - genPart_pt_2) / genPart_pt_2",
                output_path="images/fastMTT/resolution_leg2.png",
            )
        else:
            print("Resolution leg2 skipped (streaming): missing genPart_pt_2 or no samples")
    except Exception as e:
        print(f"Resolution plotting skipped (parquet streaming): {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input file (.parquet)")
    parser.add_argument("--channel", default="tt", choices=["mt", "tt"], help="Needed to infer leg types if not present")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--entry-stop",
        type=int,
        default=None,
        help="Optional row stop for quick tests (parquet rows)",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=200_000,
        help="Parquet only: stream input in chunks of N rows (0 disables streaming)",
    )
    parser.add_argument(
        "--window",
        action="store_true",
        help="Enable FastMTT likelihood window components (enableLikelihoodComponents(window=True))",
    )
    args = parser.parse_args()

    file_path = Path(args.input)
    if file_path.suffix.lower() != ".parquet":
        raise ValueError(f"Only .parquet input is supported now, got: {file_path}")

    run_fastmtt_parquet(
        input_parquet=file_path,
        channel=args.channel,
        batch_size=args.batch_size,
        num_workers=args.workers,
        entry_stop=args.entry_stop,
        chunk_rows=args.chunk_rows,
        window=args.window,
    )


if __name__ == "__main__":
    main()
