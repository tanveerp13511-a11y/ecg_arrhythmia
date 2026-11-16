"""
Data Loader Module for MIT-BIH Arrhythmia Database.

This module provides functions to:
1. Download all 48 ECG recordings and their reference annotations from PhysioNet.
2. Verify the completeness and integrity of downloaded recordings (.hea, .dat, .atr).
3. Read ECG signals and clinical annotations using the WFDB library.
4. Generate a comprehensive dataset summary report (channels, beat counts, AAMI distribution).
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import wfdb

# Allow running as a standalone script or module
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import (
    ALL_48_RECORDS,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    MIT_TO_AAMI,
    NON_BEAT_SYMBOLS,
    SAMPLING_RATE,
)

PHYSIONET_BASE_URL = "https://physionet.org/files/mitdb/1.0.0"


def get_resilient_session(retries: int = 5, backoff_factor: float = 1.0) -> requests.Session:
    """
    Create a robust requests Session configured with automatic HTTP retries and exponential backoff.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "ECG-Arrhythmia-Research/1.0"})
    return session


def verify_record_files(record_id: str, data_dir: Path = RAW_DATA_DIR) -> Dict[str, bool]:
    """
    Verify that all three required PhysioNet files exist and are non-empty:
    - .hea: Header file (metadata, leads, gains, sampling rate)
    - .dat: Signal file (must be ~1.95 MB = 1,950,000 bytes for standard 30-min MIT-BIH)
    - .atr: Annotation file (cardiologist reference beat locations & symbols)

    Parameters
    ----------
    record_id : str
        The record identifier (e.g., '100', '201').
    data_dir : Path
        Directory where records are stored.

    Returns
    -------
    Dict[str, bool]
        Dictionary indicating existence and validity of each required file extension.
    """
    record_path = data_dir / record_id
    hea_file = Path(f"{record_path}.hea")
    dat_file = Path(f"{record_path}.dat")
    atr_file = Path(f"{record_path}.atr")

    hea_ok = hea_file.is_file() and hea_file.stat().st_size > 50
    dat_ok = dat_file.is_file() and dat_file.stat().st_size > 1_000_000  # Expected ~1.95MB
    atr_ok = atr_file.is_file() and atr_file.stat().st_size > 100

    complete = hea_ok and dat_ok and atr_ok
    return {
        "hea": hea_ok,
        "dat": dat_ok,
        "atr": atr_ok,
        "complete": complete,
    }


def verify_all_records(records: List[str] = ALL_48_RECORDS, data_dir: Path = RAW_DATA_DIR) -> Tuple[List[str], List[str]]:
    """
    Check the integrity of all specified records in the target directory.

    Returns
    -------
    Tuple[List[str], List[str]]
        (complete_records, missing_or_corrupted_records)
    """
    complete = []
    missing = []
    for rec in records:
        status = verify_record_files(rec, data_dir)
        if status["complete"]:
            complete.append(rec)
        else:
            missing.append(rec)
    return complete, missing


def download_single_file(url: str, dest_path: Path, session: requests.Session) -> bool:
    """
    Download a single file using streaming requests with a temporary file to prevent partial writes.
    """
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    try:
        with session.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
        if temp_path.exists():
            if dest_path.exists():
                dest_path.unlink()
            temp_path.rename(dest_path)
            return True
        return False
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise e


def download_mitdb(
    records: Optional[List[str]] = None,
    dest_dir: Path = RAW_DATA_DIR,
    force_redownload: bool = False
) -> Tuple[int, List[str]]:
    """
    Download recordings from the PhysioNet MIT-BIH Arrhythmia Database.
    
    Streams `.hea`, `.dat`, and `.atr` directly from PhysioNet.
    Verifies that WFDB can parse each record after downloading.

    Parameters
    ----------
    records : Optional[List[str]]
        List of records to download. Defaults to all 48 records.
    dest_dir : Path
        Destination directory to store the database files.
    force_redownload : bool
        If True, downloads even if the record files already exist locally.

    Returns
    -------
    Tuple[int, List[str]]
        (number_of_successful_records, list_of_failed_records)
    """
    if records is None:
        records = ALL_48_RECORDS

    dest_dir.mkdir(parents=True, exist_ok=True)
    session = get_resilient_session()
    failed_records = []
    
    rel_dest = dest_dir.name
    print(f"[DataLoader] Checking status of {len(records)} MIT-BIH recordings in {rel_dest}...")
    records_to_process = []
    for rec in records:
        if force_redownload or not verify_record_files(rec, dest_dir)["complete"]:
            records_to_process.append(rec)

    if not records_to_process:
        print("[DataLoader] All 48 recordings are already present and verified on disk.")
        return len(records), []

    print(f"[DataLoader] Downloading {len(records_to_process)} missing/incomplete recordings from PhysioNet...")
    
    extensions = ["hea", "dat", "atr"]
    for idx, rec in enumerate(records_to_process, 1):
        print(f"  --> [{idx}/{len(records_to_process)}] Processing record {rec}...", end=" ", flush=True)
        rec_success = True
        
        for ext in extensions:
            target_file = dest_dir / f"{rec}.{ext}"
            # Check if this specific file is already valid
            if not force_redownload and target_file.exists():
                if ext == "dat" and target_file.stat().st_size > 1_000_000:
                    continue
                elif ext in ["hea", "atr"] and target_file.stat().st_size > 50:
                    continue

            url = f"{PHYSIONET_BASE_URL}/{rec}.{ext}"
            download_ok = False
            for attempt in range(1, 4):
                try:
                    download_ok = download_single_file(url, target_file, session)
                    if download_ok:
                        break
                except Exception as e:
                    time.sleep(1.0 * attempt)
            
            if not download_ok:
                rec_success = False
                print(f"[FAILED on {ext}]", end=" ", flush=True)
                break

        # If files were downloaded, test parsing with wfdb
        if rec_success:
            try:
                rec_path = str(dest_dir / rec)
                _ = wfdb.rdrecord(rec_path, sampto=100)
                _ = wfdb.rdann(rec_path, "atr", sampto=100)
                print("OK")
            except Exception as e:
                print(f"[WFDB Parse Error: {e}]")
                rec_success = False

        if not rec_success:
            failed_records.append(rec)

    successful_count = len(records) - len(failed_records)
    print(f"\n[DataLoader] Download complete: {successful_count}/{len(records)} records verified.")
    if failed_records:
        print(f"[DataLoader] WARNING: Failed records: {failed_records}")

    return successful_count, failed_records


def load_record(
    record_id: str,
    data_dir: Path = RAW_DATA_DIR,
    primary_lead_only: bool = False
) -> Dict[str, Any]:
    """
    Load a single MIT-BIH ECG recording and its annotations.

    Parameters
    ----------
    record_id : str
        Record identifier (e.g. '100').
    data_dir : Path
        Directory where record files are located.
    primary_lead_only : bool
        If True, only extracts the primary lead (channel 0, typically MLII).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'record_id': string identifier
        - 'signal': numpy array of shape (N_samples, N_channels) or (N_samples,) in mV
        - 'lead_names': list of lead names (e.g. ['MLII', 'V5'])
        - 'fs': sampling frequency (360 Hz)
        - 'n_samples': total number of samples
        - 'duration_sec': duration in seconds (~1800 s = 30 min)
        - 'ann_samples': 1D array of sample indices for reference annotations
        - 'ann_symbols': list of beat annotation characters
        - 'aami_labels': list of mapped AAMI classes (or 'NON_BEAT' for rhythm/artifact markers)
    """
    record_path = str(data_dir / record_id)
    
    # 1. Read binary ECG signal and header metadata
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal  # Physical values in mV
    lead_names = record.sig_name
    fs = record.fs
    n_samples = record.sig_len
    duration_sec = n_samples / fs

    # 2. Read cardiologist reference annotations (.atr)
    annotation = wfdb.rdann(record_path, "atr")
    ann_samples = annotation.sample
    ann_symbols = annotation.symbol

    # 3. Map annotations to standard AAMI 5-class standard
    aami_labels = []
    for sym in ann_symbols:
        if sym in MIT_TO_AAMI:
            aami_labels.append(MIT_TO_AAMI[sym])
        elif sym in NON_BEAT_SYMBOLS:
            aami_labels.append("NON_BEAT")
        else:
            aami_labels.append("UNKNOWN")

    if primary_lead_only:
        signal = signal[:, 0]

    return {
        "record_id": record_id,
        "signal": signal,
        "lead_names": lead_names,
        "fs": fs,
        "n_samples": n_samples,
        "duration_sec": duration_sec,
        "ann_samples": ann_samples,
        "ann_symbols": ann_symbols,
        "aami_labels": aami_labels,
    }


def generate_dataset_summary(
    records: List[str] = ALL_48_RECORDS,
    data_dir: Path = RAW_DATA_DIR,
    save_output: bool = True
) -> pd.DataFrame:
    """
    Inspect all recordings in the dataset and generate a comprehensive clinical summary table.

    Summarizes:
    - Channel leads available (e.g. MLII, V1, V5)
    - Total duration in minutes
    - Beat counts per AAMI class (N, S, V, F, Q)
    - Total non-beat annotations

    Returns
    -------
    pd.DataFrame
        Record-wise summary DataFrame.
    """
    records_data = []

    print(f"[DataLoader] Analyzing all {len(records)} recordings...")
    for rec in records:
        status = verify_record_files(rec, data_dir)
        if not status["complete"]:
            print(f"[DataLoader] Record {rec} is missing or incomplete! Skipping summary for this record.")
            continue

        data = load_record(rec, data_dir=data_dir)
        labels = data["aami_labels"]
        
        # Count AAMI beat classes
        n_count = labels.count("N")
        s_count = labels.count("S")
        v_count = labels.count("V")
        f_count = labels.count("F")
        q_count = labels.count("Q")
        non_beat_count = labels.count("NON_BEAT")
        total_beats = n_count + s_count + v_count + f_count + q_count

        records_data.append({
            "record_id": rec,
            "leads": "/".join(data["lead_names"]),
            "sampling_rate": data["fs"],
            "duration_min": round(data["duration_sec"] / 60.0, 2),
            "total_annotations": len(labels),
            "total_heartbeats": total_beats,
            "N": n_count,
            "S": s_count,
            "V": v_count,
            "F": f_count,
            "Q": q_count,
            "Non_Beat": non_beat_count,
        })

    df = pd.DataFrame(records_data)
    
    if save_output and not df.empty:
        summary_csv = RESULTS_DIR / "dataset_summary.csv"
        df.to_csv(summary_csv, index=False)
        print(f"[DataLoader] Dataset summary saved to results/dataset_summary.csv")

    return df


if __name__ == "__main__":
    print("=" * 70)
    print("MIT-BIH Arrhythmia Database - Complete Acquisition & Verification")
    print("=" * 70)
    
    success, failed = download_mitdb(records=ALL_48_RECORDS)
    
    if success == len(ALL_48_RECORDS):
        print(f"\n[SUCCESS] All {len(ALL_48_RECORDS)} MIT-BIH recordings successfully verified.")
        df_summary = generate_dataset_summary(records=ALL_48_RECORDS)
        
        print("\n" + "=" * 70)
        print("DATABASE SUMMARY (Sample rows):")
        print("=" * 70)
        print(df_summary.head(10).to_string(index=False))
        
        print("\nTOTAL BEATS ACROSS ENTIRE DATABASE:")
        for col in ["N", "S", "V", "F", "Q", "Non_Beat", "total_heartbeats"]:
            print(f"  {col:>18}: {df_summary[col].sum():,}")
    else:
        print(f"\n[ERROR] Failed to acquire all records. Missing: {failed}")
        sys.exit(1)
