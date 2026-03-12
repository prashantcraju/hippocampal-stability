#!/usr/bin/env python3
"""
build_dataset.py - Build the complete annotated Aronov dataset.

Loads RESULTS_T.mat (755 titmouse units) and RESULTS_Z.mat (238 finch units),
classifies cells as excitatory/inhibitory using hierarchical clustering on
waveform features, and exports a comprehensive CSV.

Usage:
    python build_dataset.py /path/to/data/

Output:
    aronov_dataset.csv - 993 units with cell_type labels and all analysis results
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import fcluster, linkage
import scipy.io as sio
import scipy.io as sio
import struct
import zlib
import io
import sys
import warnings
import traceback

# ===================================================================
# MAT v5 binary helpers
# ===================================================================

def _read_tag(buf, pos, endian='<'):
    """Read a data element tag at `pos`. Returns (type, size, data_start, next_pos)."""
    raw0 = struct.unpack(f'{endian}I', buf[pos:pos+4])[0]
    raw1 = struct.unpack(f'{endian}I', buf[pos+4:pos+8])[0]
    # Small data element: type in lower 16 bits, size in upper 16 bits
    if (raw0 >> 16) != 0:
        t = raw0 & 0xFFFF
        s = raw0 >> 16
        return t, s, pos+4, pos+8
    # Normal element: 8-byte tag, then data padded to 8-byte boundary
    pad = (8 - raw1 % 8) % 8 if raw1 % 8 != 0 else 0
    return raw0, raw1, pos+8, pos+8+raw1+pad


def _parse_matrix_header(buf, pos, endian='<'):
    """
    Parse array flags, dims, and name from a miMATRIX starting at `pos`.
    `pos` should point to the first byte AFTER the miMATRIX tag.
    Returns (mclass, dims, name, offset_after_name).
    """
    # Flags subelement
    _, _, fd, fn = _read_tag(buf, pos, endian)
    mc = struct.unpack(f'{endian}I', buf[fd:fd+4])[0] & 0xFF

    # Dims subelement
    _, ds, dd, dn = _read_tag(buf, fn, endian)
    ndims = ds // 4
    dims = struct.unpack(f'{endian}{ndims}i', buf[dd:dd+ndims*4]) if ndims > 0 else ()

    # Name subelement
    _, ns, nd, nn = _read_tag(buf, dn, endian)
    name = buf[nd:nd+ns].decode('ascii', errors='replace').rstrip('\x00') if ns > 0 else ''

    return mc, dims, name, nn


# ===================================================================
# Recursive miMATRIX parser (pure Python, no scipy)
# ===================================================================

# MAT v5 class IDs
_mxCELL, _mxSTRUCT, _mxOBJECT, _mxCHAR = 1, 2, 3, 4
_mxDOUBLE, _mxSINGLE = 6, 7
_mxINT8, _mxUINT8, _mxINT16, _mxUINT16 = 8, 9, 10, 11
_mxINT32, _mxUINT32, _mxINT64, _mxUINT64 = 12, 13, 14, 15
_mxOPAQUE = 17

# miTYPE -> numpy dtype for numeric data elements
_MI_DTYPE = {
    1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
    5: np.int32, 6: np.uint32, 7: np.float32, 9: np.float64,
    12: np.int64, 13: np.uint64,
}

# mxCLASS -> numpy dtype for the target array
_MX_DTYPE = {
    _mxDOUBLE: np.float64, _mxSINGLE: np.float32,
    _mxINT8: np.int8, _mxUINT8: np.uint8,
    _mxINT16: np.int16, _mxUINT16: np.uint16,
    _mxINT32: np.int32, _mxUINT32: np.uint32,
    _mxINT64: np.int64, _mxUINT64: np.uint64,
}


def _parse_matrix(buf, pos, endian='<'):
    """
    Parse one miMATRIX element at `pos` in `buf`.
    Returns (value, next_pos) where value is a Python/numpy object.
    """
    t, s, dstart, nxt = _read_tag(buf, pos, endian)
    if t != 14:  # Not miMATRIX
        return None, nxt

    mat_end = pos + 8 + s  # end of this matrix element
    mc, dims, name, after_name = _parse_matrix_header(buf, pos+8, endian)

    n_elements = 1
    for d in dims:
        n_elements *= d

    # --- mxCELL: array of matrices ---
    if mc == _mxCELL:
        cells = []
        cell_pos = after_name
        for _ in range(n_elements):
            if cell_pos >= mat_end - 8:
                cells.append(None)
                continue
            val, cell_pos = _parse_matrix(buf, cell_pos, endian)
            cells.append(val)
        return cells, mat_end

    # --- mxCHAR: character array ---
    if mc == _mxCHAR:
        ct, cs, cd, _ = _read_tag(buf, after_name, endian)
        raw = buf[cd:cd+cs]
        if ct == 4:  # miUINT16
            text = raw.decode('utf-16-le', errors='replace')
        elif ct == 16:  # miUTF8
            text = raw.decode('utf-8', errors='replace')
        else:
            text = raw.decode('ascii', errors='replace')
        return text.rstrip('\x00'), mat_end

    # --- Numeric types ---
    if mc in _MX_DTYPE:
        ct, cs, cd, _ = _read_tag(buf, after_name, endian)
        if cs == 0:
            return np.array([], dtype=_MX_DTYPE[mc]), mat_end
        stored_dtype = _MI_DTYPE.get(ct, np.uint8)
        arr = np.frombuffer(buf[cd:cd+cs], dtype=stored_dtype).copy()
        target_dtype = _MX_DTYPE[mc]
        if stored_dtype != target_dtype:
            arr = arr.astype(target_dtype)
        if len(dims) > 1 and n_elements == len(arr):
            arr = arr.reshape(dims, order='F')  # MATLAB = Fortran order
        return arr, mat_end

    # --- mxSTRUCT ---
    if mc == _mxSTRUCT:
        # Field name length
        _, fnls, fnld, fnln = _read_tag(buf, after_name, endian)
        fnl = struct.unpack(f'{endian}I', buf[fnld:fnld+4])[0] if fnls >= 4 else 0
        # Field names
        _, fns, fnd, fnn = _read_tag(buf, fnln, endian)
        nfields = fns // fnl if fnl > 0 else 0
        field_names = []
        for fi in range(nfields):
            fn = buf[fnd+fi*fnl:fnd+(fi+1)*fnl].decode('ascii', errors='replace').rstrip('\x00')
            field_names.append(fn)
        # Parse field values (n_elements structs, each with nfields values)
        result = [{} for _ in range(n_elements)]
        field_pos = fnn
        for si in range(n_elements):
            for fn in field_names:
                if field_pos >= mat_end - 8:
                    result[si][fn] = None
                    continue
                val, field_pos = _parse_matrix(buf, field_pos, endian)
                result[si][fn] = val
        # Squeeze single struct
        if n_elements == 1:
            return result[0], mat_end
        return result, mat_end

    # --- mxOPAQUE ---
    if mc == _mxOPAQUE:
        # Read class name (miINT8 string after the matrix header)
        _, cns, cnd, cnn = _read_tag(buf, after_name, endian)
        class_name = buf[cnd:cnd+cns].decode('ascii', errors='replace').rstrip('\x00')
        # Parse remaining subelements (the actual payload)
        sub_pos = cnn
        children = []
        while sub_pos < mat_end - 8:
            val, sub_pos = _parse_matrix(buf, sub_pos, endian)
            if val is not None:
                children.append(val)
        return {'_class': class_name, '_children': children}, mat_end

    # --- Unknown class: skip ---
    return None, mat_end


# ===================================================================
# Subsystem extraction
# ===================================================================

def _extract_mcos_cells(filepath):
    """
    Extract the MCOS cell array from a MAT v5 file's subsystem.

    Returns the list of 11 cell values, or None.
    """
    filepath = Path(filepath)
    filesize = filepath.stat().st_size

    with open(filepath, 'rb') as f:
        header = f.read(128)
        endian = '<' if header[126:128] == b'IM' else '>'
        subsys_offset = struct.unpack(f'{endian}Q', header[116:124])[0]

        if subsys_offset == 0 or subsys_offset >= filesize:
            return None, endian

        # Read and decompress subsystem
        f.seek(subsys_offset)
        tag = f.read(8)
        dtype = struct.unpack(f'{endian}I', tag[:4])[0]
        size = struct.unpack(f'{endian}I', tag[4:8])[0]

        if dtype == 15:  # miCOMPRESSED
            decompressed = zlib.decompress(f.read(size))
        elif dtype == 14:
            decompressed = tag + f.read(size)
        else:
            return None, endian

    # Parse the mxUINT8 wrapper to get the payload
    wrapper_buf = decompressed
    t, s, dstart, nxt = _read_tag(wrapper_buf, 0, endian)
    if t != 14:
        return None, endian

    mc, dims, name, after_name = _parse_matrix_header(wrapper_buf, 8, endian)
    # Read the data subelement (the uint8 payload)
    dt, ds, dd, _ = _read_tag(wrapper_buf, after_name, endian)
    payload = wrapper_buf[dd:dd+ds]

    # Parse mini-MAT header
    if len(payload) < 8:
        return None, endian
    mini_endian = payload[2:4]
    if mini_endian == b'IM':
        endian = '<'
    elif mini_endian == b'MI':
        endian = '>'

    # Parse the struct/opaque/cell tree from the mini-MAT data
    data = payload[8:]
    val, _ = _parse_matrix(data, 0, endian)

    # Navigate: struct -> 'MCOS' field -> opaque -> children -> cell array
    if isinstance(val, dict) and 'MCOS' in val:
        mcos = val['MCOS']
        if isinstance(mcos, dict) and '_children' in mcos:
            for child in mcos['_children']:
                if isinstance(child, list) and len(child) >= 9:
                    return child, endian
    # Fallback: if val is already a list
    if isinstance(val, list) and len(val) >= 9:
        return val, endian

    return None, endian


def _mcos_cells_to_table(cells):
    """
    Convert the 11 MCOS cells to a DataFrame.

    Layout (from byte-level analysis):
        cells[0] : metadata bytes (skip)
        cells[1] : padding/garbage (skip)
        cells[2] : mxCELL(1, N_cols) -- the actual column data
        cells[3] : n_rows (scalar)
        cells[4] : n_rows repeated
        cells[5] : empty cell (row names, unused)
        cells[6] : n_cols (scalar)
        cells[7] : mxCELL(1, N_cols) of strings -- column names
        cells[8] : table properties struct (metadata)
        cells[9] : int32 array (skip)
        cells[10]: metadata cells (skip)
    """
    if cells is None or len(cells) < 8:
        return None

    # Extract column names from cells[7]
    col_names_cell = cells[7]
    if not isinstance(col_names_cell, list):
        return None

    col_names = []
    for item in col_names_cell:
        if isinstance(item, str):
            col_names.append(item)
        elif isinstance(item, np.ndarray) and item.dtype.kind in ('U', 'S'):
            col_names.append(str(item.flat[0]) if item.size > 0 else '')
        else:
            col_names.append(str(item) if item is not None else '')

    # Extract column data from cells[2]
    col_data_cell = cells[2]
    if not isinstance(col_data_cell, list):
        return None

    if len(col_names) != len(col_data_cell):
        warnings.warn(
            f"Column count mismatch: {len(col_names)} names vs "
            f"{len(col_data_cell)} data columns"
        )

    # Determine number of units (rows)
    n_units = None
    if isinstance(cells[3], np.ndarray) and cells[3].size == 1:
        n_units = int(cells[3].flat[0])
    elif isinstance(cells[3], (int, float)):
        n_units = int(cells[3])

    # Build the DataFrame
    data = {}
    for i, cname in enumerate(col_names):
        if i >= len(col_data_cell):
            break
        col = col_data_cell[i]
        data[cname] = _process_column(col, cname, n_units)

    return pd.DataFrame(data)


def _process_column(col, name, n_units):
    """Convert a single column's data into a list suitable for a DataFrame."""

    # Cell array (e.g., spike times, names, waveforms, subdivisions)
    if isinstance(col, list):
        result = []
        for item in col:
            if isinstance(item, np.ndarray) and item.dtype.kind == 'f':
                result.append(item.flatten())
            elif isinstance(item, str):
                result.append(item)
            elif isinstance(item, np.ndarray) and item.dtype.kind in ('U', 'S'):
                result.append(str(item.flat[0]) if item.size > 0 else '')
            elif item is None:
                result.append(None)
            else:
                result.append(item)
        return result

    # Numeric array (e.g., waveform_fs, AP, ML, DV, depth)
    if isinstance(col, np.ndarray):
        if col.dtype.kind in ('f', 'i', 'u'):
            return col.flatten().tolist()
        elif col.dtype == object:
            return [col.flat[i] for i in range(col.size)]

    # Scalar
    if isinstance(col, (int, float)):
        return [col] * (n_units or 1)

    return [col]


# ===================================================================
# Behavior parser (uses scipy -- B is always a struct, no issues)
# ===================================================================

def _parse_behavior(filepath):
    """Load and parse the B (behavior) variable."""
    mat = sio.loadmat(
        str(filepath), squeeze_me=False,
        struct_as_record=True, mat_dtype=True,
        variable_names=['B'],
    )
    if 'B' not in mat:
        return None

    B = mat['B']
    b = B[0, 0] if B.ndim >= 2 else B[0]
    result = {}
    for name in B.dtype.names:
        val = b[name]
        while isinstance(val, np.ndarray) and val.ndim > 1 and val.shape[0] == 1:
            val = val[0]
        if isinstance(val, np.ndarray) and val.ndim == 1 and val.shape[0] == 1:
            val = val[0]
        if name in ('fps', 'tstart'):
            result[name] = float(val if np.isscalar(val) else val.flat[0])
        else:
            result[name] = np.asarray(val).flatten()
    return result


# ===================================================================
# Public API
# ===================================================================

def load_session(filepath, verbose=False):
    """
    Load one recording session.

    Parameters
    ----------
    filepath : str or Path
        Path to an H*.mat session file.
    verbose : bool
        Print progress info.

    Returns
    -------
    dict with keys:
        behavior : dict (fps, xx, yy, head_angle, tt, ...)
        spikes   : DataFrame (st, name, waveform, waveform_fs, AP, ML, DV, depth, subdivision)
        bird     : str
        date     : str
        filename : str
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(filepath)

    parts = filepath.stem.split('_')
    result = {
        'filename': filepath.name,
        'bird': parts[0] if parts else filepath.stem,
        'date': parts[1] if len(parts) > 1 else '',
    }

    # --- Behavior ---
    result['behavior'] = _parse_behavior(filepath)

    # --- Spikes: try scipy first (works if S is a struct) ---
    try:
        mat = sio.loadmat(
            str(filepath), squeeze_me=False,
            struct_as_record=True, mat_dtype=True,
            variable_names=['S'],
        )
        if 'S' in mat:
            S = mat['S']
            if isinstance(S, np.ndarray) and S.dtype.names is not None:
                result['spikes'] = _parse_struct_spikes(S)
                return result
    except Exception:
        pass

    # --- Spikes: extract from MCOS subsystem ---
    if verbose:
        print(f"  {filepath.name}: extracting table from MCOS subsystem")

    cells, endian = _extract_mcos_cells(filepath)
    if cells is not None:
        df = _mcos_cells_to_table(cells)
        if df is not None and len(df) > 0:
            result['spikes'] = df
            return result

    # --- Failed ---
    warnings.warn(f"Could not extract spike data from {filepath.name}")
    result['spikes'] = None
    return result


def _parse_struct_spikes(S):
    """Parse S when it loaded as a structured array (already a struct)."""
    s = S[0, 0] if S.ndim >= 2 and S.shape == (1, 1) else S

    n_units = 1
    for name in S.dtype.names:
        val = s[name]
        while isinstance(val, np.ndarray) and val.ndim > 1 and val.shape[0] == 1:
            val = val[0]
        if isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] > 1:
            n_units = val.shape[0]
            break

    data = {}
    for name in S.dtype.names:
        val = s[name]
        while isinstance(val, np.ndarray) and val.ndim > 1 and val.shape[0] == 1:
            val = val[0]
        data[name] = _process_column(
            list(val.flatten()) if isinstance(val, np.ndarray) and val.dtype == object else val,
            name, n_units,
        )

    return pd.DataFrame(data)


def load_results(filepath):
    """Load a RESULTS_*.mat or swr_*.mat file (plain structs, no tables)."""
    mat = sio.loadmat(str(filepath), squeeze_me=True, struct_as_record=True)
    return {k: v for k, v in mat.items() if not k.startswith('__')}


def load_all_sessions(data_dir, file_pattern="H*.mat", verbose=True):
    """
    Load all session files from a directory.

    Skips RESULTS_* and swr_* files.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(file_pattern))
    files = [f for f in files
             if not f.name.startswith('RESULTS')
             and not f.name.startswith('swr')
             and not f.name.startswith('file_list')]

    if verbose:
        print(f"Found {len(files)} session files in {data_dir}")

    sessions = []
    errors = []
    for i, f in enumerate(files):
        try:
            sess = load_session(f, verbose=verbose)
            sessions.append(sess)
            spk = sess.get('spikes')
            n = len(spk) if isinstance(spk, pd.DataFrame) else '?'
            if verbose and (i % 25 == 0 or i == len(files) - 1 or i < 3):
                print(f"  [{i+1}/{len(files)}] {f.name}: {n} units")
        except Exception as e:
            errors.append((f.name, str(e)))
            if verbose:
                print(f"  [{i+1}/{len(files)}] {f.name}: ERROR -- {e}")

    if verbose:
        total = sum(
            len(s['spikes']) for s in sessions
            if isinstance(s.get('spikes'), pd.DataFrame)
        )
        print(f"\nLoaded {len(sessions)}/{len(files)} sessions, {total} total units")
        if errors:
            print(f"Errors: {len(errors)}")

    return sessions


def sessions_to_dataframe(sessions):
    """Combine all sessions into one DataFrame (one row per unit)."""
    frames = []
    for s in sessions:
        spk = s.get('spikes')
        if isinstance(spk, pd.DataFrame) and len(spk) > 0:
            df = spk.copy()
            df['bird'] = s['bird']
            df['session_date'] = s['date']
            df['filename'] = s['filename']
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()



def load_results_table(filepath):
    """Load a RESULTS .mat file and return a DataFrame."""
    cells, endian = _extract_mcos_cells(filepath)
    if cells is None:
        return None
    return _mcos_cells_to_table(cells)


def classify_ei(df, species):
    """
    Classify cells as excitatory (E) or inhibitory (I) using Ward
    hierarchical clustering on standardized (spike_width, pp_ratio).

    This reproduces the paper's classification with ~99% accuracy
    (536/219 vs reported 538/217 for titmice).
    """
    sw = df['spike_width'].values * 1000  # convert to ms
    pp = df['pp_ratio'].values

    features = np.column_stack([sw, pp])
    feat_std = (features - features.mean(axis=0)) / features.std(axis=0)

    Z = linkage(feat_std, method='ward')
    labels = fcluster(Z, t=2, criterion='maxclust')

    # Cluster with lower mean spike_width = narrow-spiking = inhibitory
    mean_sw_1 = sw[labels == 1].mean()
    mean_sw_2 = sw[labels == 2].mean()

    if mean_sw_1 < mean_sw_2:
        cell_type = np.where(labels == 1, 'I', 'E')
    else:
        cell_type = np.where(labels == 2, 'I', 'E')

    n_e = np.sum(cell_type == 'E')
    n_i = np.sum(cell_type == 'I')
    print(f"  {species}: {n_e} excitatory + {n_i} inhibitory = {len(df)}")

    return cell_type


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')

    print("Loading RESULTS tables...")
    rt = load_results_table(data_dir / 'RESULTS_T.mat')
    rz = load_results_table(data_dir / 'RESULTS_Z.mat')

    if rt is None or rz is None:
        print("ERROR: Could not load RESULTS files")
        sys.exit(1)

    print(f"  RESULTS_T: {len(rt)} titmouse units")
    print(f"  RESULTS_Z: {len(rz)} zebra finch units")

    # Classify E/I
    print("\nClassifying cell types...")
    rt['cell_type'] = classify_ei(rt, "Titmouse")
    rz['cell_type'] = classify_ei(rz, "Zebra finch")

    # Add species label
    rt['species'] = 'titmouse'
    rz['species'] = 'zebra_finch'

    # RESULTS_Z lacks 'subdivision' column -- add as NaN
    if 'subdivision' not in rz.columns:
        rz['subdivision'] = np.nan

    # Combine
    df = pd.concat([rt, rz], ignore_index=True)

    # Add derived features
    df['spike_width_ms'] = df['spike_width'] * 1000

    # Compute spatial information significance
    # info_shuffle is a (200,) array of shuffled info values
    # A cell is spatially selective if info > 99th percentile of shuffle
    sig = []
    for _, row in df.iterrows():
        shuf = row.get('info_shuffle')
        info_val = row.get('info')
        if isinstance(shuf, np.ndarray) and shuf.size > 0 and not np.isnan(info_val):
            p = np.mean(shuf >= info_val)
            sig.append(p)
        else:
            sig.append(np.nan)
    df['info_pvalue'] = sig
    df['spatially_selective'] = df['info_pvalue'] < 0.01  # p < 0.01

    # Reorder columns for clarity
    id_cols = ['name', 'session', 'bird', 'species', 'cell_type', 'subdivision']
    waveform_cols = ['spike_width', 'spike_width_ms', 'pp_ratio', 'waveform']
    anatomy_cols = ['AP', 'ML', 'DV', 'depth']
    activity_cols = ['nspikes', 'rate', 'cv', 'cv2', 'duration_masked',
                     'distance', 'mean_speed']
    spatial_cols = ['info', 'info_pvalue', 'spatially_selective', 'info_shuffle',
                    'coverage', 'coverage_bias', 'xcorr_map', 'xcorr_map_shuffle',
                    'map', 'delay_info', 'delay_xcorr']

    all_cols = id_cols + waveform_cols + anatomy_cols + activity_cols + spatial_cols
    # Add any remaining columns not in our list
    remaining = [c for c in df.columns if c not in all_cols]
    final_cols = [c for c in all_cols if c in df.columns] + remaining
    df = df[final_cols]

    # Summary stats
    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"\n  Total units: {len(df)}")
    print(f"\n  By species:")
    for sp in df['species'].unique():
        sub = df[df['species'] == sp]
        n_e = (sub['cell_type'] == 'E').sum()
        n_i = (sub['cell_type'] == 'I').sum()
        n_sel = sub['spatially_selective'].sum()
        print(f"    {sp}: {len(sub)} units ({n_e} E + {n_i} I), "
              f"{n_sel} spatially selective")

    print(f"\n  By bird:")
    bird_stats = df.groupby(['bird', 'species']).agg(
        n_units=('name', 'count'),
        n_E=('cell_type', lambda x: (x == 'E').sum()),
        n_I=('cell_type', lambda x: (x == 'I').sum()),
        n_spatial=('spatially_selective', 'sum'),
        sessions=('session', 'nunique'),
    ).reset_index()
    for _, row in bird_stats.iterrows():
        print(f"    {row['bird']} ({row['species']}): {row['n_units']} units "
              f"({row['n_E']}E/{row['n_I']}I), "
              f"{int(row['n_spatial'])} spatial, {row['sessions']} sessions")

    print(f"\n  By subdivision (titmouse only):")
    tit = df[df['species'] == 'titmouse']
    for subdiv in tit['subdivision'].dropna().unique():
        sub = tit[tit['subdivision'] == subdiv]
        n_e = (sub['cell_type'] == 'E').sum()
        n_i = (sub['cell_type'] == 'I').sum()
        n_sel = sub['spatially_selective'].sum()
        print(f"    {subdiv}: {len(sub)} units ({n_e}E/{n_i}I), "
              f"{n_sel} spatially selective")

    print(f"\n  Waveform features:")
    for ct in ['E', 'I']:
        sub = df[df['cell_type'] == ct]
        print(f"    {ct}: spike_width={sub['spike_width_ms'].mean():.3f} +/- "
              f"{sub['spike_width_ms'].std():.3f} ms, "
              f"pp_ratio={sub['pp_ratio'].mean():.3f} +/- "
              f"{sub['pp_ratio'].std():.3f}")

    # Export CSV (exclude large array columns for a clean CSV)
    array_cols = ['waveform', 'info_shuffle', 'xcorr_map_shuffle',
                  'map', 'delay_info', 'delay_xcorr']
    csv_cols = [c for c in df.columns if c not in array_cols]
    df_csv = df[csv_cols].copy()

    outpath = data_dir / 'aronov_dataset.csv'
    df_csv.to_csv(outpath, index=False, float_format='%.6g')
    print(f"\n  Exported: {outpath}")
    print(f"  Columns in CSV: {list(df_csv.columns)}")
    print(f"  (Array columns excluded: {array_cols})")

    # Also save full dataset as pickle for Python use
    pkl_path = data_dir / 'aronov_dataset.pkl'
    df.to_pickle(pkl_path)
    print(f"  Full dataset (with arrays): {pkl_path}")

    return df


if __name__ == '__main__':
    df = main()