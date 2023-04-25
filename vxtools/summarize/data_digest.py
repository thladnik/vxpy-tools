import datetime
import hashlib
import logging
import os
from typing import Any, Dict
import yaml

import h5py
import numpy as np
import pandas as pd
import scipy.stats
import scipy.signal
import scipy.interpolate
import tifffile
from tqdm import tqdm
from matplotlib import pyplot as plt

from vxtools.summarize import config
from vxtools.summarize.structure import OPENMODE, open_summary

log = logging.getLogger(__name__)


def create_summary(folder_path: str, file_name: str = 'vxtools_summary.hdf5'):

    # Combine paths
    full_path = f'{folder_path}{file_name}'

    log.info(f'Create summary of all recordings in {folder_path} in file {file_name}')
    # Open file in create mode and digest all valid folders in base directory
    with open_summary(full_path, mode=OPENMODE.CREATE) as f:
        f.attrs['root_path'] = folder_path
        digest_folder(folder_path, f)


def unravel_dict(dict_data: dict, group: h5py.Group):
    for key, item in dict_data.items():
        try:
            if isinstance(item, np.ndarray):
                if item.squeeze().shape == (1,):
                    item = item[0]
                else:
                    group.create_dataset(key, data=item)
                    continue
            elif isinstance(item, dict):
                unravel_dict(item, group.require_group(key))
                continue

            elif isinstance(item, datetime.datetime):
                group.attrs[f'{key}_date'] = item.strftime('%Y-%m-%d')
                group.attrs[f'{key}_datetime'] = item.strftime('%Y-%m-%d-%H-%M-%S')
                continue

            group.attrs[key] = item

        except Exception as _:
            log.error(f'Failed to unpack data for key {key}, item {item}, type {type(item)}')


def load_metadata(folder_path: str) -> Dict[str, Any]:
    """Function searches for and returns metadata on a given folder path

    Function scans the `folder_path` for metadata yaml files (ending in `meta.yaml`)
    and returns a dictionary containing their contents
    """

    meta_files = [f for f in os.listdir(folder_path) if f.endswith('meta.yaml')]

    log.info(f'Found {len(meta_files)} metadata files in {folder_path}.')

    meta_data = {}
    for f in meta_files:
        with open(os.path.join(folder_path, f), 'r') as stream:
            try:
                meta_data.update(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                print(exc)

    return meta_data

def phase_correlation(ref: np.ndarray, im: np.ndarray):
    """Function to calculate phase correlation between two 2D datasets"""
    fft_ref = np.fft.fft2(ref)
    fft_im = np.fft.fft2(im)
    conj_b = np.ma.conjugate(fft_im)
    r = fft_ref*conj_b
    r /= np.absolute(r)
    return np.fft.ifft2(r).real

def calculate_ca_frame_times(mirror_position: np.ndarray, mirror_time: np.ndarray):

    peak_prominence = (mirror_position.max() - mirror_position.min()) / 4
    peak_idcs, _ = scipy.signal.find_peaks(mirror_position, prominence=peak_prominence)
    trough_idcs, _ = scipy.signal.find_peaks(-mirror_position, prominence=peak_prominence)

    # Find first trough
    first_peak = peak_idcs[0]
    first_trough = trough_idcs[trough_idcs < first_peak][-1]

    # Discard all before (and including) first trough
    trough_idcs = trough_idcs[first_trough < trough_idcs]
    frame_idcs = np.sort(np.concatenate([trough_idcs, peak_idcs]))

    # Get corresponding times
    frame_times = mirror_time[frame_idcs]

    return frame_idcs, frame_times


def plot_y_mirror_debug_info(mirror_position: np.ndarray, mirror_time: np.ndarray,
                             frame_idcs: np.ndarray, recording_path: str):

    # Plot frame time detection results
    fig_name = 'frame_timing_detection'
    fig, ax = plt.subplots(1, 3, figsize=(18, 4), num=fig_name)

    frame_num = 30

    markersize = 3.
    start_times = mirror_time < mirror_time[frame_idcs[frame_num]]
    ax[0].plot(mirror_time[start_times], mirror_position[start_times], color='blue')
    ax[0].plot(mirror_time[frame_idcs[:frame_num]], mirror_position[frame_idcs[:frame_num]], 'o', color='red', markersize=markersize)
    ax[0].set_xlim(mirror_time[0], mirror_time[frame_idcs[frame_num]])

    ax[1].hist(np.diff(mirror_time[frame_idcs]))

    end_times = mirror_time > mirror_time[frame_idcs[-frame_num]]
    ax[2].plot(mirror_time[end_times], mirror_position[end_times], color='blue')
    ax[2].plot(mirror_time[frame_idcs[-frame_num:]], mirror_position[frame_idcs[-frame_num:]], 'o', color='red', markersize=markersize)
    ax[2].set_xlim(mirror_time[frame_idcs[-frame_num]], mirror_time[-1])

    fig.tight_layout()
    plt.savefig(os.path.join(recording_path, config.DEBUG_FOLDER, f'{fig_name}.pdf'), format='pdf')
    plt.clf()


def digest_folder(root_path: str, out_file: h5py.File):

    log.info(f'Digest {root_path}')

    # Load zstack, if it exists
    zstack = None
    filenames = [f for f in os.listdir(root_path) if os.path.isfile(f'{root_path}{f}')]
    zstack_filenames = [f for f in filenames if 'zstack' in f]
    if len(zstack_filenames) > 0:

        if len(zstack_filenames) > 1:
            log.warning(f'Multiple zstack files found. Using first one.')

        # Load ZStack
        log.info(f'Load zstack from file {zstack_filenames[0]}')
        zstack = tifffile.imread(f'{root_path}{zstack_filenames[0]}')
    else:
        log.warning(f'No zstack found for root path {root_path}')

    # Load metadata
    root_meta = load_metadata(root_path)

    # Create root group and add data
    root_hash = hashlib.md5(root_path.encode()).hexdigest()
    root_group = out_file.require_group('folder_roots').require_group(root_hash)
    root_group.attrs['folder_path'] = root_path
    root_group.require_group('meta').attrs.update(root_meta)
    if zstack is not None:
        root_group.create_dataset('zstack', data=zstack)

    log.info('Go through potential recording subfolders')
    for rec_folder in os.listdir(root_path):
        current_path = f'{root_path}{rec_folder}/'

        if not os.path.isdir(current_path):
            continue

        log.info(f'Search {current_path}')

        # Look for suite2p folder
        missing = []
        if not 'suite2p' in os.listdir(current_path):
            log.warning(f'No suite2p folder found in {current_path}')
            missing.append('suite2p')

        # Try next level if there's no imaging data here
        if len(missing) > 0:
            digest_folder(current_path, out_file)
            continue

        log.info(f'Load folder {current_path}')

        # Create debug folder
        debug_folder_path = os.path.join(current_path, config.DEBUG_FOLDER)
        if not os.path.exists(debug_folder_path):
            os.mkdir(debug_folder_path)

        # Create group
        rec_group = out_file.require_group('recording').require_group(rec_folder)

        # Add hash of root_folder group for reference
        rec_group.attrs['folder_root_hash'] = root_hash

        # Add manually added metadata
        rec_meta = load_metadata(current_path)
        rec_group.require_group('meta').attrs.update({**root_meta, **rec_meta})

        # Expected recording folder format "YYYY-mm-dd_fishX_recY_nZ/pZ"
        rec_props = rec_folder.split('_')
        rec_date = int(rec_props[0].replace('-', ''))
        fish_id = int(rec_props[1].replace('fish', ''))
        rec_id = int(rec_props[2].replace('rec', ''))
        rec_depth = rec_props[3]

        # Add record props to  group
        rec_group.attrs['date'] = rec_date
        rec_group.attrs['fish_id'] = fish_id
        rec_group.attrs['rec_id'] = rec_id
        rec_group.attrs['rec_id'] = rec_id
        rec_group.attrs['rec_depth'] = rec_depth
        # Set relative path from summary file's root directory
        rec_group.attrs['rel_path'] = os.path.relpath(current_path, out_file.attrs['root_path'])

        # Load s2p processed data
        s2p_path = f'{current_path}suite2p/plane0/'
        signal = np.load(f'{s2p_path}F.npy', allow_pickle=True)
        # Fneu = np.load(f'{s2p_path}Fneu.npy', allow_pickle=True)  # Typically neuropile signals are discarded
        spks = np.load(f'{s2p_path}spks.npy', allow_pickle=True)
        roi_stats = np.load(f'{s2p_path}stat.npy', allow_pickle=True)
        ops = np.load(f'{s2p_path}ops.npy', allow_pickle=True).item()
        iscell = np.load(f'{s2p_path}iscell.npy', allow_pickle=True)

        # Add suite2p's analysis options
        log.info('Include s2p_ops')
        ops_group = rec_group.require_group('s2p_ops')
        unravel_dict(ops, ops_group)

        # Add suite2p's analysis ROI stats
        log.info('Include s2p_roi_stats')
        roi_stats_group = rec_group.require_group('s2p_roi_stats')
        roi_stats_group.create_dataset('cell_probability', data=iscell[:,1])

        # Import individual stat dicts into pandas DataFrame for easier handling
        stat_df = pd.DataFrame.from_records(roi_stats)
        # Sort in into scalar and object columns (anything non-scalar in pandas, is stored as generic object)
        scalar_cols = [col_name for col_name, series in stat_df.items() if series.dtype != object]
        object_cols = [col_name for col_name, series in stat_df.items() if series.dtype == object]

        # Create datasets for scalar values (scalar stat values are stored in a dataset of shape (roi num, )
        for col_name in scalar_cols:
            roi_stats_group.create_dataset(col_name, data=stat_df.loc[:,col_name].values)

        # Create roi groups to store each object column (typically variable-length arrays) as attribute
        for roi_id in range(signal.shape[0]):
            roi_grp = roi_stats_group.create_group(str(roi_id))
            for col_name in object_cols:
                roi_grp.attrs[col_name] = stat_df.loc[roi_id, col_name]

        # Use Y mirror information to calculate frame timing relative to visual stimulus/behavior/etc
        with h5py.File(os.path.join(current_path, config.IO_FILENAME), 'r') as io_file:
            log.info('Calculate frame timing of signal')
            try:
                mirror_position = np.squeeze(io_file[config.Y_MIRROR_SIGNAL])[:]
            except:
                # New version of vxpy has channel type prefix
                try:
                    mirror_position = np.squeeze(io_file[f'ai_{config.Y_MIRROR_SIGNAL}'])[:]
                except:
                    raise KeyError('Mirror signal key is not in io data file')
            try:
                mirror_time = np.squeeze(io_file[f'{config.Y_MIRROR_SIGNAL}{config.TIME_POSTFIX}'])[:]
            except:
                try:
                    mirror_time = np.squeeze(io_file[f'ai_{config.Y_MIRROR_SIGNAL}_time'])[:]
                except:
                    raise KeyError('Mirror signal time key is not in io data file')

            # Calculate frame timing
            frame_idcs, frame_times = calculate_ca_frame_times(mirror_position, mirror_time)

            # Plot debug info
            plot_y_mirror_debug_info(mirror_position, mirror_time, frame_idcs, current_path)

            log.info('Calculate interpolation for signal\'s record group IDs')
            try:
                record_group_ids = io_file['record_group_id'][:].squeeze()
                record_group_ids_time = io_file['global_time'][:].squeeze()
            except:
                try:
                    # New version of vxpy has double leading underscores for system side attributes
                    record_group_ids = io_file['__record_group_id'][:].squeeze()
                    record_group_ids_time = io_file['__time'][:].squeeze()
                except:
                    raise KeyError('Record group ID and/or global time not in io file')

            ca_rec_group_id_fun = scipy.interpolate.interp1d(record_group_ids_time, record_group_ids, kind='nearest')

        # Check if frame times and signal match
        if frame_times.shape[0] != signal.shape[1]:
            log.warning(f'Detected frame times\' length doesn\'t match frame count. '
                        f'Detected frame times ({frame_times.shape[0]}) / Frames ({signal.shape[1]})')

            # Shorten signal
            if frame_times.shape[0] < signal.shape[1]:
                signal = signal[:, :frame_times.shape[0]]
                log.warning('Truncated signal at end to resolve mismatch. Check debug output to verify')

            # Shorten frame times
            else:
                frame_times = frame_times[:signal.shape[1]]
                log.warning('Truncated detected frame times at end to resolve mismatch. Check debug output to verify')

        log.info('Calculate DFF and ZSCORE from intensities')
        dff = np.apply_along_axis(lambda d: (d - d.mean()) / d.mean(), 1, signal)
        zscore = scipy.stats.zscore(dff, axis=1)

        log.info('Interpolate record group IDs for signal based on valid frame timings')
        ca_rec_group_ids = ca_rec_group_id_fun(frame_times)

        # Add signal data to record group
        ca_data_group = rec_group.require_group('ca_data')
        ca_data_group.create_dataset('signal', data=signal)
        ca_data_group.create_dataset('signal_devonv', data=spks)
        ca_data_group.create_dataset('dff', data=dff)
        ca_data_group.create_dataset('zscore', data=zscore)
        ca_data_group.create_dataset('frame_times', data=frame_times)
        ca_data_group.create_dataset('record_group_ids', data=ca_rec_group_ids.astype(np.int64))

        # Fetch ROI reference dataset
        if 'roi_refs' not in out_file:
            out_file.create_dataset('roi_refs', shape=(0, 4), dtype=np.uint64, maxshape=(None, 4))
        roi_ref_dataset: h5py.Dataset = out_file['roi_refs']

        # Add new ROIs
        new_roi_num = signal.shape[0]
        new_roi_refs = np.array(new_roi_num * [[rec_date, fish_id, rec_id, 0]])
        new_roi_refs[:, -1] = range(new_roi_num)
        roi_ref_dataset.resize((roi_ref_dataset.shape[0] + new_roi_num, roi_ref_dataset.shape[1]))
        roi_ref_dataset[-new_roi_num:] = new_roi_refs

        # Save data for all display phases
        log.info('Include display data')
        with h5py.File(os.path.join(current_path, config.DISPLAY_FILENAME), 'r') as disp_file:
            disp_data_group = rec_group.create_group('display_data')

            # Get smallest phase_id
            smallest_phase_id = min([int(grp.name.replace('/phase', '')) for grp in disp_file.values()
                                     if isinstance(grp, h5py.Group) and grp.name.startswith('/phase')])

            log.info('Add display phase data')
            disp_phase_group = disp_data_group.require_group('phases')
            for grp in disp_file.values():
                # Filter all non-phase members
                if not isinstance(grp, h5py.Group) or not grp.name.startswith('/phase'):
                    continue

                # Save phase_id, for further ease of processing phase_id is set to start at 0
                #  (original phase ID is kept for reference to original)
                phase_id = int(grp.name.replace('/phase', ''))
                phase_grp = disp_phase_group.create_group(str(phase_id - smallest_phase_id))
                phase_grp.attrs.update({'__original_phase_id': phase_id})
                phase_grp.attrs.update({'__phase_id': phase_id - smallest_phase_id})

                # Add display phase group attributes
                # phase_grp.attrs.update(grp.attrs)
                # This would be simpler, but can cause issues
                # We need to go through all attributes individually
                #  and fix 0-dimension np.ndarrays,
                #  otherwise they may cause problems during pandas analysis
                for attr_name in grp.attrs:
                    attr = grp.attrs[attr_name]
                    # Remove extra dimensions
                    if isinstance(attr, np.ndarray):
                        attr = attr.squeeze()
                        # Convert
                        if attr.shape == ():
                            attr = float(attr)

                    # Write to output dictionary
                    phase_grp.attrs[attr_name] = attr

        log.info('Run zstack registration')
        # Get mean
        ref = ops_group['meanImg'][:]

        # Determine padding and make sure it is divisible by 2
        padding = ref.shape[0] / 4
        padding = int(padding // 2 * 2)

        # Pad reference on all sides
        ref_im = np.pad(ref, (padding // 2, padding // 2))

        # Calculate maximum correlations and pixel shifts
        corrs = []
        xy_shifts = []
        for im in tqdm(zstack):

            # Pad image
            image = np.pad(im, (0, padding))

            # Calculate phase correlation
            corrimg = phase_correlation(ref_im, image)

            # Get maximum phase correlation and corresponding x/y shifts
            maxcorr = corrimg.max()
            y, x = np.unravel_index(corrimg.argmax(), corrimg.shape)

            # Subtract padding from shifts
            x -= padding // 2
            y -= padding // 2

            # Add max correlation and corresponding shift
            corrs.append(maxcorr)
            xy_shifts.append([x, y])

        # Convert
        corrs = np.array(corrs)
        xy_shifts = np.array(xy_shifts)

        # Add registration
        zstack_group = rec_group.require_group('zstack_registration')
        # All results (debug)
        zstack_group.create_dataset('max_correlations', data=corrs)
        zstack_group.create_dataset('xyshifts', data=xy_shifts)
        # Add most likely layer information for quick access
        zlayer_index = corrs.argmax()
        zstack_group.attrs['zlayer_index'] = zlayer_index
        zstack_group.attrs['x_pixelshift'] = xy_shifts[zlayer_index, 0]
        zstack_group.attrs['y_pixelshift'] = xy_shifts[zlayer_index, 1]
        zstack_group.create_dataset('zlayer_image', data=zstack[zlayer_index])




if __name__ == '__main__':
    pass
