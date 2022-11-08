import datetime
import logging
import os

import h5py
import numpy as np
import pandas as pd
import scipy.stats
import scipy.signal
import scipy.interpolate
from matplotlib import pyplot as plt

from vxtools.summarize import config

log = logging.getLogger(__name__)


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
        except:
            print('Failed for:')
            print(key, item, type(item))


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
    for rec_folder in os.listdir(root_path):
        current_path = f'{root_path}{rec_folder}/'

        if not os.path.isdir(current_path):
            continue

        log.info(f'Search {current_path}')

        missing = []
        if not 'suite2p' in os.listdir(current_path):
            log.warning(f'No suite2p folder found in {current_path}')
            missing.append('suite2p')

        if len(missing) > 0:
            # Try next level
            digest_folder(current_path, out_file)
            continue

        log.info(f'Load folder {current_path}')

        # Create debug folder
        debug_folder_path = os.path.join(current_path, config.DEBUG_FOLDER)
        if not os.path.exists(debug_folder_path):
            os.mkdir(debug_folder_path)

        # Create group
        rec_group = out_file.require_group('recording').require_group(rec_folder)

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
        scalar_cols = [col_name for col_name, series in stat_df.iteritems() if series.dtype != object]
        object_cols = [col_name for col_name, series in stat_df.iteritems() if series.dtype == object]

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
            downsample_by = 50  # TODO: TEMP
            mirror_position = np.squeeze(io_file[config.Y_MIRROR_SIGNAL])[::downsample_by]
            mirror_time = np.squeeze(io_file[f'{config.Y_MIRROR_SIGNAL}{config.TIME_POSTFIX}'])[::downsample_by]

            # Calculate frame timing
            frame_idcs, frame_times = calculate_ca_frame_times(mirror_position, mirror_time)

            # Plot debug info
            plot_y_mirror_debug_info(mirror_position, mirror_time, frame_idcs, current_path)

            log.info('Calculate interpolation for signal\'s record group IDs')
            record_group_ids = io_file['record_group_id'][::downsample_by].squeeze()
            record_group_ids_time = io_file['global_time'][::downsample_by].squeeze()
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

        log.info('Include display data')
        with h5py.File(os.path.join(current_path, config.DISPLAY_FILENAME), 'r') as disp_file:
            disp_data_group = rec_group.create_group('display_data')

            log.info('Add display phase data')
            disp_phase_group = disp_data_group.require_group('phases')
            for grp in disp_file.values():
                # Filter all non-phase members
                if not isinstance(grp, h5py.Group) or not grp.name.startswith('/phase'):
                    continue

                phase_grp = disp_phase_group.create_group(grp.name.replace('/phase', ''))
                phase_grp.attrs.update(grp.attrs)


if __name__ == '__main__':
    pass
