from __future__ import annotations
import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict, Any, Union, Callable

import h5py
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class OPENMODE(Enum):
    CREATE = 1
    INSPECT = 2
    ANALYZE = 3


def open_summary(file_path: str, mode: OPENMODE = OPENMODE.INSPECT) -> Union[h5py.File, None]:

    fmode = None
    fdriver = None  # Default driver (None)
    if mode == OPENMODE.CREATE:
        fmode = 'w'
    elif mode == OPENMODE.INSPECT:
        fmode = 'r'
    elif mode == OPENMODE.ANALYZE:
        fdriver = 'core'  # core driver profile will load whole dataset into memory for faster access
        fmode = 'r+'

    log.info(f'Open HDF5 file {file_path} ({fmode=}, {fdriver=})')

    return h5py.File(file_path, mode=fmode, driver=fdriver)


class Summary:

    def __init__(self, file_path: str, mode: OPENMODE = OPENMODE.INSPECT):
        self.file_path = file_path
        self.file_mode = mode

        # Open file container
        self.h5file = open_summary(self.file_path, self.file_mode)

        # Load recordings
        self._recordings: List[Recording] = []
        for grp in self.h5file['recording'].values():
            rec_folder = grp.name.split('/')[-1]
            date = grp.attrs['date']
            fish_id = grp.attrs['fish_id']
            rec_id = grp.attrs['rec_id']
            self._recordings.append(Recording(date, fish_id, rec_id, rec_folder, self))

        # Load ROIs
        self._rois: List[Roi] = []
        for date, fish_id, rec_id, roi_id in self.h5file['roi_refs'][:]:
            roi = Roi(date, fish_id, rec_id, roi_id, self)
            roi.initialize()
            self._rois.append(roi)

    def __repr__(self):
        return f'{Summary.__name__}(\'{self.file_path}\', mode={self.file_mode})'

    def __enter__(self):
        return self

    def recordings(self, date: int = None, fish_id: int = None, rec_id: int = None) -> List[Recording]:
        return [rec for rec in self._recordings
                if (date is None or date == rec.date)
                and (fish_id is None or fish_id == rec.fish_id)
                and (rec_id is None or rec_id == rec.rec_id)]

    def rois(self, date: int = None, fish_id: int = None, rec_id: int = None, roi_id: int = None) -> List[Roi]:
        return [roi for roi in self._rois
                if (date is None or date == roi.date)
                and (fish_id is None or fish_id == roi.fish_id)
                and (rec_id is None or rec_id == roi.rec_id)
                and (roi_id is None or roi_id == roi.roi_id)]

    def add_data(self, name: str, data: np.numeric) -> None:

        # Get custom user data group
        grp = self.h5file['/'].require_group('user_data')

        # Delete if existing
        if name in grp:
            del grp[name]

        # Apply function and save to dataset
        grp.create_dataset(name, data=data)

    def data_exists(self, name: str):
        return name in self.h5file['/'].require_group('user_data')

    def get_data(self, name: str) -> Union[Any, None]:

        grp = self.h5file['/'].require_group('user_data')

        if name not in grp:
            return None

        dset = grp[name]

        # Return scalar dataset
        if dset.shape == ():
            return dset[()]

        return dset[:]

    def close(self):
        self.h5file.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.h5file.close()


class Recording:

    def __init__(self, date: int, fish_id: int, rec_id: int, rec_folder: str, summary_file: Summary):

        self.date: int = date
        self.fish_id: int = fish_id
        self.rec_id: int = rec_id
        self.summary: Summary = summary_file
        self.h5group: h5py.Group = self.summary.h5file['recording'][rec_folder]

        # Dict which may contain a regressor for each simple parameter
        self._simple_regressors: Dict[str, np.ndarray] = {}
        # List of the start and end indices within the calcium recording
        self._simple_phase_index_intervals: List[Tuple[int, int]]
        # Dataframe containing all phase information
        self._phase_df: pd.DataFrame() = None

    def __repr__(self):
        return f'{self.__class__.__name__}(date={self.date}, fish_id={self.fish_id}, rec_id={self.rec_id})'

    @property
    def record_group_id(self):
        return self.h5group['ca_data']['record_group_ids'][:]

    @property
    def ca_times(self):
        return self.h5group['ca_data']['frame_times'][:]

    @property
    def image_dimensions(self) -> Tuple[int, int]:
        return self.h5group['s2p_ops']['meanImg'].shape

    def dff(self, roi_id: int):
        return self.h5group['ca_data']['dff'][roi_id]

    def signal(self, roi_id: int):
        return self.h5group['ca_data']['signal'][roi_id]

    def signal_devonc(self, roi_id: int):
        return self.h5group['ca_data']['signal_deconv'][roi_id]

    def zscore(self, roi_id: int):
        return self.h5group['ca_data']['zscore'][roi_id]

    def roi_stats(self, roi_id: int):
        return self.h5group['s2p_roi_stats'][str(roi_id)]

    def _create_display_phase_dataframe(self):
        rows = []
        for phase in self.h5group['display_data']['phases'].values():
            start_index = np.argmin(np.abs(self.ca_times - phase.attrs['start_time']))
            end_index = np.argmin(np.abs(self.ca_times - phase.attrs['end_time']))

            phase_data = {'ca_start_index': start_index, 'ca_end_index': end_index}
            phase_data.update(phase.attrs)

            rows.append(phase_data)

        return pd.DataFrame(rows)

    @property
    def phase_dataframe(self) -> pd.DataFrame:
        if self._phase_df is None:
            self._phase_df = self._create_display_phase_dataframe()

        return self._phase_df

    def add_recording_data(self, name: str, data: np.numeric) -> None:

        # Get custom user data group
        grp = self.h5group.require_group(f'user_recording_data')

        # Delete if existing
        if name in grp:
            del grp[name]

        # Apply function and save to dataset
        grp.create_dataset(name, data=data)

    def recording_data_exists(self, name: str):
        return name in self.h5group.require_group(f'user_recording_data')

    def get_recording_data(self, name: str) -> Union[Any, None]:

        grp = self.h5group.require_group(f'user_recording_data')

        if name not in grp:
            return None

        dset = grp[name]

        # Return scalar dataset
        if dset.shape == ():
            return dset[()]

        return dset[:]

    def add_roi_data(self, roi_id: int, name: str, data: Any) -> None:

        # Get ROI-specific group
        grp = self.h5group.require_group(f'user_roi_data/{roi_id}')

        # Delete if existing
        if name in grp:
            del grp[name]

        # Update attributes
        grp.create_dataset(name, data=data)

    def roi_data_exists(self, roi_id: int, name: str):
        return name in self.h5group.require_group(f'user_roi_data/{roi_id}')

    def get_roi_data(self, roi_id: int, name: str) -> Union[Any, None]:

        grp = self.h5group.require_group(f'user_roi_data/{roi_id}')

        if name not in grp:
            return None

        dset = grp[name]

        # Return scalar dataset
        if dset.shape == ():
            return dset[()]

        return dset[:]


@dataclass
class Roi:
    date: int
    fish_id: int
    rec_id: int
    roi_id: int
    summary: Summary

    def initialize(self):
        pass

    @property
    def shortname(self):
        return f'Roi({self.date=}, {self.fish_id}, {self.rec_id}, {self.roi_id})'

    @property
    def rec(self):
        return self.summary.recordings(self.date, self.fish_id, self.rec_id)[0]

    @property
    def stats(self):
        return self.rec.roi_stats(self.roi_id)

    @property
    def dff(self):
        return self.rec.dff(self.roi_id)

    @property
    def signal(self):
        return self.rec.signal(self.roi_id)

    @property
    def signal_devonc(self):
        return self.rec.signal_devonc(self.roi_id)

    @property
    def zscore(self):
        return self.rec.zscore(self.roi_id)

    @property
    def times(self):
        return self.rec.ca_times

    @property
    def record_group_id(self):
        return self.rec.record_group_id

    @property
    def binary_mask(self):
        x, y = self.rec.image_dimensions
        mask = np.zeros((x, y), dtype=bool)
        mask[self.stats.attrs['xpix'], self.stats.attrs['ypix']] = True

        return mask

    def add_data(self, name: str, data: Any) -> None:
        self.rec.add_roi_data(self.roi_id, name, data)

    def data_exists(self, name: str) -> bool:
        return self.rec.roi_data_exists(self.roi_id, name)

    def get_data(self, name: str) -> Union[Any, None]:
        return self.rec.get_roi_data(self.roi_id, name)


if __name__ == '__main__':
    f = Summary('../temp_data/2022-09-21_Local_mov_grating_RE_left_OT/Summary.hdf5')

    # import matplotlib.pyplot as plt

    # for i, roi in enumerate(f.rois()):
    #     dff = roi.dff
    #     plt.plot(i + (dff - dff.min()) / (dff.max() - dff.min()), color='black', linewidth='1.')
    # plt.show()

    # for rec in f.recordings():
    #     plt.plot(rec.simple_static_display_regressor('grating_angular_velocity'))
    # plt.show()

    pass
