from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict

import h5py
import numpy as np

log = logging.getLogger(__name__)


class SummaryFile:

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.h5f = h5py.File(self.file_path, 'r')

        # Load recordings
        self._recordings: List[Recording] = []
        for grp in self.h5f['recording'].values():
            date = grp.attrs['date']
            fish_id = grp.attrs['fish_id']
            rec_id = grp.attrs['rec_id']
            self._recordings.append(Recording(date, fish_id, rec_id, grp))

        # Load ROIs
        self._rois = [Roi(*ref, self) for ref in self.h5f['roi_refs'][:]]

    def __repr__(self):
        return f'{SummaryFile.__name__}(\'{self.file_path}\')'

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


class Recording:

    def __init__(self, date: int, fish_id: int, rec_id: int, h5group: h5py.Group):

        self.date = date
        self.fish_id = fish_id
        self.rec_id = rec_id
        self.h5group = h5group

        # Dict which may contain a regressor for each simple parameter
        self._simple_regressors: Dict[str, np.ndarray] = {}
        # List of the start and end indices within the calcium recording
        self._simple_phase_index_intervals: List[Tuple[int, int]]

    def __repr__(self):
        return f'{self.__class__.__name__}()'

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

    def simple_static_display_regressor(self, parameter_name: str):
        ref_name = f'__display_static_{parameter_name}'

        # If it has been calculated before for this parameter, return the stored result
        if ref_name in self._simple_regressors:
            return self._simple_regressors[ref_name]

        regr = np.zeros(self.record_group_id.shape)
        for phase_id in np.unique(self.record_group_id):
            if str(phase_id) not in self.h5group['display_data']['phases']:
                continue

            phase_grp = self.h5group['display_data']['phases'][str(phase_id)]

            if parameter_name not in phase_grp.attrs:
                continue

            regr[self.record_group_id == phase_id] = phase_grp.attrs[parameter_name]

        if (regr != 0).sum() == 0:
            log.warning(f'Tried building static regressor of parameter "{parameter_name}". '
                        f'Parameter not found')

        self._simple_regressors[ref_name] = regr

        return regr

    def simple_camera_regressor(self, parameter_name: str):
        ref_name = f'__camera_{parameter_name}'
        pass

    def _calculate_simple_phase_index_list(self):
        unordered = []
        for grp in self.h5group['display_data']['phases'].values():
            if not isinstance(grp, h5py.Group):
                continue

            minidx = np.argmin(self.ca_times - grp.attrs['start_time'])
            maxidx = np.argmin(self.ca_times - grp.attrs['end_time'])
            print(minidx, maxidx)

            # unordered.append((int(grp.name), start_idx, end_idx))

    def phase_metrics(self, roi_id: int = None):
        pass


@dataclass
class Roi:
    date: int
    fish_id: int
    rec_id: int
    roi_id: int
    file: SummaryFile

    @property
    def rec(self):
        return self.file.recordings(self.date, self.fish_id, self.rec_id)[0]

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

    def simple_static_display_regressor(self, parameter_name: str):
        return self.rec.simple_static_display_regressor(parameter_name)


if __name__ == '__main__':
    f = SummaryFile('../temp_data/2022-09-21_Local_mov_grating_RE_left_OT/Summary.hdf5')

    import matplotlib.pyplot as plt

    # for i, roi in enumerate(f.rois()):
    #     dff = roi.dff
    #     plt.plot(i + (dff - dff.min()) / (dff.max() - dff.min()), color='black', linewidth='1.')
    # plt.show()

    for rec in f.recordings():
        plt.plot(rec.simple_static_display_regressor('grating_angular_velocity'))
    plt.show()

    pass
