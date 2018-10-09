from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Mapping
from .generic_utility import get_keyword_properties


__all__ = ['RecordingTuple', 'ROITuple']


class RecordingTuple(Mapping):

    def __init__(self, experiment, subject, recording, full_path):
        self._experiment = experiment
        self._subject = subject
        self._recording = recording
        self._full_path = full_path
        self._keyword_properties = get_keyword_properties(self, just_names=True)

    @property
    def experiment(self):
        return self._experiment

    @property
    def subject(self):
        return self._subject

    @property
    def recording(self):
        return self._recording

    @property
    def full_path(self):
        return self._full_path

    def __len__(self):
        return len(self._keyword_properties)

    def __iter__(self):
        for x in self._keyword_properties:
            yield x

    def __getitem__(self, item):
        # together with __iter__ this enables the keyword unpacking syntax: **recording_tuple
        return getattr(self, item)

    @property
    def exp_sub_rec_str(self):
        return '({}, {}, {})'.format(self.experiment, self.subject, self.recording)

    def __str__(self):
        return self.exp_sub_rec_str

    def subject_id(self):
        return hash(self.experiment + self.subject)

    def experiment_id(self):
        return hash(self.experiment)

    def recording_ordinal(self):
        try:
            return int(self.recording)
        except ValueError:
            return hash(self.recording)


class ROITuple:

    def __init__(self, experiment, subject, roi, label_key, presentation, full_path):
        self._experiment = experiment
        self._subject = subject
        self._roi = roi
        self._label_key = label_key
        self._presentation = presentation
        self._full_path = full_path
        self._keyword_properties = get_keyword_properties(self, just_names=True)

    @property
    def experiment(self):
        return self._experiment

    @property
    def subject(self):
        return self._subject

    @property
    def roi(self):
        return self._roi

    @property
    def label_key(self):
        return self._label_key

    @property
    def presentation(self):
        return self._presentation

    @property
    def full_path(self):
        return self._full_path

    def __len__(self):
        return len(self._keyword_properties)

    def __iter__(self):
        for x in self._keyword_properties:
            yield x

    def __getitem__(self, item):
        # together with __iter__ this enables the keyword unpacking syntax: **recording_tuple
        return getattr(self, item)

    def __str__(self):
        return '({}, {}, {}, {}, {})'.format(self.experiment, self.subject, self.roi, self.label_key, self.presentation)

    def subject_id(self):
        return hash(self.experiment + self.subject)

    def experiment_id(self):
        return hash(self.experiment)

    def roi_id(self):
        return hash(self.roi)
