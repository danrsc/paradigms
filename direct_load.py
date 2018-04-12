from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import Pool
from contextlib import contextmanager
from functools import partial
from itertools import chain
import numpy
import mne
from .twenty_questions import load_block_stimuli_20questions, load_block_stimuli_60words
from .master_stimuli import MasterStimuliPaths, create_master_stimuli
from .generic_fif_reader import load_block_stimuli


__all__ = ['DirectLoad', 'SubjectBlockReduceArgs']


@contextmanager
def multiprocessing_pool():
    pool = Pool()
    yield pool
    pool.terminate()


def _reduce_stimuli(subject_args, direct_load, filter_fn, map_fn, reduce_fn):
    mne_raw = None
    try:
        if subject_args.structural is not None:
            inverse_op, region_labels = direct_load.load_structural(
                subject_args.experiment, subject_args.subject,
                subject_args.structural, subject_args.structural_label_regex)
        else:
            inverse_op = None
            region_labels = None
        mne_raw, stimuli = direct_load.load_block(subject_args.experiment, subject_args.subject, subject_args.block)
        result = None
        is_first = True
        for s in stimuli:
            if inverse_op is not None:
                item = dict(
                    direct_load=direct_load,
                    experiment=subject_args.experiment,
                    subject=subject_args.subject,
                    block=subject_args.block,
                    mne_raw=mne_raw,
                    stimulus=s,
                    all_stimuli=stimuli,
                    inverse_op=inverse_op,
                    region_labels=region_labels)
            else:
                item = dict(
                    direct_load=direct_load,
                    experiment=subject_args.experiment,
                    subject=subject_args.subject,
                    block=subject_args.block,
                    mne_raw=mne_raw,
                    stimulus=s,
                    all_stimuli=stimuli)
            if filter_fn is not None and not filter_fn(item):
                continue
            item = map_fn(item)
            if is_first:
                result = item
                is_first = False
            else:
                result = reduce_fn(result, item)
        return result
    finally:
        if mne_raw is not None:
            mne_raw.close()


class SubjectBlockReduceArgs:

    def __init__(self, experiment, subject, block, structural=None, structural_label_regex=None):
        self._experiment = experiment
        self._subject = subject
        self._block = block
        self._structural = structural
        self._structural_label_regex = structural_label_regex

    @property
    def experiment(self):
        return self._experiment

    @property
    def subject(self):
        return self._subject

    @property
    def block(self):
        return self._block

    @property
    def structural(self):
        return self._structural

    @property
    def structural_label_regex(self):
        return self._structural_label_regex


class DirectLoad:
    """
    High level class for directly loading data (for example from a Jupyter notebook)
    """

    def __init__(
            self,
            session_stimuli_path_format,
            fif_path_format,
            inverse_operator_path_format=None,
            structural_directory=None):
        self.session_stimuli_path_format = session_stimuli_path_format
        self.fif_path_format = fif_path_format
        self.inverse_operator_path_format = inverse_operator_path_format
        self.structural_directory = structural_directory

    def load_epochs(
            self, experiment, subject, blocks, stimulus_to_name_time_pairs, verbose=False, **kwargs):
        all_raw_objects = list()
        names = list()
        events_list = list()
        event_id_offset = 0
        for block in blocks:
            mne_raw, stimuli = self.load_block(experiment, subject, block)
            events = list()
            for name, time in chain.from_iterable(map(stimulus_to_name_time_pairs, stimuli)):
                sample_index = numpy.searchsorted(mne_raw.times, time, side='left')
                events.append(numpy.array([sample_index + mne_raw.first_samp, 0, len(events) + event_id_offset]))
                names.append(name)
            event_id_offset += len(events)
            events_list.append(numpy.array(events))
            all_raw_objects.append(mne_raw)

        virtual_raw, all_events = mne.concatenate_raws(all_raw_objects, preload=False, events_list=events_list)
        try:
            epochs = mne.Epochs(virtual_raw, all_events, add_eeg_ref=False, verbose=verbose, **kwargs)
        except TypeError:
            # add_eeg_ref is gone
            epochs = mne.Epochs(virtual_raw, all_events, verbose=verbose, **kwargs)
        return epochs, names

    def load_block(self, experiment, subject, block):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            try:
                mne_raw = mne.io.Raw(self.fif_path_format.format(
                    experiment=experiment, subject=subject, block=block), add_eeg_ref=False, verbose=False)
            except TypeError:
                # add_eeg_ref is gone
                mne_raw = mne.io.Raw(self.fif_path_format.format(
                    experiment=experiment, subject=subject, block=block), verbose=False)
        if experiment == '20questions':
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                stimuli = load_block_stimuli_20questions(mne_raw)
            return mne_raw, stimuli, False
        if experiment == '60words':
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                stimuli = load_block_stimuli_60words(mne_raw)
            return mne_raw, stimuli, False

        experiment_to_path = {
            'PassAct3': MasterStimuliPaths.passive_active_3,
            'PassAct3Aud': MasterStimuliPaths.passive_active_3,
            'PassAct2': MasterStimuliPaths.passive_active_2,
        }

        if experiment not in experiment_to_path:
            raise ValueError('Don\'t know how to locate master stimuli for experiment: {}'.format(experiment))

        master_stimuli_path = experiment_to_path[experiment]

        master_stimuli, configuration, _ = create_master_stimuli(master_stimuli_path)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            stimuli = load_block_stimuli(
                master_stimuli,
                configuration,
                experiment,
                mne_raw,
                self.session_stimuli_path_format.format(subject=subject, experiment=experiment, block=block),
                int(block) - 1)
        return mne_raw, stimuli

    def load_structural(self, experiment, subject, structural, structural_label_regex=None):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            filled_inv_path = self.inverse_operator_path_format.format(
                subject=subject, experiment=experiment, structural=structural)
            inv = mne.minimum_norm.read_inverse_operator(filled_inv_path, verbose=False)
            labels = mne.read_labels_from_annot(
                structural, parc='aparc', subjects_dir=self.structural_directory, regexp=structural_label_regex)
        return inv, labels

    @staticmethod
    def map_bounds(stimulus_to_time, relative_lower_in_seconds, relative_upper_in_seconds, mne_raw, stimulus):
        zero_time = stimulus_to_time(stimulus)
        start_sample = numpy.searchsorted(mne_raw.times, zero_time + relative_lower_in_seconds, side='left')
        upper_bound_in_samples = numpy.searchsorted(
            mne_raw.times, zero_time + relative_upper_in_seconds, side='right') - 1
        return start_sample, upper_bound_in_samples

    @staticmethod
    def _map_to_flat_index(items, flat_result):
        if isinstance(items, list) or isinstance(items, tuple):
            result = list()
            for item in items:
                result.append(DirectLoad._map_to_flat_index(item, flat_result))
            if isinstance(items, tuple):
                result = tuple(result)
            return result
        elif isinstance(items, dict):
            result = dict()
            for key in items:
                result[key] = DirectLoad._map_to_flat_index(items[key], flat_result)
            return result
        else:
            flat_result.append(items)
            return len(flat_result) - 1

    @staticmethod
    def _reconstruct(items, flat_result):
        if isinstance(items, list) or isinstance(items, tuple):
            result = list()
            for item in items:
                result.append(DirectLoad._reconstruct(item, flat_result))
            if isinstance(items, tuple):
                result = tuple(result)
            return result
        elif isinstance(items, dict):
            result = dict()
            for key in items:
                result[key] = DirectLoad._reconstruct(items[key], flat_result)
            return result
        else:
            return flat_result[items]

    def reduce_subject_stimuli(self, subject_block_reduce_arguments, filter_fn, map_fn, reduce_fn):
        flattened = list()
        struct_with_indices = DirectLoad._map_to_flat_index(subject_block_reduce_arguments, flattened)
        pool_map_fn = partial(
            _reduce_stimuli, direct_load=self, filter_fn=filter_fn, map_fn=map_fn, reduce_fn=reduce_fn)
        with multiprocessing_pool() as pool:
            block_results = list(pool.map(pool_map_fn, flattened))
        return DirectLoad._reconstruct(struct_with_indices, block_results)
