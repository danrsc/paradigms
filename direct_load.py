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
from .master_stimuli import MasterStimuli


__all__ = ['DirectLoad', 'SubjectBlockReduceArgs', 'gather_epoch_events', 'region_label_indices']


@contextmanager
def multiprocessing_pool(processes=None, maxtasksperchild=None):
    pool = Pool(processes=processes, maxtasksperchild=maxtasksperchild)
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


def gather_epoch_events(keys):
    result = dict()
    for i, k in enumerate(keys):
        if k not in result:
            result[k] = ['{}'.format(i)]
        else:
            result[k].append('{}'.format(i))
    return result


def region_label_indices(source_estimates, label):
    # code taken from https://github.com/mne-tools/mne-python/blob/maint/0.15/mne/source_estimate.py
    # modified to enable indexing into a numpy array later instead of keeping a SourceEstimates instance

    def _hemilabel_indices(src_estimates, lbl):
        if lbl.hemi == 'lh':
            source_vertices = src_estimates.vertices[0]
        else:
            source_vertices = src_estimates.vertices[1]

        # find index of the Label's vertices
        idx = numpy.nonzero(numpy.in1d(source_vertices, lbl.vertices))[0]

        # find data
        if lbl.hemi == 'rh':
            return idx + len(src_estimates.vertices[0])
        else:
            return idx

    if label.hemi == 'both':
        left_indices = _hemilabel_indices(source_estimates, label.lh)
        right_indices = _hemilabel_indices(source_estimates, label.rh)
        return numpy.concatenate((left_indices, right_indices), axis=0)
    elif label.hemi == 'lh':
        return _hemilabel_indices(source_estimates, label)
    elif label.hemi == 'rh':
        return _hemilabel_indices(source_estimates, label)
    else:
        raise TypeError('Expected  Label or BiHemiLabel; got {}'.format(label))


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

    def load_source_estimates(self, experiment, subject, blocks, structural, stimulus_to_name_time_pairs, tmin, tmax):
        results = list(self.iterate_source_estimates(
            experiment, subject, blocks, structural, stimulus_to_name_time_pairs,
            partition_key_fn=lambda s: 0, tmin=tmin, tmax=tmax))
        assert(len(results) == 1)
        return results[0][1]

    def iterate_source_estimates(
            self,
            experiment,
            subject,
            blocks,
            structural,
            stimulus_to_name_time_pairs,
            partition_key_fn,
            tmin,
            tmax):

        epochs, keys = self.load_epochs(
            experiment, subject, blocks, stimulus_to_name_time_pairs, tmin=tmin, tmax=tmax)

        key_to_events = gather_epoch_events(keys)

        partitioned_keys = dict()
        for key in key_to_events:
            partition_key = partition_key_fn(key)
            if partition_key not in partitioned_keys:
                partitioned_keys[partition_key] = [key]
            else:
                partitioned_keys[partition_key].append(key)

        inv_op, region_labels = self.load_structural(experiment, subject, structural)

        time = None
        for partition_key in partitioned_keys:
            result = dict()
            stimuli_keys = partitioned_keys[partition_key]
            for key in stimuli_keys:
                key_epochs = epochs[key_to_events[key]]
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=DeprecationWarning)
                    key_evoked = key_epochs.average()
                    if time is None:
                        time = key_evoked.times
                    result[key] = mne.minimum_norm.apply_inverse(key_evoked, inv_op, verbose=False)
            yield partition_key, (result, time, region_labels)

    def multiprocess_imap_unordered_epochs(
            self, experiment, subject, blocks, stimulus_to_name_time_pairs, group_key_fn, map_fn,
            is_debug_with_single_process=False, processes=None, maxtasksperchild=None, **load_epochs_kwargs):

        epochs, keys, events_list = self._load_epochs_internal(
            experiment, subject, blocks, stimulus_to_name_time_pairs, **load_epochs_kwargs)

        fif_paths = [
            self.fif_path_format.format(experiment=experiment, subject=subject, block=block) for block in blocks]

        key_to_events = gather_epoch_events(keys)

        group_keys = dict()
        for key in key_to_events:
            group_key = group_key_fn(key)
            if group_key not in group_keys:
                group_keys[group_key] = [key]
            else:
                group_keys[group_key].append(key)

        map_fn_wrapper = partial(
            DirectLoad._multiprocess_map_epochs_fn,
            events_list, fif_paths, key_to_events, group_keys, map_fn, **load_epochs_kwargs)

        if is_debug_with_single_process:
            for item in map(map_fn_wrapper, group_keys):
                yield item
        else:
            with multiprocessing_pool(processes=processes, maxtasksperchild=maxtasksperchild) as pool:
                for item in pool.imap_unordered(map_fn_wrapper, group_keys):
                    yield item

    @staticmethod
    def _multiprocess_map_epochs_fn(
            events_list, fif_paths, key_to_events, group_keys, map_fn, group_key, **load_epochs_kwargs):

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            epochs = DirectLoad._load_pre_mapped_epochs(events_list, fif_paths, **load_epochs_kwargs)
        current_keys = group_keys[group_key]
        to_map = dict([(key, epochs[key_to_events[key]]) for key in current_keys])
        return map_fn(group_key, to_map)

    @staticmethod
    def _load_pre_mapped_epochs(events_list, fif_paths, **kwargs):
        try:
            all_raw_objects = [mne.io.Raw(fif_path, add_eeg_ref=False, verbose=False) for fif_path in fif_paths]
        except TypeError:
            all_raw_objects = [mne.io.Raw(fif_path, verbose=False) for fif_path in fif_paths]

        virtual_raw, all_events = mne.concatenate_raws(all_raw_objects, preload=False, events_list=events_list)
        try:
            epochs = mne.Epochs(virtual_raw, all_events, add_eeg_ref=False, verbose=False, **kwargs)
        except TypeError:
            # add_eeg_ref is gone
            epochs = mne.Epochs(virtual_raw, all_events, verbose=False, **kwargs)
        return epochs

    def load_epochs(
            self, experiment, subject, blocks, stimulus_to_name_time_pairs, verbose=False, **kwargs):
        epochs, names, events_list = self._load_epochs_internal(
            experiment, subject, blocks, stimulus_to_name_time_pairs, verbose, **kwargs)
        return epochs, names

    def _load_epochs_internal(
            self, experiment, subject, blocks, stimulus_to_name_time_pairs, verbose=False, **kwargs):
        all_raw_objects = list()
        names = list()
        events_list = list()
        event_id_offset = 0
        for block in blocks:
            mne_raw, stimuli, event_load_fix_info = self.load_block(experiment, subject, block)
            events = list()
            for item in chain.from_iterable(map(stimulus_to_name_time_pairs, stimuli)):
                if len(item) != 2:
                    raise ValueError('Expected stimulus_to_name_time_pairs to return a list of pairs for each '
                                     'stimulus. Are you returning just a single pair? Got: {}'.format(item))
                name, time = item
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
        return epochs, names, events_list

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
            return mne_raw, stimuli, None
        if experiment == '60words':
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                stimuli = load_block_stimuli_60words(mne_raw)
            return mne_raw, stimuli, None

        paradigm = MasterStimuli.paradigm_from_experiment(experiment)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            stimuli, event_load_fix_info = paradigm.load_block_stimuli(
                mne_raw,
                self.session_stimuli_path_format.format(subject=subject, experiment=experiment, block=block),
                int(block) - 1)
        return mne_raw, stimuli, event_load_fix_info

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
