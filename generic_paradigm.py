from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy
import mne

from brain_gen.core import flags, zip_equal
from .stimulus import Stimulus, Event


__all__ = [
    'EventGroupSpec',
    'group_events',
    'GenericParadigm',
    'make_map_to_stimulus_fn',
    'map_recording_to_session_stimuli_path',
    'load_session_stimuli_block_events',
    'load_fif_block_events']


class EventGroupSpec(object):

    def __init__(self, start_triggers, key, allow_repeated=False):
        if numpy.isscalar(start_triggers):
            self._start_triggers = {start_triggers}
        else:
            self._start_triggers = set(start_triggers)
        self._key = key
        self._allow_repeated = allow_repeated

    def is_match(self, trigger):
        return trigger in self._start_triggers

    @property
    def key(self):
        return self._key

    @property
    def allow_repeated(self):
        return self._allow_repeated


def group_events(event_group_specs, block_events):
    current_events = list()
    current_key = None
    allow_repeat_trigger = None
    for event in block_events:

        if event.trigger == allow_repeat_trigger:
            assert(len(current_events) > 0)
            current_events.append(event)
            continue
        elif allow_repeat_trigger is not None:
            allow_repeat_trigger = None
            assert(len(current_events) > 0)
            assert(current_key is not None)
            yield current_key, current_events
            current_key = None
            current_events = list()

        matched_key = None
        for spec in event_group_specs:
            if spec.is_match(event.trigger):
                matched_key = spec.key
                if spec.allow_repeated:
                    allow_repeat_trigger = event.trigger
                break
        if matched_key is not None:
            if len(current_events) > 0:
                assert(current_key is not None)
                yield current_key, current_events
            current_key = matched_key
            current_events = [event]
        else:
            if current_key is None:
                raise ValueError('Unable to start group. Don\'t know what to do at this point in stream.')
            current_events.append(event)

    if len(current_events) > 0:
        assert(current_key is not None)
        yield current_key, current_events


class GenericParadigm(object):

    def __init__(
            self,
            instruction_trigger,
            event_group_specs,
            primary_stimulus_key,
            normalize,
            visual_stimulus_delay_in_seconds=.037,
            auditory_stimulus_delay_in_seconds=.024):
        self._instruction_trigger = instruction_trigger
        self._event_group_specs = event_group_specs
        self._primary_stimulus_key = primary_stimulus_key
        self._normalize = normalize
        self._visual_stimulus_delay_in_seconds = visual_stimulus_delay_in_seconds
        self._auditory_stimulus_delay_in_seconds = auditory_stimulus_delay_in_seconds

    @property
    def normalize(self):
        return self._normalize

    @property
    def visual_stimulus_delay_in_seconds(self):
        return self._visual_stimulus_delay_in_seconds

    @property
    def auditory_stimulus_delay_in_seconds(self):
        return self._auditory_stimulus_delay_in_seconds

    def _is_auditory_trigger(self, trigger):
        raise NotImplementedError('{} does not implement _is_auditory_trigger'.format(type(self)))

    def _infer_auditory_events(self, master_stimulus, key, auditory_event):
        raise NotImplementedError('{} does not implement _infer_auditory_events'.format(type(self)))

    def _map_primary_events(self, key, events):
        raise NotImplementedError('{} does not implement _map_primary_events'.format(type(self)))

    def _map_additional_events(self, master_stimulus, key, events):
        raise NotImplementedError('{} does not implement _map_additional_events'.format(type(self)))

    def iterate_stimulus_events(self, event_stream):
        stimulus_accumulator = list()
        for key, events in group_events(self._event_group_specs, event_stream):
            if len(stimulus_accumulator) == 0 and key != self._primary_stimulus_key:
                raise ValueError('Unable to start stimulus')
            if key == self._primary_stimulus_key:
                if len(stimulus_accumulator) > 0:
                    yield stimulus_accumulator
                stimulus_accumulator = [(key, events)]
            else:
                stimulus_accumulator.append((key, events))
        if len(stimulus_accumulator) > 0:
            yield stimulus_accumulator

    def load_block_stimuli(self, fif_raw, session_stimuli_mat, index_block):
        block_events, _ = load_fif_block_events(
            fif_raw, session_stimuli_mat, index_block, self._instruction_trigger)
        return self.match(block_events)

    def match(self, event_stream):

        stimulus_counts = dict()
        word_counts = dict()

        stimuli = list()
        for index_stimulus, stimulus_events in enumerate(self.iterate_stimulus_events(event_stream)):
            master = None
            masters = list()
            delays = list()
            final_events = list()
            for key, events in stimulus_events:
                if key == self._primary_stimulus_key:
                    master = self._map_primary_events(key, events)
                    masters.append(master)
                    stimulus_count = stimulus_counts[master] if master in stimulus_counts else 0
                    stimulus_count += 1
                    stimulus_counts[master] = stimulus_count
                else:
                    assert(master is not None)
                    masters.append(self._map_additional_events(master, key, events))

                stimulus_delay = self.visual_stimulus_delay_in_seconds

                if any(self._is_auditory_trigger(ev.trigger) for ev in events):
                    # for now we only support a single stimulus event if it is auditory
                    if len(events) > 1:
                        raise ValueError('Expected only one auditory stimulus event')
                    # map_additional_events can return a key, StimulusBuilder pair. If so, pass the StimulusBuilder
                    m = masters[-1]
                    if isinstance(m, tuple):
                        m = m[1]
                    events = self._infer_auditory_events(m, key, events)
                    stimulus_delay = self.auditory_stimulus_delay_in_seconds

                delays.append(stimulus_delay)
                final_events.append(events)

            stimulus = masters[0].copy_with_event_attributes(
                final_events[0],
                delays[0],
                stimulus_counts[masters[0]],
                self.normalize,
                word_counts,
                masters[1:],
                final_events[1:],
                delays[1:])
            stimuli.append(stimulus)

        return stimuli

    def make_compute_lower_upper_bounds_from_master_stimuli(self, recording_tuple):

        def compute_lower_upper_bounds(mne_raw):
            session_stimuli_path = map_recording_to_session_stimuli_path(recording_tuple)
            stimuli = self.load_block_stimuli(mne_raw, session_stimuli_path, int(recording_tuple.recording) - 1)

            return [(
                stimulus,
                stimulus[Stimulus.time_stamp_attribute_name],
                stimulus[Stimulus.time_stamp_attribute_name] + stimulus[Stimulus.duration_attribute_name]
            ) for stimulus in stimuli]

        return compute_lower_upper_bounds


def make_map_to_stimulus_fn(master_stimuli, normalize):
    # this works for both audio and visual stimuli
    normalized_text_to_master = dict()
    for master_stimulus in master_stimuli:
        normalized = ''.join(map(lambda s: normalize(s.text), master_stimulus.iter_level(Stimulus.word_level)))
        if normalized in normalized_text_to_master:
            raise ValueError('Two master stimuli map to the same key: {}'.format(normalized))
        normalized_text_to_master[normalized] = master_stimulus

    def _match(events):
        key = ''.join(map(lambda ev: normalize(ev.stimulus), events))
        if key not in normalized_text_to_master:
            raise KeyError('Unable to match presentation to master stimuli: {0}'.format(key))
        else:
            return normalized_text_to_master[key]

    return _match


def map_recording_to_session_stimuli_path(recording_tuple):
    file_name = flags().relative_session_stimuli_path_format.format(**recording_tuple)
    return os.path.join(flags().data_root, file_name)


def load_session_stimuli_block_events(session_stimuli_mat, index_block):
    """
    Loads the output of psychtoolbox into a list of event-code, text pairs
    Args:
        session_stimuli_mat: The pyschtoolbox output for the experiment, ususally called 'sentenceBlock.mat'
        index_block: Which block of the experiment to load

    Returns:
        An array of event codes and a list of
    """

    if isinstance(session_stimuli_mat, type('')):
        from scipy.io import loadmat
        session_stimuli_mat = loadmat(session_stimuli_mat)

    block_stimuli_data = session_stimuli_mat['experiment'].squeeze()[index_block]

    def _read_field(field_names_to_try_in_order):
        for field_name in field_names_to_try_in_order:
            if field_name in block_stimuli_data.dtype.fields:
                return block_stimuli_data[field_name]
        raise ValueError(
            'One of the following fields must exist in the referenced session_stimuli_mat file: {0}'.format(
                field_names_to_try_in_order))

    # in various experiments the name of the text field has differed. These are in priority order of field names to try
    block_text_arr = _read_field(['stim_txt', 'stimulus', 'story']).squeeze()
    block_text = list()
    for index_stimulus in range(block_text_arr.shape[0]):
        block_text.append(str(block_text_arr[index_stimulus].squeeze().tolist()))

    stimuli_events = _read_field(['event', 'trigger', 'parPort']).squeeze().tolist()
    return zip_equal(stimuli_events, block_text)


def load_fif_block_events(mne_raw, session_stimuli_mat, index_block, instruction_trigger):

    event_code_text_pairs = load_session_stimuli_block_events(session_stimuli_mat, index_block)
    # filter out the 0 event codes
    event_code_text_pairs = [(c, t) for c, t in event_code_text_pairs if c != 0]

    index_sample_column = 0
    index_event_id_column = 2

    fif_events = mne.find_events(
        mne_raw, stim_channel='STI101', shortest_event=1, uint_cast=True, min_duration=0, verbose=False)
    # filter out the instructions
    fif_events = fif_events[fif_events[:, index_event_id_column] != instruction_trigger]

    # validate the matching
    if len(event_code_text_pairs) != fif_events.shape[0]:
        raise ValueError(
            'Number of events in session_stimuli_mat ({}) is not equal to the number of fif_events ({}). '
            'File: {}'.format(len(event_code_text_pairs), fif_events.shape[0], mne_raw.info['filename']))

    time_indices = fif_events[:, index_sample_column] - mne_raw.first_samp
    time_stamps = mne_raw.times[time_indices]
    durations = numpy.concatenate([numpy.diff(time_stamps), numpy.array([mne_raw.times[-1] - time_stamps[-1]])])

    result = list()
    for index, ((session_code, session_text), fif_event, time_stamp, duration) in enumerate(
            zip(event_code_text_pairs, fif_events, time_stamps, durations)):
        if session_code != fif_event[index_event_id_column]:
            raise ValueError('Mismatch in event codes at index {}. Session event: {}, fif event: {}. File: {}'.format(
                index, session_code, fif_event[index_event_id_column], mne_raw.info['filename']))

        # some mild clean up of the punctuation is done here to match what was done in preprocessed files
        clean_text = session_text[index].strip().replace(u'\u2019', "'").replace(u'\u201c', '"').replace(u'\u201d', '"')

        result.append(Event(
            stimulus=clean_text,
            duration=duration,
            trigger=fif_event[index_event_id_column],
            time_stamp=time_stamp
        ))

    return result
