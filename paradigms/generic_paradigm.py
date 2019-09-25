from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy
import mne
from six.moves import zip_longest

from .generic_utility import zip_equal
from .stimulus import Stimulus, Event


__all__ = [
    'EventGroupSpec',
    'group_events',
    'GenericParadigm',
    'EventLoadFixInfo',
    'make_map_to_stimulus_fn',
    'load_session_stimuli_block_events',
    'load_fif_block_events']


class EventGroupSpec(object):
    def __init__(self, start_triggers, key, allow_repeated=False):
        """
        An instance of this class is used as a parameter to group_events. This specification describes
        What kind of group is indicated by the triggers in start_triggers so higher level grouping can be done
        easily
        Args:
            start_triggers: One or more integers. The triggers in the fif events that indicate the start of this
                kind of group
            key: A key for this kind of group. For example 'stimulus', 'question', or 'sentence'
            allow_repeated: Some paradigms (PassAct2) repeat the start of group trigger for every event in the group
                for some kinds of groups (for questions). If this boolean is set, the repeated triggers will be
                grouped into a single group
        """
        if start_triggers is None:
            self._start_triggers = set()
        elif numpy.isscalar(start_triggers):
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


def group_events(event_group_specs, event_stream, trigger_key_fn=None):
    """
    Groups together the events in event_stream, yielding a (key, events) pair for each group. The order of the
    events is unchanged by this operation
    Args:
        event_group_specs: An iterable of EventGroupSpec instances defining how to group events together
        event_stream: A stream of events to group
        trigger_key_fn: a function which returns a trigger given an event. For the purpose of grouping, the
            returned valued overrides the trigger on the event. The event's trigger is not modified
    Returns:
        An iterable of (key, events) pairs with one pair for each group in the stream
    """
    current_events = list()
    current_key = None
    allow_repeat_trigger = None
    for event in event_stream:

        trigger_key = event.trigger
        if trigger_key_fn is not None:
            trigger_key = trigger_key_fn(event)

        if trigger_key == allow_repeat_trigger:
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
            if spec.is_match(trigger_key):
                matched_key = spec.key
                if spec.allow_repeated:
                    allow_repeat_trigger = trigger_key
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
            trigger_key_fn=None,
            visual_stimulus_delay_in_seconds=.037,
            auditory_stimulus_delay_in_seconds=.024):
        self._instruction_trigger = instruction_trigger
        self._event_group_specs = event_group_specs
        self._primary_stimulus_key = primary_stimulus_key
        self._normalize = normalize
        self._trigger_key_fn = trigger_key_fn
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
        for key, events in group_events(self._event_group_specs, event_stream, self._trigger_key_fn):
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
        block_events, event_load_fix_info = load_fif_block_events(
            fif_raw, session_stimuli_mat, index_block, self._instruction_trigger)
        return self.match(block_events), event_load_fix_info

    def flatten(self, stimuli, single_auditory_events=True):
        """
        A pseudo-inverse of match which returns a stream of Event instances. Mostly useful for validating that the
        input event_stream is faithfully represented by the stimuli
        Args:
            stimuli: The stream of stimuli to flatten
            single_auditory_events: If True, then a single Event will be produced for an auditory event to mirror
                the structure of the event_stream that match gets as an input. If False, the inferred events for
                each word of the auditory input are produced
        Returns:
            A stream of Event instances
        """
        raise NotImplementedError('{} does not implement flatten'.format(type(self)))

    # noinspection PyMethodMayBeStatic
    def _get_master_remove_attributes(self, master, stimulus_events):
        keys = set([k for k, v in stimulus_events])
        return [k for k in master if isinstance(master[k], Stimulus) and k not in keys]

    def match(self, event_stream):
        """
        Converts a low level event stream (i.e. Event instances extracted from a fif file/session stimuli combination)
        to a high level stimulus stream where each Stimulus has attributes and children which can themselves be Stimulus
        instances. This function also handles the
        Args:
            event_stream: The stream of Event instances to convert
        Returns:
            A stream of Stimulus instances
        """
        stimulus_counts = dict()
        word_counts = dict()

        stimuli = list()
        for index_stimulus, stimulus_events in enumerate(self.iterate_stimulus_events(event_stream)):
            master = None
            masters = list()
            delays = list()
            final_events = list()
            simple_attribute_adds = list()
            simple_attribute_deletes = list()
            for key, events in stimulus_events:

                # this is a useful place to print stuff for debugging
                # print(key, [(ev.trigger, ev.stimulus) for ev in events])

                if key == self._primary_stimulus_key:
                    master, adds, deletes = self._map_primary_events(key, events)
                    masters.append(master)
                    if adds is None:
                        adds = dict()
                    if deletes is None:
                        deletes = set()
                    deletes.update(self._get_master_remove_attributes(master, stimulus_events))
                    simple_attribute_adds.append(adds)
                    simple_attribute_deletes.append(deletes)
                    stimulus_count = stimulus_counts[master] if master in stimulus_counts else 0
                    stimulus_count += 1
                    stimulus_counts[master] = stimulus_count
                else:
                    assert(master is not None)
                    additional_master, additional_master_adds, additional_master_deletes = \
                        self._map_additional_events(master, key, events)
                    masters.append(additional_master)
                    if additional_master_adds is None:
                        additional_master_adds = dict()
                    if additional_master_deletes is None:
                        additional_master_deletes = set()
                    simple_attribute_adds.append(additional_master_adds)
                    simple_attribute_deletes.append(additional_master_deletes)

                stimulus_delay = self.visual_stimulus_delay_in_seconds

                if any(self._is_auditory_trigger(ev.trigger) for ev in events):
                    # for now we only support a single stimulus event if it is auditory
                    if len(events) > 1:
                        raise ValueError('Expected only one auditory stimulus event')
                    # map_additional_events can return a key, StimulusBuilder pair. If so, pass the StimulusBuilder
                    m = masters[-1]
                    if isinstance(m, tuple):
                        m = m[1]
                    events = self._infer_auditory_events(m, key, events[0])
                    stimulus_delay = self.auditory_stimulus_delay_in_seconds
                    if Stimulus.modality_attribute_name not in simple_attribute_adds[-1]:
                        simple_attribute_adds[-1][Stimulus.modality_attribute_name] = Stimulus.auditory_modality
                else:
                    if Stimulus.modality_attribute_name not in simple_attribute_adds[-1]:
                        simple_attribute_adds[-1][Stimulus.modality_attribute_name] = Stimulus.visual_modality

                delays.append(stimulus_delay)
                final_events.append(events)

            stimulus = masters[0].copy_with_event_attributes(
                final_events[0],
                delays[0],
                simple_attribute_adds[0],
                simple_attribute_deletes[0],
                stimulus_counts[masters[0]],
                self.normalize,
                word_counts,
                masters[1:],
                final_events[1:],
                delays[1:],
                simple_attribute_adds[1:],
                simple_attribute_deletes[1:])
            stimuli.append(stimulus)

        return stimuli

    @property
    def relative_session_stimuli_path_format(self):
        return '{experiment}/meta/{subject}/sentenceBlock.mat'

    def map_recording_to_session_stimuli_path(self, recording_tuple):
        experiment_dir = os.path.split(recording_tuple.full_path)[0]
        for _ in range(3):  # move up 3 levels (data/processing/subject)
            experiment_dir = os.path.split(experiment_dir)[0]
        file_name = self.relative_session_stimuli_path_format.format(**recording_tuple)
        return os.path.join(experiment_dir, file_name)

    def make_compute_lower_upper_bounds_from_master_stimuli(self, recording_tuple):

        def compute_lower_upper_bounds(mne_raw):
            session_stimuli_path = self.map_recording_to_session_stimuli_path(recording_tuple)
            stimuli, event_load_fix_info = self.load_block_stimuli(
                mne_raw, session_stimuli_path, int(recording_tuple.recording) - 1)

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


class EventLoadFixInfo:

    def __init__(self, num_removed, is_first_removed, is_session_removed):
        self._num_removed = num_removed
        self._is_first_removed = is_first_removed
        self._is_session_removed = is_session_removed

    @property
    def num_removed(self):
        return self._num_removed

    @property
    def is_first_removed(self):
        return self._is_first_removed

    @property
    def is_session_removed(self):
        return self._is_session_removed


def load_fif_block_events(mne_raw, session_stimuli_mat, index_block, instruction_trigger):

    event_code_text_pairs = load_session_stimuli_block_events(session_stimuli_mat, index_block)
    # filter out the 0 event codes
    event_code_text_pairs = [(c, t) for c, t in event_code_text_pairs if c != 0 and c != instruction_trigger]

    index_sample_column = 0
    index_prev_id_column = 1
    index_event_id_column = 2

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        fif_events = mne.find_events(
            mne_raw, stim_channel='STI101', shortest_event=1, uint_cast=True, min_duration=0.000, verbose=False)

    # filter out the instructions
    fif_events = fif_events[fif_events[:, index_event_id_column] != instruction_trigger]

    # hack: filter out events where the middle column of the next event is not 0
    # this column indicates the value of the channel just before the current event. In our paradigms, this should
    # always be a 0. When it is not 0, the previous event is probably bogus
    # In practice this seems to solve event finding problems
    indices_non_zero = numpy.where(fif_events[:, index_prev_id_column] != 0)[0]
    if len(indices_non_zero) > 0:
        indices_bogus_events = indices_non_zero - 1
        indicator_real_events = numpy.full(fif_events.shape[0], True)
        indicator_real_events[indices_bogus_events] = False
        fif_events = fif_events[indicator_real_events]

    event_load_fix_info = None
    if len(event_code_text_pairs) != fif_events.shape[0]:
        fif_codes = fif_events[:, index_event_id_column]
        session_codes = numpy.array([c for c, t in event_code_text_pairs])

        which_part = None
        if fif_codes.shape[0] > session_codes.shape[0]:
            which_side = 'fif'
            diff = fif_codes.shape[0] - session_codes.shape[0]
            if numpy.array_equal(fif_codes[diff:], session_codes):
                which_part = 'first'
                fif_events = fif_events[diff:]
            elif numpy.array_equal(fif_codes[:-diff], session_codes):
                which_part = 'last'
                fif_events = fif_events[:-diff]
        else:
            which_side = 'session'
            diff = session_codes.shape[0] - fif_codes.shape[0]
            if numpy.array_equal(fif_codes, session_codes[diff:]):
                which_part = 'first'
                event_code_text_pairs = event_code_text_pairs[diff:]
            elif numpy.array_equal(fif_codes, session_codes[:-diff]):
                which_part = 'last'
                event_code_text_pairs = event_code_text_pairs[:-diff]

        if len(event_code_text_pairs) != fif_events.shape[0]:
            for f, e in zip_longest(fif_events, event_code_text_pairs):
                print(f, e)
            raise ValueError(
                'Number of events in session_stimuli_mat ({}) is not equal to the number of fif_events ({}). '
                'File: {}'.format(
                    len(event_code_text_pairs),
                    fif_events.shape[0],
                    mne_raw.info['filename'] if 'filename' in mne_raw.info else 'unknown'))
        else:
            event_load_fix_info = EventLoadFixInfo(diff, which_part == 'first', which_side == 'session')
            print('Warning: mismatched events between fif file and session stimuli file. '
                  'Fixed by removing the {} {} events from the {} events.'.format(which_part, diff, which_side))

    time_indices = fif_events[:, index_sample_column] - mne_raw.first_samp
    time_stamps = mne_raw.times[time_indices]
    durations = numpy.concatenate([numpy.diff(time_stamps), numpy.array([mne_raw.times[-1] - time_stamps[-1]])])

    result = list()
    for index, ((session_code, session_text), fif_event, time_stamp, duration) in enumerate(
            zip(event_code_text_pairs, fif_events, time_stamps, durations)):
        if session_code != fif_event[index_event_id_column]:
            for f, e in zip_longest(fif_events, event_code_text_pairs):
                print(f, e)

            raise ValueError('Mismatch in event codes at index {}. Session event: {}, fif event: {}. File: {}'.format(
                index, session_code, fif_event[index_event_id_column],
                mne_raw.info['filename'] if 'filename' in mne_raw.info else 'unknown'))

        # some mild clean up of the punctuation is done here to match what was done in preprocessed files
        clean_text = session_text.strip().replace(u'\u2019', "'").replace(u'\u201c', '"').replace(u'\u201d', '"')

        result.append(Event(
            stimulus=clean_text,
            duration=duration,
            trigger=fif_event[index_event_id_column],
            time_stamp=time_stamp
        ))

    return result, event_load_fix_info
