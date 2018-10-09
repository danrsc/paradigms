from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import string
from itertools import zip_longest
import numpy
import mne
from .generic_utility import zip_equal
from .stimulus import Stimulus, StimulusBuilder, add_word_stimuli_to_parent, make_root_stimulus_builder, Event
from .generic_paradigm import EventLoadFixInfo


__all__ = [
    'make_krns_2_master_sentence_stimuli',
    'load_fif_block_events_krns_2',
    'load_session_stimuli_block_events_krns_2',
    'KRNS2']


def make_krns_2_master_sentence_stimuli():

    word_result = list()
    sentence_result = list()

    def _make_sentence_stimulus(
            tagged_sentence,
            is_active,
            question,
            is_yes_correct):

        def _process_tagged(parent, s):
            add_word_stimuli_to_parent(parent, s)
            for index_word_stimulus, word_stimulus in enumerate(parent.children):
                word_stimulus[Stimulus.position_in_root_attribute_name] = index_word_stimulus

        sb = make_root_stimulus_builder(Stimulus.sentence_level, len(sentence_result) + len(word_result))
        sb['is_active'] = is_active
        sb['is_yes_correct'] = is_yes_correct
        sb['question'] = StimulusBuilder(Stimulus.word_level, attributes=sb.copy_attributes())
        sb['question'][Stimulus.text_attribute_name] = question

        _process_tagged(sb, tagged_sentence)

        sentence_result.append(sb.make_stimulus())

    def _make_word_stimulus(word, part_of_speech, question, is_yes_correct):
        sb = make_root_stimulus_builder(Stimulus.word_level, len(sentence_result) + len(word_result))
        sb['is_yes_correct'] = is_yes_correct
        sb['question'] = StimulusBuilder(Stimulus.word_level, attributes=sb.copy_attributes())
        sb['question'][Stimulus.text_attribute_name] = question
        sb[Stimulus.text_attribute_name] = word
        sb[Stimulus.part_of_speech_attribute_name] = part_of_speech
        word_result.append(sb.make_stimulus())

    active_sentences = list()
    passive_sentences = list()

    def _add_stimulus_pair(tagged_active_sentence, tagged_passive_sentence, question, is_yes_correct):
        active_sentences.append((tagged_active_sentence, True, question, is_yes_correct))
        passive_sentences.append((tagged_passive_sentence, False, question, is_yes_correct))

    _add_stimulus_pair(
        'A/DT dog/NN found/VBD the/DT peach./NN',
        'The/DT peach/NN was/VBD found/VBN by/IN a/DT dog./NN',
        'Was there a vegetable?', False)
    _add_stimulus_pair(
        'The/DT dog/NN kicked/VBD a/DT school./NN',
        'A/DT school/NN was/VBD kicked/VBN by/IN the/DT dog./NN',
        'Could you get hurt doing this?', True)
    _add_stimulus_pair(
        'A/DT dog/NN inspected/VBD a/DT hammer./NN',
        'A/DT hammer/NN was/VBD inspected/VBN by/IN a/DT dog./NN',
        'Was a tool seen?', True)
    _add_stimulus_pair(
        'The/DT dog/NN touched/VBD the/DT door./NN',
        'The/DT door/NN was/VBD touched/VBN by/IN the/DT dog./NN',
        'Was it bouncy?', False)

    _add_stimulus_pair(
        'The/DT doctor/NN found/VBD a/DT school./NN',
        'A/DT school/NN was/VBD found/VBN by/IN the/DT doctor./NN',
        'Could this item be put in a pocket?', False)
    _add_stimulus_pair(
        'A/DT doctor/NN kicked/VBD the/DT peach./NN',
        'The/DT peach/NN was/VBD kicked/VBN by/IN a/DT doctor./NN',
        'Was fruit damaged?', True)
    _add_stimulus_pair(
        'A/DT doctor/NN inspected/VBD a/DT door./NN',
        'A/DT door/NN was/VBD inspected/VBN by/IN a/DT doctor./NN',
        'Was it a hard surface?', True)
    _add_stimulus_pair(
        'The/DT doctor/NN touched/VBD the/DT hammer./NN',
        'The/DT hammer/NN was/VBD touched/VBN by/IN the/DT doctor./NN',
        'Was it bendy?', False)

    _add_stimulus_pair(
        'The/DT student/NN found/VBD a/DT door./NN',
        'A/DT door/NN was/VBD found/VBN by/IN the/DT student./NN',
        'Was this item bigger than a whale?', False)
    _add_stimulus_pair(
        'A/DT student/NN kicked/VBD the/DT hammer./NN',
        'The/DT hammer/NN was/VBD kicked/VBN by/IN a/DT student./NN',
        'Was it a soft item?', False)
    _add_stimulus_pair(
        'The/DT student/NN inspected/VBD the/DT school./NN',
        'The/DT school/NN was/VBD inspected/VBN by/IN the/DT student./NN',
        'Did this involve a building?', True)
    _add_stimulus_pair(
        'A/DT student/NN touched/VBD a/DT peach./NN',
        'A/DT peach/NN was/VBD touched/VBN by/IN a/DT student./NN',
        'Was there something fuzzy?', True)

    _add_stimulus_pair(
        'A/DT monkey/NN found/VBD the/DT hammer./NN',
        'The/DT hammer/NN was/VBD found/VBN by/IN a/DT monkey./NN',
        'Was this item smaller than an elephant?', True)
    _add_stimulus_pair(
        'The/DT monkey/NN kicked/VBD a/DT door./NN',
        'A/DT door/NN was/VBD kicked/VBN by/IN the/DT monkey./NN',
        'Is this something you could do?', True)
    _add_stimulus_pair(
        'The/DT monkey/NN inspected/VBD the/DT peach./NN',
        'The/DT peach/NN was/VBD inspected/VBN by/IN the/DT monkey./NN',
        'Was something blue seen?', False)
    _add_stimulus_pair(
        'A/DT monkey/NN touched/VBD a/DT school./NN',
        'A/DT school/NN was/VBD touched/VBN by/IN a/DT monkey./NN',
        'Was this item squishy?', False)

    _make_word_stimulus('dog', 'NN', 'Is it manufactured', False)
    _make_word_stimulus('doctor', 'NN', 'Is it skilled?', True)
    _make_word_stimulus('peach', 'NN', 'Can you hold it in one hand?', True)
    _make_word_stimulus('hammer', 'NN', 'Does it grow?', False)

    for t in active_sentences:
        _make_sentence_stimulus(*t)
    for t in passive_sentences:
        _make_sentence_stimulus(*t)

    return tuple(word_result), tuple(sentence_result)


krns_2_projector_delay_in_seconds = .037


_krns_2_sentence_stimuli = None
_krns_2_word_stimuli = None


class KRNS2(object):

    @staticmethod
    def _normalize(s):
        return s.strip(string.punctuation).lower() if s is not None else ''

    def __init__(self):

        global _krns_2_sentence_stimuli
        global _krns_2_word_stimuli
        if _krns_2_sentence_stimuli is None:
            _krns_2_word_stimuli, _krns_2_sentence_stimuli = make_krns_2_master_sentence_stimuli()

        self._master_sentence_stimuli = _krns_2_sentence_stimuli
        self._master_word_stimuli = _krns_2_word_stimuli

        self._min_sentence_length = None
        self._max_sentence_length = None

        self._normalized_text_to_master_sentence = dict()
        for master_stimulus in self._master_sentence_stimuli:
            words = list(master_stimulus.iter_level(Stimulus.word_level))
            if self._min_sentence_length is None:
                self._min_sentence_length = len(words)
                self._max_sentence_length = len(words)
            else:
                self._min_sentence_length = min(self._min_sentence_length, len(words))
                self._max_sentence_length = max(self._max_sentence_length, len(words))
            normalized = ''.join(map(lambda s: KRNS2._normalize(s.text), words))
            self._normalized_text_to_master_sentence[normalized] = master_stimulus

        self._normalized_text_to_master_word = dict()
        for master_stimulus in self._master_word_stimuli:
            normalized = KRNS2._normalize(master_stimulus.text)
            self._normalized_text_to_master_word[normalized] = master_stimulus

    def load_block_stimuli(self, fif_raw, word_stimuli_mat, sentence_stimuli_mat, index_block):
        block_events, is_word_block, event_load_fix_info = load_fif_block_events_krns_2(
            fif_raw, word_stimuli_mat, sentence_stimuli_mat, index_block)
        return self.match(block_events, is_word_block), event_load_fix_info

    @staticmethod
    def _label_events(event_stream):
        def _is_prompt(ev):
            return ev.stimulus == 'Y + N' or ev.stimulus == 'N + Y'

        last_event = None
        for event in event_stream:
            if last_event is None:
                assert(not _is_prompt(event))
            else:
                if _is_prompt(event):
                    assert(not _is_prompt(last_event))
                    yield ('question', (last_event,))
                elif _is_prompt(last_event):
                    yield ('prompt', (last_event,))
                else:
                    yield ('stimulus', (last_event,))
            last_event = event
        if _is_prompt(last_event):
            yield ('prompt', (last_event,))
        else:
            yield ('stimulus', (last_event,))

    def _accumulate_sentences(self, labeled_events):
        sentence_accumulator = list()
        for key, events in labeled_events:
            if key == 'stimulus':
                sentence_accumulator.extend(events)
                if len(sentence_accumulator) > self._max_sentence_length:
                    raise ValueError('Unable to match a known sentence. Current stimuli: {}'.format(
                        ' '.join([e.stimulus for e in sentence_accumulator])))
                elif len(sentence_accumulator) >= self._min_sentence_length:
                    normalized = ''.join(map(lambda e: KRNS2._normalize(e.stimulus), sentence_accumulator))
                    if normalized in self._normalized_text_to_master_sentence:
                        yield ('stimulus', sentence_accumulator)
                        sentence_accumulator = list()
            else:
                if len(sentence_accumulator) > 1:
                    raise ValueError('Unable to match a known sentence. Current stimuli: {}'.format(
                        ' '.join([e.stimulus for e in sentence_accumulator])))
                yield key, events
        if len(sentence_accumulator) > 0:
            raise ValueError('Unable to match a known sentence. Current stimuli: {}'.format(
                ' '.join([e.stimulus for e in sentence_accumulator])))

    def iterate_stimulus_events(self, event_stream, is_word_block):
        stimulus_accumulator = list()
        labeled_events = KRNS2._label_events(event_stream)
        if not is_word_block:
            labeled_events = self._accumulate_sentences(labeled_events)
        for key, events in labeled_events:
            if len(stimulus_accumulator) == 0 and key != 'stimulus':
                raise ValueError('Unable to start stimulus')
            if key == 'stimulus':
                if len(stimulus_accumulator) > 0:
                    yield stimulus_accumulator
                stimulus_accumulator = [(key, events)]
            else:
                stimulus_accumulator.append((key, events))
        if len(stimulus_accumulator) > 0:
            yield stimulus_accumulator

    # noinspection PyMethodMayBeStatic
    def _get_master_remove_attributes(self, master, stimulus_events):
        keys = set([k for k, v in stimulus_events])
        return [k for k in master if isinstance(master[k], Stimulus) and k not in keys]

    # noinspection PyMethodMayBeStatic
    def _map_additional_events(self, master_stimulus, key, events):
        if key == 'prompt':
            if len(events) != 1:
                raise ValueError('Expected exactly 1 prompt event')
            prompt_result = (
                key,
                StimulusBuilder(
                    Stimulus.word_level, attributes={Stimulus.text_attribute_name: events[0].stimulus}).make_stimulus())
            return prompt_result, None, None
        if key not in master_stimulus:
            raise ValueError('Unable to find attribute in master stimulus: {}'.format(key))
        return master_stimulus[key], None, None

    def match(self, event_stream, is_word_block):
        """
        Converts a low level event stream (i.e. Event instances extracted from a fif file/session stimuli combination)
        to a high level stimulus stream where each Stimulus has attributes and children which can themselves be Stimulus
        instances. This function also handles the
        Args:
            event_stream: The stream of Event instances to convert
            is_word_block: Whether the current event_stream is from a word block (as opposed to a sentence block)
        Returns:
            A stream of Stimulus instances
        """
        stimulus_counts = dict()
        word_counts = dict()

        stimuli = list()
        for index_stimulus, stimulus_events in enumerate(self.iterate_stimulus_events(event_stream, is_word_block)):
            master = None
            masters = list()
            delays = list()
            final_events = list()
            simple_attribute_adds = list()
            simple_attribute_deletes = list()
            for key, events in stimulus_events:

                # this is a useful place to print stuff for debugging
                # print(key, [(ev.trigger, ev.stimulus) for ev in events])

                if key == 'stimulus':
                    if is_word_block:
                        if len(events) != 1:
                            raise ValueError('Expected a single event for word block')
                        normalized = KRNS2._normalize(events[0].stimulus)
                        master = self._normalized_text_to_master_word[normalized]
                    else:
                        normalized = ''.join(map(lambda e: KRNS2._normalize(e.stimulus), events))
                        master = self._normalized_text_to_master_sentence[normalized]
                    masters.append(master)
                    adds = dict()
                    deletes = set()
                    deletes.update(self._get_master_remove_attributes(master, stimulus_events))
                    simple_attribute_adds.append(adds)
                    simple_attribute_deletes.append(deletes)
                    stimulus_count = stimulus_counts[master] if master in stimulus_counts else 0
                    stimulus_count += 1
                    stimulus_counts[master] = stimulus_count
                else:
                    assert (master is not None)
                    additional_master, additional_master_adds, additional_master_deletes = \
                        self._map_additional_events(master, key, events)
                    masters.append(additional_master)
                    if additional_master_adds is None:
                        additional_master_adds = dict()
                    if additional_master_deletes is None:
                        additional_master_deletes = set()
                    simple_attribute_adds.append(additional_master_adds)
                    simple_attribute_deletes.append(additional_master_deletes)

                simple_attribute_adds[-1][Stimulus.modality_attribute_name] = Stimulus.visual_modality
                delays.append(krns_2_projector_delay_in_seconds)
                final_events.append(events)

            stimulus = masters[0].copy_with_event_attributes(
                final_events[0],
                delays[0],
                simple_attribute_adds[0],
                simple_attribute_deletes[0],
                stimulus_counts[masters[0]],
                KRNS2._normalize,
                word_counts,
                masters[1:],
                final_events[1:],
                delays[1:],
                simple_attribute_adds[1:],
                simple_attribute_deletes[1:])
            stimuli.append(stimulus)

        return stimuli

    def flatten(self, stimuli, single_auditory_events=True):
        for stimulus in stimuli:
            if single_auditory_events and stimulus[Stimulus.modality_attribute_name] == Stimulus.auditory_modality:
                yield Event(
                    stimulus=stimulus[Stimulus.text_attribute_name],
                    duration=stimulus[Stimulus.duration_attribute_name],
                    time_stamp=stimulus[Stimulus.time_stamp_attribute_name],
                    trigger=None)
            else:
                for word_stimulus in stimulus.iter_level(Stimulus.word_level):
                    yield Event(
                        stimulus=word_stimulus[Stimulus.text_attribute_name],
                        duration=word_stimulus[Stimulus.duration_attribute_name],
                        time_stamp=word_stimulus[Stimulus.time_stamp_attribute_name],
                        trigger=None)
            if 'question' in stimulus:
                question_stimulus = stimulus['question']
                if single_auditory_events and stimulus[Stimulus.modality_attribute_name] == Stimulus.auditory_modality:
                    yield Event(
                        stimulus=question_stimulus[Stimulus.text_attribute_name],
                        duration=question_stimulus[Stimulus.duration_attribute_name],
                        time_stamp=question_stimulus[Stimulus.time_stamp_attribute_name],
                        trigger=None)
                else:
                    for word_stimulus in question_stimulus.iter_level(Stimulus.word_level):
                        yield Event(
                            stimulus=word_stimulus[Stimulus.text_attribute_name],
                            duration=word_stimulus[Stimulus.duration_attribute_name],
                            time_stamp=word_stimulus[Stimulus.time_stamp_attribute_name],
                            trigger=None)
            if 'prompt' in stimulus:
                prompt_stimulus = stimulus['prompt']
                yield Event(
                    stimulus=prompt_stimulus[Stimulus.text_attribute_name],
                    duration=prompt_stimulus[Stimulus.duration_attribute_name],
                    time_stamp=prompt_stimulus[Stimulus.time_stamp_attribute_name],
                    trigger=None)


def load_session_stimuli_block_events_krns_2(word_stimuli_mat, sentence_stimuli_mat, index_block):
    """
    Loads the output of psychtoolbox into a list of event-code, text pairs
    Args:
        word_stimuli_mat: The pyschtoolbox output for the word blocks of the experiment, usually called
            'wordBlock.mat'
        sentence_stimuli_mat: The psychtoolbox output for the sentence blocks of the experiment, usually called
            'sentenceBlock.mat'
        index_block: Which block of the experiment to load
    Returns:
        An array of event codes and a list of
    """
    is_word_block = True
    if isinstance(word_stimuli_mat, str):
        from scipy.io import loadmat
        word_stimuli_mat = loadmat(word_stimuli_mat)
        word_stimuli_data = word_stimuli_mat['experiment'].squeeze()
        if index_block >= word_stimuli_data.shape[0]:
            is_word_block = False
            index_block -= word_stimuli_data.shape[0]
            if isinstance(sentence_stimuli_mat, str):
                sentence_stimuli_mat = loadmat(sentence_stimuli_mat)
                block_stimuli_data = sentence_stimuli_mat['experiment'].squeeze()[index_block]
        else:
            block_stimuli_data = word_stimuli_data[index_block]

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
    return zip_equal(stimuli_events, block_text), is_word_block


def load_fif_block_events_krns_2(mne_raw, word_stimuli_mat, sentence_stimuli_mat, index_block):

    event_code_text_pairs, is_word_block = load_session_stimuli_block_events_krns_2(
        word_stimuli_mat, sentence_stimuli_mat, index_block)

    instruction_trigger = 1

    # filter out the 0 event codes and the instructions
    event_code_text_pairs = [
        (c, t) for c, t in event_code_text_pairs if c != 0 and c != instruction_trigger]

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

    return result, is_word_block, event_load_fix_info
