import itertools
import string
import mne
import numpy
from brain_gen.core import zip_equal, flags
from .stimulus import Event, StimulusEvents, Stimulus
from .master_stimuli import is_audio_experiment, map_recording_to_session_stimuli_path, \
    MasterStimuliPaths, create_master_stimuli


__all__ = ['split_events', 'match_time_series_to_master', 'load_fif_block_events', 'load_block_stimuli',
           'make_compute_lower_upper_bounds_from_master_stimuli']


def split_events(
        block_events,
        stimulus_onset_trigger,
        substimulus_onset_triggers,
        start_post_stimulus_triggers, # for example, start of question
        allow_repeat_post_stimulus_triggers=False):

    stimuli = list()
    sub_stimuli = None
    current_accumulator = list()
    is_post_stimulus = False
    post_stimulus = None
    for meta in block_events:
        if meta.trigger == stimulus_onset_trigger:
            if len(current_accumulator) > 0:
                if is_post_stimulus:
                    post_stimulus = current_accumulator
                else:
                    if sub_stimuli is None:
                        sub_stimuli = list()
                    sub_stimuli.append(current_accumulator)
                current_accumulator = list()
            is_post_stimulus = False
            if sub_stimuli is not None or post_stimulus is not None:
                stimuli.append(StimulusEvents(sub_stimuli, post_stimulus))
            sub_stimuli = None
            post_stimulus = None
            current_accumulator.append(meta)
        elif substimulus_onset_triggers is not None and meta.trigger in substimulus_onset_triggers:
            if is_post_stimulus:
                raise ValueError('Did not expect substimulus onset trigger '
                                 'between end stimulus trigger and stimulus onset trigger')
            if len(current_accumulator) > 0:
                if sub_stimuli is None:
                    sub_stimuli = list()
                sub_stimuli.append(current_accumulator)
                current_accumulator = list()
            current_accumulator.append(meta)
        elif start_post_stimulus_triggers is not None and meta.trigger in start_post_stimulus_triggers:
            if is_post_stimulus:
                if allow_repeat_post_stimulus_triggers:
                    # in some experiments (e.g. PassAct2, the post stimulus trigger was also used as the word
                    # trigger for every word in the question, so if the allow option is set, we just treat
                    # these as normal word triggers
                    current_accumulator.append(meta)
                else:
                    raise ValueError('Did not expect multiple start-post-stimulus trigger '
                                     'between start-post-stimulus trigger and stimulus onset trigger')
            else:
                is_post_stimulus = True
                if len(current_accumulator) > 0:
                    if sub_stimuli is None:
                        sub_stimuli = list()
                    sub_stimuli.append(current_accumulator)
                    current_accumulator = list()
                current_accumulator.append(meta)
        else:
            current_accumulator.append(meta)
    if len(current_accumulator) > 0:
        if is_post_stimulus:
            post_stimulus = current_accumulator
        else:
            if sub_stimuli is None:
                sub_stimuli = list()
            sub_stimuli.append(current_accumulator)
        if sub_stimuli is not None or post_stimulus is not None:
            stimuli.append(StimulusEvents(sub_stimuli, post_stimulus))
    return stimuli


def match_time_series_to_master(
        experiment_name,
        stimuli_events,
        master_stimuli,
        is_first_block,
        normalize=None):

    if is_audio_experiment(experiment_name):
        return _match_time_series_to_master_for_audio(
            experiment_name, stimuli_events, master_stimuli, is_first_block, normalize)

    stimulus_delay_in_seconds = .037

    def __default_normalize(s):
        return s.strip(string.punctuation).lower() if s is not None else ''

    if normalize is None:
        normalize = __default_normalize

    normalized_text_to_master = dict()
    for master_stimulus in master_stimuli:
        normalized = ' '.join(map(lambda stimulus: normalize(stimulus.text),
                                  master_stimulus.iter_level(Stimulus.word_level)))
        normalized_text_to_master[normalized] = master_stimulus

    stimulus_counts = dict()
    word_counts = dict()

    full_meta = list()
    for index_stimulus, stimulus_events in enumerate(stimuli_events):
        normalized = ' '.join(map(lambda meta: normalize(meta.stimulus), itertools.chain(*stimulus_events.sub_stimuli)))
        if normalized not in normalized_text_to_master:
            if is_first_block and index_stimulus == 0:
                pass  # we assume that these are the instructions
            else:
                raise KeyError('Unable to match presentation to master stimuli: {0}'.format(normalized))
        else:
            match = normalized_text_to_master[normalized]
            stimulus_count = stimulus_counts[normalized] if normalized in stimulus_counts else 0
            stimulus_count += 1
            stimulus_counts[normalized] = stimulus_count
            full_meta.append(match.copy_with_event_attributes(
                stimulus_events, normalize, stimulus_count, word_counts, stimulus_delay_in_seconds))

    return full_meta


def _match_time_series_to_master_for_audio(
        experiment_name,
        stimuli_events,
        master_stimuli,
        word_duration,
        word_spacing_duration,
        normalize=None):

    stimulus_delay_in_seconds = .024

    def __default_normalize(s):
        return s.strip(string.punctuation).lower() if s is not None else ''

    if normalize is None:
        normalize = __default_normalize

    normalized_text_to_master = dict()
    for master_stimulus in master_stimuli:
        normalized = ''.join(map(lambda s: normalize(s.text), master_stimulus.iter_level(Stimulus.word_level)))
        normalized_text_to_master[normalized] = master_stimulus

    stimulus_counts = dict()
    word_counts = dict()

    full_meta = list()
    for index_stimulus, stimulus_events in enumerate(stimuli_events):
        normalized = ''.join(map(lambda meta: normalize(meta.stimulus), itertools.chain(*stimulus_events.sub_stimuli)))
        if normalized not in normalized_text_to_master:
            raise KeyError('Unable to match presentation to master stimuli: {0}'.format(normalized))
        else:
            match = normalized_text_to_master[normalized]
            stimulus_count = stimulus_counts[normalized] if normalized in stimulus_counts else 0
            stimulus_count += 1
            stimulus_counts[normalized] = stimulus_count

            if len(stimulus_events.sub_stimuli) > 1 or len(stimulus_events.sub_stimuli[0]) > 1:
                raise ValueError('Expected only one stimulus event')

            stimulus_event = stimulus_events.sub_stimuli[0][0]

            # now we need to artificially insert the experimental stimuli for each word
            inferred_events = list()
            for index_word, word_stimulus in enumerate(match.iter_level(Stimulus.word_level)):
                inferred_events.append(
                    Event(word_stimulus.text, word_duration + word_spacing_duration, stimulus_event.trigger,
                          stimulus_event.time_stamp + index_word * (word_duration + word_spacing_duration)))

            stimulus_events.sub_stimuli[0] = inferred_events

            full_meta.append(match.copy_with_event_attributes(
                stimulus_events, normalize, stimulus_count, word_counts, stimulus_delay_in_seconds))

    return full_meta


def load_fif_block_events(
        fif_block_file, session_stimuli_mat, index_block, stimulus_event_filter=None, fif_event_filter=None):

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
    if stimulus_event_filter is None:
        is_included = [e != 0 for e in stimuli_events]
    else:
        is_included = stimulus_event_filter(stimuli_events)
    block_text = [s for s, b in zip_equal(block_text, is_included) if b]
    stimuli_events = numpy.array([e for e, b in zip_equal(stimuli_events, is_included) if b])

    fif_events = mne.find_events(
        fif_block_file, stim_channel='STI101', shortest_event=1, uint_cast=True, min_duration=.005, verbose=False)

    index_sample_column = 0
    index_event_id_column = 2

    if fif_event_filter is not None:
        indicator_fif_events = fif_event_filter(fif_events[:, index_event_id_column])
        fif_events = fif_events[indicator_fif_events]

    if not numpy.array_equal(stimuli_events, fif_events[:, index_event_id_column]):
        print(stimuli_events)
        print(fif_events[:, index_event_id_column])
        raise ValueError('Events in fif_block_file do not match events in session_stimuli_mat')

    time_indices = fif_events[:, index_sample_column] - fif_block_file.first_samp
    time_stamps = fif_block_file.times[time_indices]
    durations = numpy.concatenate([numpy.diff(time_stamps), numpy.array([fif_block_file.times[-1] - time_stamps[-1]])])

    result_events = list()
    for index in range(len(block_text)):
        # some mild clean up of the punctuation is done here to match what was done in preprocessed files
        clean_text = block_text[index].strip().replace(u'\u2019', "'").replace(u'\u201c', '"').replace(u'\u201d', '"')
        result_events.append(Event(
            stimulus=clean_text,
            duration=durations[index],
            trigger=fif_events[index, index_event_id_column],
            time_stamp=time_stamps[index]
        ))

    return result_events, fif_block_file.times


def load_block_stimuli(master_stimuli, configuration, experiment_name, fif_raw, session_stimuli_mat, index_block):

    if is_audio_experiment(experiment_name):
        return _load_block_stimuli_for_audio(
            master_stimuli, configuration, experiment_name, fif_raw, session_stimuli_mat, index_block)

    allow_repeat_post_stimulus_triggers = False
    instruction_trigger = configuration['instruction_trigger']
    if 'allow_repeat_post_stimulus_triggers' in configuration:
        allow_repeat_post_stimulus_triggers = configuration['allow_repeat_post_stimulus_triggers']

    def stimuli_event_filter(stimuli_events):
        return [e != 0 and e != instruction_trigger for e in stimuli_events]

    def fif_event_filter(fif_events):
        return fif_events != instruction_trigger

    block_events, times = load_fif_block_events(
        fif_raw, session_stimuli_mat, index_block,
        stimulus_event_filter=stimuli_event_filter, fif_event_filter=fif_event_filter)

    # stimuli_events_ = split_events(
    #     block_events,
    #     configuration['startStimulusTrigger'],
    #     [configuration['startSentenceTrigger']],
    #     configuration['endStimulusTriggers'])

    stimuli_events_ = split_events(
        block_events,
        configuration['start_stimulus_trigger'],
        [],  # [configuration['startSentenceTrigger']],
        configuration['end_stimulus_triggers'],
        allow_repeat_post_stimulus_triggers)

    return match_time_series_to_master(experiment_name, stimuli_events_, master_stimuli, index_block == 0)


def _load_block_stimuli_for_audio(
        master_stimuli, configuration, experiment_name, fif_raw, session_stimuli_mat, index_block):

    instruction_trigger = configuration['instruction_trigger']
    question_trigger = configuration['question_trigger']
    yes_no_triggers = configuration['yes_no_triggers']
    word_duration = configuration['word_duration']
    word_spacing_duration = configuration['word_spacing_duration']

    def stimulus_event_filter(stimuli_events):
        # not sure how robust this is. It seems like the question triggers are repeated in session_stimuli_mat
        # once for '?', and once for the audio file. However, in the fif file only a single question trigger
        # appears
        # It also seems that the yes_no_triggers do not show up in the fif file at all?
        result = list()
        last_non_zero = None
        for e in stimuli_events:
            if e == 0:
                result.append(False)
            else:
                if e == instruction_trigger:
                    result.append(False)
                elif e in yes_no_triggers:
                    result.append(False)
                elif last_non_zero == question_trigger and e == question_trigger:
                    result.append(False)
                else:
                    result.append(True)
                last_non_zero = e
        return result

    block_events, times = load_fif_block_events(
        fif_raw, session_stimuli_mat, index_block, stimulus_event_filter=stimulus_event_filter)

    stimuli_events_ = split_events(
        block_events,
        configuration['start_stimulus_trigger_audio'],
        [],  # [configuration['startSentenceTrigger']],
        configuration['end_stimulus_triggers'])

    return _match_time_series_to_master_for_audio(
        experiment_name, stimuli_events_, master_stimuli, word_duration, word_spacing_duration)


def make_compute_lower_upper_bounds_from_master_stimuli(recording_tuple):

    def compute_lower_upper_bounds(mne_raw):
        session_stimuli_path = map_recording_to_session_stimuli_path(recording_tuple)

        master_stimuli_path = getattr(MasterStimuliPaths, flags().master_stimuli, None)
        if master_stimuli_path is None:
            raise ValueError('Unknown master stimuli: {}'.format(flags().master_stimuli))
        master_stimuli, configuration, _ = create_master_stimuli(master_stimuli_path)

        stimuli = load_block_stimuli(
            master_stimuli, configuration, recording_tuple.experiment, mne_raw, session_stimuli_path,
            int(recording_tuple.recording) - 1)

        return [(
            stimulus,
            stimulus[Stimulus.time_stamp_attribute_name],
            stimulus[Stimulus.time_stamp_attribute_name] + stimulus[Stimulus.duration_attribute_name]
        ) for stimulus in stimuli]

    return compute_lower_upper_bounds
