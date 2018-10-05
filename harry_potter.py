import numpy
import mne
from .stimulus import Stimulus, make_root_stimulus_builder


__all__ = ['create_stimuli_harry_potter']


def create_stimuli_harry_potter(mne_raw, harry_potter_event_text_path, block, presentation_delay_in_seconds=0.037):
    stimuli = list(_iterate_stimuli_text_from_harry_potter_event_text(harry_potter_event_text_path))

    index_sample_column = 0
    index_prev_id_column = 1
    index_event_id_column = 2

    trigger_first_plus = 1
    trigger_content = 30

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        fif_events = mne.find_events(
            mne_raw, stim_channel='STI101', shortest_event=1, uint_cast=True, min_duration=0.000, verbose=False)

    # filter out the first plus, which has trigger == 1
    assert(fif_events[0, index_event_id_column] == 1)
    len_with_trigger_1 = len(fif_events)
    fif_events = fif_events[fif_events[:, index_event_id_column] != trigger_first_plus]
    assert(len(fif_events) == len_with_trigger_1 - 1)
    assert(stimuli[0] == '+')
    stimuli = stimuli[1:]

    # if the leading stimulus is not a '+' and the first event trigger is a 40, that's ok. Seems like a mistake in the
    # trigger id
    if stimuli[0].strip() != '+' and (
            fif_events[0, index_event_id_column] != trigger_content):
        fif_events[0, index_event_id_column] = trigger_content

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

    if len(fif_events) != len(stimuli):

        if len(fif_events) == len(stimuli) - 1:
            # there are a couple of cases where the final '+' did not get recorded in the events
            if stimuli[-1] == '+':
                stimuli = stimuli[:-1]
        else:
            raise ValueError('Mismatched number of events. Fif: {}, text: {}'.format(len(fif_events), len(stimuli)))

    # check the alignment by looking at where the '+' fall
    indicator_plus = fif_events[:, index_event_id_column] != trigger_content
    for expect_plus, s in zip(indicator_plus, stimuli):
        assert((s == '+') == expect_plus)

    time_indices = fif_events[:, index_sample_column] - mne_raw.first_samp
    time_stamps = mne_raw.times[time_indices]
    durations = numpy.concatenate([numpy.diff(time_stamps), numpy.array([mne_raw.times[-1] - time_stamps[-1]])])

    block_stimuli = list()
    for index_word, (s, d, t) in enumerate(zip(stimuli, durations, time_stamps)):
        # make the index unique by adding 10000 * block
        sb = make_root_stimulus_builder(Stimulus.word_level, index_word + block * 10000)
        sb[Stimulus.text_attribute_name] = s
        sb[Stimulus.time_stamp_attribute_name] = t + presentation_delay_in_seconds
        sb[Stimulus.duration_attribute_name] = d
        block_stimuli.append(sb.make_stimulus())

    return block_stimuli


def _iterate_stimuli_text_from_harry_potter_event_text(stimuli_path):

    with open(stimuli_path, 'rt') as stimuli_file:
        for line in stimuli_file:
            line = line.strip()
            if len(line) == 0:
                continue
            fields = line.split('\t')
            index_stim = None
            for index_field, field in enumerate(fields):
                if field == 'stim':
                    index_stim = index_field
                    break
            if index_stim is None:
                raise ValueError('Unable to find stimulus field')
            if index_stim >= len(fields) - 1:
                raise ValueError('Bad field specification. Expected at least one field after \'stim\'')
            stimulus = fields[index_stim + 1]
            yield stimulus.strip()
