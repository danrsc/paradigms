from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import string

from .stimulus import Stimulus, StimulusBuilder, add_word_stimuli_to_parent, make_root_stimulus_builder, Event
from .generic_paradigm import GenericParadigm, make_map_to_stimulus_fn, EventGroupSpec


__all__ = ['make_pass_act_3_master_stimuli', 'PassAct3', 'PassAct3Aud', 'PassAct3AudVis']


def make_pass_act_3_master_stimuli():

    result = list()

    def _make_stimulus(
            tagged_sentence,
            is_active,
            tagged_question,
            is_yes_correct):

        def _process_tagged(parent, s):
            add_word_stimuli_to_parent(parent, s)
            for index_word_stimulus, word_stimulus in enumerate(parent.children):
                word_stimulus[Stimulus.position_in_root_attribute_name] = index_word_stimulus

        sb = make_root_stimulus_builder(Stimulus.sentence_level, len(result))
        sb['is_active'] = is_active
        sb['is_yes_correct'] = is_yes_correct
        sb['question'] = StimulusBuilder(Stimulus.sentence_level, attributes=sb.copy_attributes())

        _process_tagged(sb, tagged_sentence)
        _process_tagged(sb['question'], tagged_question)

        result.append(sb.make_stimulus())

    # Active Sentences
    _make_stimulus('The/DT man/NN kicked/VBD the/DT girl./NN', True, 'Did/VBD he/PRP see/VB someone?/NN', True),
    _make_stimulus('The/DT girl/NN helped/VBD the/DT boy./NN', True, 'Did/VBD she/PRP do/VB nothing?/NN', False),
    _make_stimulus('The/DT woman/NN approached/VBD the/DT man./NN', True, 'Was/VBD he/PRP seen?/VBN', False),
    _make_stimulus('The/DT boy/NN punched/VBD the/DT woman./NN', True, 'Was/VBD she/PRP attacked?/VBN', True),

    _make_stimulus('The/DT man/NN kicked./VBD', True, 'Was/VBD he/PRP sleeping?/VBG', False),
    _make_stimulus('The/DT girl/NN helped./VBD', True, 'Did/VBD she/PRP act?/VB', True),
    _make_stimulus('The/DT woman/NN approached./VBD', True, 'Did/VBD she/PRP move?/VB', True),
    _make_stimulus('The/DT boy/NN punched./VBD', True, 'Was/VBD he/PRP still?/RB', False),

    _make_stimulus('The/DT girl/NN kicked/VBD the/DT man./NN', True, 'Did/VBD she/PRP behave/VB nicely?/RB', False),
    _make_stimulus('The/DT boy/NN helped/VBD the/DT girl./NN', True, 'Did/VBD he/PRP do/VB something?/NN', True),
    _make_stimulus('The/DT man/NN approached/VBD the/DT woman./NN', True, 'Was/VBD she/PRP visible?/JJ', True),
    _make_stimulus('The/DT woman/NN punched/VBD the/DT boy./NN', True, 'Was/VBD he/PRP safe?/JJ', False),

    _make_stimulus('The/DT girl/NN kicked./VBD', True, 'Was/VBD she/PRP sleeping?/VBG', False),
    _make_stimulus('The/DT boy/NN helped./VBD', True, 'Did/VBD he/PRP act?/VB', True),
    _make_stimulus('The/DT man/NN approached./VBD', True, 'Did/VBD he/PRP move?/VB', True),
    _make_stimulus('The/DT woman/NN punched./VBD', True, 'Was/VBD she/PRP sleeping?/VBG', False),

    # Passive Sentences
    _make_stimulus('The/DT girl/NN was/VBD kicked/VBD by/IN the/DT man./NN',
                   False, 'Did/VBD he/PRP see/VB someone?/NN', True),
    _make_stimulus('The/DT boy/NN was/VBD helped/VBD by/IN the/DT girl./NN',
                   False, 'Did/VBD she/PRP do/VB nothing?/NN', False),
    _make_stimulus('The/DT man/NN was/VBD approached/VBD by/IN the/DT woman./NN',
                   False, 'Was/VBD he/PRP seen?/VBN', False),
    _make_stimulus('The/DT woman/NN was/VBD punched/VBD by/IN the/DT boy./NN',
                   False, 'Was/VBD she/PRP attacked?/VBN', True),

    _make_stimulus('The/DT girl/NN was/VBD kicked./VBD', False, 'Was/VBD she/PRP hurt?/JJ', True),
    _make_stimulus('The/DT boy/NN was/VBD helped./VBD', False, 'Was/VBD he/PRP ignored?/VBN', False),
    _make_stimulus('The/DT man/NN was/VBD approached./VBD', False, 'Was/VBD he/PRP visible?/JJ', True),
    _make_stimulus('The/DT woman/NN was/VBD punched./VBD', False, 'Was/VBD she/PRP unharmed?/JJ', False),

    _make_stimulus('The/DT man/NN was/VBD kicked/VBD by/IN the/DT girl./NN',
                   False, 'Did/VBD she/PRP behave/VB nicely?/RB', False),
    _make_stimulus('The/DT girl/NN was/VBD helped/VBD by/IN the/DT boy./NN',
                   False, 'Did/VBD he/PRP do/VB something?/NN', True),
    _make_stimulus('The/DT woman/NN was/VBD approached/VBD by/IN the/DT man./NN',
                   False, 'Was/VBD she/PRP visible?/JJ', True),
    _make_stimulus('The/DT boy/NN was/VBD punched/VBD by/IN the/DT woman./NN',
                   False, 'Was/VBD he/PRP safe?/JJ', False),

    _make_stimulus('The/DT man/NN was/VBD kicked./VBD', False, 'Was/VBD he/PRP hurt?/JJ', True),
    _make_stimulus('The/DT girl/NN was/VBD helped./VBD', False, 'Was/VBD she/PRP ignored?/VBN', False),
    _make_stimulus('The/DT woman/NN was/VBD approached./VBD', False, 'Was/VBD she/PRP visible?/JJ', True),
    _make_stimulus('The/DT boy/NN was/VBD punched./VBD', False, 'Was/VBD he/PRP unharmed?/JJ', False)

    return tuple(result)


_pass_act_3_stimuli = None


class _PassAct3Base(GenericParadigm):

    def __init__(
            self,
            stimulus_triggers,
            question_mark_triggers,
            question_triggers,
            prompt_triggers,
            trigger_key_fn=None):

        global _pass_act_3_stimuli
        if _pass_act_3_stimuli is None:
            _pass_act_3_stimuli = make_pass_act_3_master_stimuli()

        def _normalize(s):
            return s.strip(string.punctuation).lower() if s is not None else ''

        self._master_stimuli = _pass_act_3_stimuli
        self._map_primary = make_map_to_stimulus_fn(self._master_stimuli, _normalize)

        super(_PassAct3Base, self).__init__(
            instruction_trigger=1,
            event_group_specs=[
                EventGroupSpec(stimulus_triggers, 'stimulus'),
                EventGroupSpec(question_mark_triggers, 'question_mark'),
                EventGroupSpec(question_triggers, 'question'),
                EventGroupSpec(prompt_triggers, 'prompt')],
            primary_stimulus_key='stimulus',
            normalize=_normalize,
            trigger_key_fn=trigger_key_fn)

    @property
    def master_stimuli(self):
        return self._master_stimuli

    def _is_auditory_trigger(self, trigger):
        raise NotImplementedError('{} does not implement _is_auditory_trigger'.format(type(self)))

    def _infer_auditory_events(self, master_stimulus, key, auditory_event):
        word_duration = 0.3
        word_spacing_duration = 0.2

        # now we need to artificially insert the experimental stimuli for each word
        inferred_events = list()
        for index_word, word_stimulus in enumerate(master_stimulus.iter_level(Stimulus.word_level)):
            inferred_events.append(
                Event(stimulus=word_stimulus.text,
                      duration=word_duration + word_spacing_duration,
                      trigger=auditory_event.trigger if index_word == 0 else 2,
                      time_stamp=auditory_event.time_stamp + index_word * (word_duration + word_spacing_duration)))
        return inferred_events

    def _map_primary_events(self, key, events):
        return self._map_primary(events), None, None

    def _map_additional_events(self, master_stimulus, key, events):
        if key == 'prompt' or key == 'question_mark':
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
            if 'question_mark' in stimulus:
                question_mark_stimulus = stimulus['question_mark']
                yield Event(
                    stimulus=question_mark_stimulus[Stimulus.text_attribute_name],
                    duration=question_mark_stimulus[Stimulus.duration_attribute_name],
                    time_stamp=question_mark_stimulus[Stimulus.time_stamp_attribute_name],
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


class PassAct3(_PassAct3Base):

    def __init__(self):
        super(PassAct3, self).__init__(
            stimulus_triggers=3, question_mark_triggers=None, question_triggers=4, prompt_triggers=[254, 255])

    def _infer_auditory_events(self, master_stimulus, key, auditory_event):
        raise RuntimeError('This function should never be called on {}'.format(type(self)))

    def _is_auditory_trigger(self, trigger):
        return False


class PassAct3Aud(_PassAct3Base):

    def __init__(self):

        def _trigger_key_fn(event):
            if event.stimulus == '?':
                return event.trigger + 100
            return event.trigger

        super(PassAct3Aud, self).__init__(
            stimulus_triggers=5, question_mark_triggers=104, question_triggers=4, prompt_triggers=[254, 255],
            trigger_key_fn=_trigger_key_fn)

    def iterate_stimulus_events(self, event_stream):
        for stimulus_events in super(PassAct3Aud, self).iterate_stimulus_events(event_stream):
            modified_events = list()
            for key, events in stimulus_events:
                if key == 'question_mark':
                    if len(events) != 1:
                        raise ValueError('Expected question_mark to consist of a single event')
                    # manufacture an event with a different trigger
                    mark_event = Event(
                        stimulus=events[0].stimulus,
                        duration=events[0].duration,
                        trigger=events[0].trigger + 100,
                        time_stamp=events[0].time_stamp)
                    modified_events.append((key, [mark_event]))
                else:
                    modified_events.append((key, events))
            yield modified_events

    def _is_auditory_trigger(self, trigger):
        return trigger < 100


class PassAct3AudVis(_PassAct3Base):

    def __init__(self):

        def _trigger_key_fn(event):
            if event.stimulus == '?':
                return event.trigger + 100
            return event.trigger

        super(PassAct3AudVis, self).__init__(
            stimulus_triggers=[5, 15], question_mark_triggers=[104, 114], question_triggers=[4, 14],
            prompt_triggers=[254, 255], trigger_key_fn=_trigger_key_fn)

    def iterate_stimulus_events(self, event_stream):
        for stimulus_events in super(PassAct3AudVis, self).iterate_stimulus_events(event_stream):
            modified_events = list()
            for key, events in stimulus_events:
                if key == 'question_mark':
                    if len(events) != 1:
                        raise ValueError('Expected question_mark to consist of a single event')
                    # manufacture an event with a different trigger
                    mark_event = Event(
                        stimulus=events[0].stimulus,
                        duration=events[0].duration,
                        trigger=events[0].trigger + 100,
                        time_stamp=events[0].time_stamp)
                    modified_events.append((key, [mark_event]))
                else:
                    modified_events.append((key, events))
            yield modified_events

    def _is_auditory_trigger(self, trigger):
        return trigger in {14, 15}
