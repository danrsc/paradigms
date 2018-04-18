from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import string

from .stimulus import Stimulus, StimulusBuilder, add_word_stimuli_to_parent, make_root_stimulus_builder
from .generic_paradigm import GenericParadigm, make_map_to_stimulus_fn, EventGroupSpec


__all__ = ['make_pass_act_2_master_stimuli', 'PassAct2']


def make_pass_act_2_master_stimuli():

    result = list()

    def _make_stimulus(
            tagged_sentence,
            is_active,
            tagged_question,
            is_yes_correct,
            question_code):

        def _process_tagged(parent, s):
            add_word_stimuli_to_parent(parent, s)
            for index_word_stimulus, word_stimulus in enumerate(parent.children):
                word_stimulus[Stimulus.position_in_root_attribute_name] = index_word_stimulus

        sb = make_root_stimulus_builder(Stimulus.sentence_level, len(result))
        sb['is_active'] = is_active
        sb['is_yes_correct'] = is_yes_correct
        sb['question_code'] = question_code
        sb['question'] = StimulusBuilder(Stimulus.sentence_level, attributes=sb.copy_attributes())

        _process_tagged(sb, tagged_sentence)
        _process_tagged(sb['question'], tagged_question)

        result.append(sb.make_stimulus())

    # Active Sentences
    _make_stimulus('The/DT man/NN watched/VBD the/DT girl./NN',
                   True, 'Did/VBD he/PRP see/VB a/DT person?/NN', True, 'A_Qhe1Y')
    _make_stimulus('The/DT girl/NN liked/VBD the/DT boy./NN',
                   True, 'Did/VBD she/PRP hate/VB this/DT person?/NN', False, 'A_Qshe1N')
    _make_stimulus('The/DT woman/NN despised/VBN the/DT man./NN',
                   True, 'Was/VBD he/PRP likable?/JJ', False, 'A_Qhe2N')
    _make_stimulus('The/DT boy/NN encouraged/VBD the/DT woman./NN',
                   True, 'Did/VBD someone/NN talk/VB to/TO her?/PRP', True, 'A_Qshe2Y')

    _make_stimulus('The/DT man/NN liked/VBD the/DT boy./NN',
                   True, 'Were/VBD they/PRP alive?/JJ', True, 'A_QmtheyY')
    _make_stimulus('The/DT girl/NN despised/VBN the/DT man./NN',
                   True, 'Did/VBD she/PRP like/VB the/DT person?/NN', False, 'A_Qshe1N')
    _make_stimulus('The/DT woman/NN encouraged/VBD the/DT girl./NN',
                   True, 'Were/VBD they/PRP friendly?/JJ', True, 'A_QftheyY')
    _make_stimulus('The/DT boy/NN watched/VBD the/DT woman./NN',
                   True, 'Was/VBD he/PRP blind?/JJ', False, 'A_Qhe1N')

    _make_stimulus('The/DT man/NN despised/VBN the/DT woman./NN',
                   True, 'Did/VB he/PRP want/VBP to/TO be/VB friends?/NNS', False, 'A_Qhe1N')
    _make_stimulus('The/DT girl/NN encouraged/VBD the/DT man./NN',
                   True, 'Was/VBD she/PRP sleeping?/NN', True, 'A_Qshe1N')
    _make_stimulus('The/DT woman/NN watched/VBD the/DT boy./NN',
                   True, 'Was/VBD he/PRP visible?/JJ', True, 'A_Qhe2Y')
    _make_stimulus('The/DT boy/NN liked/VBD the/DT girl./NN',
                   True, 'Was/VBD she/PRP dead?/JJ', False, 'A_Qshe2N')

    _make_stimulus('The/DT man/NN encouraged/VBD the/DT woman./NN',
                   True, 'Was/VBD she/PRP complimented?/JJ', True, 'A_Qshe2Y')
    _make_stimulus('The/DT girl/NN watched/VBD the/DT boy./NN',
                   True, 'Was/VBD he/PRP gone?/VBN', False, 'A_Qhe2N')
    _make_stimulus('The/DT woman/NN liked/VBD the/DT girl./NN',
                   True, 'Did/VBD they/PRP know/VB each/DT other?/JJ', True, 'A_QftheyY')
    _make_stimulus('The/DT boy/NN despised/VBN the/DT man./NN',
                   True, 'Were/VBD there/EX negative/JJ feelings?/NNS', True, 'A_QmtheyY')

    # Passive Sentences
    _make_stimulus('The/DT girl/NN was/VBD watched/VBD by/IN the/DT man./NN',
                   False, 'Did/VBD he/PRP see/VB a/DT person?/NN', True, 'P_Qhe1Y')
    _make_stimulus('The/DT boy/NN was/VBD liked/VBD by/IN the/DT girl./NN',
                   False, 'Did/VBD she/PRP hate/VB this/DT person?/NN', False, 'P_Qshe1N')
    _make_stimulus('The/DT man/NN was/VBD despised/VBN by/IN the/DT woman./NN',
                   False, 'Was/VBD he/PRP likable?/JJ', False, 'P_Qhe2N')
    _make_stimulus('The/DT woman/NN was/VBD encouraged/VBD by/IN the/DT boy./NN',
                   False, 'Did/VBD someone/NN talk/VB to/TO her?/PRP', True, 'P_Qshe2Y')

    _make_stimulus('The/DT boy/NN was/VBD liked/VBD by/IN the/DT man./NN',
                   False, 'Were/VBD they/PRP alive?/JJ', True, 'P_Qmthey')
    _make_stimulus('The/DT man/NN was/VBD despised/VBN by/IN the/DT girl./NN',
                   False, 'Did/VBD she/PRP like/VB the/DT person?/NN', False, 'P_Qshe1N')
    _make_stimulus('The/DT girl/NN was/VBD encouraged/VBD by/IN the/DT woman./NN',
                   False, 'Were/VBD they/PRP friendly?/JJ', True, 'P_QftheyY')
    _make_stimulus('The/DT woman/NN was/VBD watched/VBD by/IN the/DT boy./NN',
                   False, 'Was/VBD he/PRP blind?/JJ', False, 'P_Qhe1N')

    _make_stimulus('The/DT woman/NN was/VBD despised/VBN by/IN the/DT man./NN',
                   False, 'Did/VB he/PRP want/VBP to/TO be/VB friends?/NNS', False, 'P_Qhe1N')
    _make_stimulus('The/DT man/NN was/VBD encouraged/VBD by/IN the/DT girl./NN',
                   False, 'Was/VBD she/PRP sleeping?/NN', False, 'P_Qshe1N')
    _make_stimulus('The/DT boy/NN was/VBD watched/VBD by/IN the/DT woman./NN',
                   False, 'Was/VBD he/PRP visible?/JJ', True, 'P_Qhe2Y')
    _make_stimulus('The/DT girl/NN was/VBD liked/VBD by/IN the/DT boy./NN',
                   False, 'Was/VBD she/PRP dead?/JJ', False, 'P_Qshe2N')

    _make_stimulus('The/DT woman/NN was/VBD encouraged/VBD by/IN the/DT man./NN',
                   False, 'Was/VBD she/PRP complemented?/JJ', True, 'P_Qshe2Y')
    _make_stimulus('The/DT boy/NN was/VBD watched/VBD by/IN the/DT girl./NN',
                   False, 'Was/VBD he/PRP gone?/VBN', False, 'P_Qhe2N')
    _make_stimulus('The/DT girl/NN was/VBD liked/VBD by/IN the/DT woman./NN',
                   False, 'Did/VBD they/PRP know/VB each/DT other?/JJ', True, 'P_QftheyY')
    _make_stimulus('The/DT man/NN was/VBD despised/VBN by/IN the/DT boy./NN',
                   False, 'Were/VBD there/EX negative/JJ feelings?/NNS', True, 'P_QmtheyY')

    return tuple(result)


_pass_act_2_stimuli = None


class PassAct2(GenericParadigm):

    def __init__(self):

        global _pass_act_2_stimuli
        if _pass_act_2_stimuli is None:
            _pass_act_2_stimuli = make_pass_act_2_master_stimuli()

        def _normalize(s):
            return s.strip(string.punctuation).lower() if s is not None else ''

        self._master_stimuli = _pass_act_2_stimuli
        self._map_primary = make_map_to_stimulus_fn(self._master_stimuli, _normalize)

        super(PassAct2, self).__init__(
            instruction_trigger=1,
            event_group_specs=[
                EventGroupSpec(2, 'stimulus'),
                EventGroupSpec(4, 'question', allow_repeated=True),
                EventGroupSpec([254, 255], 'prompt')],
            primary_stimulus_key='stimulus',
            normalize=_normalize)

    @property
    def master_stimuli(self):
        return self._master_stimuli

    def _is_auditory_trigger(self, trigger):
        return False

    def _infer_auditory_events(self, master_stimulus, key, auditory_event):
        raise RuntimeError('This function should never be called on {}'.format(type(self)))

    def _map_primary_events(self, key, events):
        return self._map_primary(events)

    def _map_additional_events(self, master_stimulus, key, events):
        if key == 'prompt':
            if len(events) != 1:
                raise ValueError('Expected exactly 1 prompt event')
            return (
                key,
                StimulusBuilder(
                    Stimulus.word_level, attributes={Stimulus.text_attribute_name: events[0].stimulus}).make_stimulus())
        if key not in master_stimulus:
            raise ValueError('Unable to find attribute in master stimulus: {}'.format(key))
        return master_stimulus[key]
