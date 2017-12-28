import os
import inspect
from six import iteritems
from brain_gen.core import flags
from .stimulus import Stimulus
from . import tagged_file_reader


__all__ = ['create_master_stimuli', 'MasterStimuliPaths', 'map_recording_to_session_stimuli_path']


def create_master_stimuli(master_stimuli_path):

    master_stimuli_modify_time = os.path.getmtime(master_stimuli_path)
    tagged_columns_per_stimulus, configuration = tagged_file_reader.read_tagged_file(master_stimuli_path)

    stimulus_column = 'stimulus'

    # first determine if we should use words, sentences, or passages as the root Stimulus
    root_level = Stimulus.word_level
    for tagged_columns in tagged_columns_per_stimulus:
        if stimulus_column in tagged_columns:
            stimulus_value = tagged_columns[stimulus_column]
            if len(stimulus_value) > 1:
                root_level = Stimulus.sentence_level
            if any([(
                Stimulus.is_end_substimulus_attribute_name in tagged_word and
                tagged_word[Stimulus.is_end_substimulus_attribute_name]
            ) for tagged_word in stimulus_value]):
                root_level = Stimulus.passage_level
                break

    root_stimuli = list()

    for index_stimulus, tagged_columns in enumerate(tagged_columns_per_stimulus):
        root_attributes = dict()
        root_attributes[Stimulus.position_in_parent_attribute_name] = -1
        root_attributes[Stimulus.position_in_root_attribute_name] = -1
        root_attributes[Stimulus.master_stimulus_index_attribute_name] = index_stimulus
        root_attributes[Stimulus.sort_key_attribute_name] = index_stimulus
        # default this to index_stimulus, allow it to be overridden by column in file
        root_attributes[Stimulus.stratification_key_attribute_name] = index_stimulus
        for column_name, column_value in iteritems(tagged_columns):
            if column_name == stimulus_column:
                continue
            root_attributes[column_name] = column_value

        if stimulus_column in tagged_columns:
            stimulus_value = tagged_columns[stimulus_column]
            if root_level == Stimulus.word_level:
                root_attributes.update(stimulus_value[0])
                root_stimuli.append(Stimulus(root_level, root_attributes))
            else:
                root_text = ' '.join([tagged_word[Stimulus.text_attribute_name] for tagged_word in stimulus_value])
                root_attributes[Stimulus.text_attribute_name] = root_text
                root_stimuli.append(Stimulus(root_level, root_attributes))
                if root_level == Stimulus.passage_level:

                    def __sentence_iter(passage_tagged_words):
                        sentence = list()
                        for word in passage_tagged_words:
                            sentence.append(word)
                            if (Stimulus.is_end_substimulus_attribute_name in word and
                                    word[Stimulus.is_end_substimulus_attribute_name]):
                                yield sentence
                                sentence = list()
                        if len(sentence) > 0:
                            yield sentence

                    index_word = 0
                    for index_sentence, sentence_tagged_words in enumerate(__sentence_iter(stimulus_value)):
                        sentence_attributes = dict()
                        sentence_text = ' '.join([tagged_word[Stimulus.text_attribute_name]
                                                  for tagged_word in sentence_tagged_words])
                        sentence_attributes[Stimulus.text_attribute_name] = sentence_text
                        sentence_attributes[Stimulus.position_in_root_attribute_name] = index_sentence
                        sentence_attributes[Stimulus.position_in_parent_attribute_name] = index_sentence
                        sentence_attributes[Stimulus.master_stimulus_index_attribute_name] = index_stimulus
                        sentence_stimulus = Stimulus(
                            Stimulus.sentence_level, sentence_attributes, parent=root_stimuli[-1])
                        for index_in_sentence, tagged_word in enumerate(sentence_tagged_words):
                            word_attributes = dict(tagged_word)
                            word_attributes[Stimulus.position_in_root_attribute_name] = index_word
                            index_word += 1
                            word_attributes[Stimulus.position_in_parent_attribute_name] = index_in_sentence
                            word_attributes[Stimulus.master_stimulus_index_attribute_name] = index_stimulus
                            Stimulus(Stimulus.word_level, word_attributes, parent=sentence_stimulus)

                elif root_level == Stimulus.sentence_level:
                    for index_word, tagged_word in enumerate(stimulus_value):
                        word_attributes = dict(tagged_word)
                        word_attributes[Stimulus.position_in_parent_attribute_name] = index_word
                        word_attributes[Stimulus.position_in_root_attribute_name] = index_word
                        word_attributes[Stimulus.master_stimulus_index_attribute_name] = index_stimulus
                        Stimulus(Stimulus.word_level, word_attributes, parent=root_stimuli[-1])

                else:
                    raise NotImplementedError('Unknown root level: {0}'.format(root_level))
        else:
            root_stimuli.append(Stimulus(root_level, root_attributes))

    return root_stimuli, configuration, master_stimuli_modify_time


class MasterStimuliPaths:

    base_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    passive_active_3 = os.path.join(base_path, 'passive_active_3.txt')
    passive_active_2 = os.path.join(base_path, 'passive_active_2.txt')

    def __init__(self):
        pass


def is_audio_experiment(experiment_name):
    return experiment_name.lower() == 'passact3aud'


def map_recording_to_session_stimuli_path(recording_tuple):
    file_name = flags().relative_session_stimuli_path_format.format(**recording_tuple)
    return os.path.join(flags().data_root, file_name)
