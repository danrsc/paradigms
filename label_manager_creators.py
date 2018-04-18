from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import string
import re
import numpy
from brain_gen import LabelIdManager, LabelEmbeddingManager, flags, Option, SingleItemSpec
from .stimulus import Stimulus
from .master_stimuli import MasterStimuli
from .twenty_questions import words_20questions


__all__ = [
    'word2vec_key',
    'make_word2vec_label_manager',
    'make_word_len_label_manager',
    'make_word_len_numpy_label_manager_20questions',
    'make_fold_key_manager_20questions',
    'make_label_manager_from_npz',
    'make_master_stimuli_label_manager',
    'make_master_stimuli_fold_key_manager',
    'LabelEmbeddingManagerNamedComponent',
    'words_having_part_of_speech',
    'first_non_to_be_verb',
    'first_noun',
    'stimulus_to_first_noun_text',
    'stimulus_to_first_noun_time',
    'stimulus_to_first_non_to_be_verb_text',
    'stimulus_to_first_non_to_be_verb_time',
    'stimulus_to_full_text',
    'stimulus_to_last_word_time']


def get_options():
    return [
        Option(
            'word2vec_path',
            'The path to the word2vec binary vector file',
            is_required=False,
            parse_spec=SingleItemSpec(parsed_type='path')),
        Option(
            'master_stimuli',
            'The name of the master stimuli to use, for example \'passive_active_3\'',
            is_required=False),
        Option(
            'map_stimulus_to_label_key',
            'A function (module.function), which maps from a Stimulus instance to a key from which to make the label',
            is_required=False,
            parse_spec=SingleItemSpec(parsed_type='attribute')),
        Option(
            'map_stimulus_to_fold_key',
            'A function (module.function), which maps from a Stimulus instance to a key for leave one out '
            'cross-validation',
            is_required=False,
            parse_spec=SingleItemSpec(parsed_type='attribute')),
        Option(
            'map_stimulus_to_time_0',
            'A function which returns a timepoint that should be considered time 0 for the stimulus. This '
            'allows control over how the stimuli are aligned to each other',
            parse_spec=SingleItemSpec(parsed_type='attribute'),
            is_required=False),
        Option(
            'relative_session_stimuli_path_format',
            'A format string using keywords which are properties of the recording tuple. When combined with the data '
            'root and populated with recording tuple values, this should be the path to a matlab file containing '
            'information about how the events correspond to stimuli within a subject\'s recording session',
            is_required=False),
        Option(
            'label_embedding_type',
            'If specified, this type of embedding will be used to represent the label. If None, then a one-hot '
            'representation will be used for the label',
            is_required=False),
    ]


noun_tags = {'NN', 'NNS', 'NNP', 'NNPS'}
verb_tags = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
to_be_verbs = {'be', 'am', 'is', 'are', 'being', 'was', 'were', 'been'}
punctuation_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))


def words_having_part_of_speech(stimulus, allowed_pos):
    for word_stimulus in stimulus.iter_level(Stimulus.word_level):
        if word_stimulus['POS'] in allowed_pos:
            yield word_stimulus


def first_non_to_be_verb(stimulus, is_raise=True):
    for word_stimulus in words_having_part_of_speech(stimulus, verb_tags):
        lower_text = word_stimulus.text.lower()
        if lower_text not in to_be_verbs:
            return word_stimulus
    if is_raise:
        raise ValueError('No non-to-be-verbs found in stimulus: {}'.format(stimulus))
    return None


def first_noun(stimulus, is_raise=True):
    for word_stimulus in words_having_part_of_speech(stimulus, noun_tags):
        return word_stimulus
    if is_raise:
        raise ValueError('No nouns found in stimulus: {}'.format(stimulus))
    return None


def stimulus_to_first_noun_text(stimulus):
    return punctuation_regex.sub('', first_noun(stimulus).text).lower()


def stimulus_to_first_noun_time(stimulus):
    return first_noun(stimulus)[Stimulus.time_stamp_attribute_name]


def stimulus_to_first_non_to_be_verb_text(stimulus):
    return punctuation_regex.sub('', first_non_to_be_verb(stimulus).text).lower()


def stimulus_to_first_non_to_be_verb_time(stimulus):
    return first_non_to_be_verb(stimulus)[Stimulus.time_stamp_attribute_name]


def stimulus_to_full_text(stimulus):
    return stimulus.text.lower()


def stimulus_to_last_word_time(stimulus):
    result = None
    for stimulus_word in stimulus.iter_level(Stimulus.word_level):
        result = stimulus_word
    return result[Stimulus.time_stamp_attribute_name]


def make_master_stimuli_label_manager():
    master_stimuli = MasterStimuli.stimuli_from_name(flags().master_stimuli)
    map_to_key = flags().map_stimulus_to_label_key
    if map_to_key is None:
        raise ValueError('map_stimulus_to_label_key option must be specified to use this label manager')
    keys = list(sorted(set([map_to_key(s) for s in master_stimuli])))
    embedding_type = flags().label_embedding_type
    if embedding_type is None:
        return LabelIdManager(keys, map_to_key)
    elif embedding_type == 'word2vec':
        embedding_dict = {}
        import gensim
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(flags().word2vec_path, binary=True)
        for k in keys:
            embedding_dict[k] = word2vec[k]
        del word2vec
        return LabelEmbeddingManager(embedding_dict, map_to_key)
    else:
        raise ValueError('Unknown embedding type: {}'.format(embedding_type))


def make_master_stimuli_fold_key_manager():
    map_to_fold = flags().map_stimulus_to_fold_key
    if map_to_fold is None:
        return None
    master_stimuli = MasterStimuli.stimuli_from_name(flags().master_stimuli)
    fold_keys = list(sorted(set([map_to_fold(s) for s in master_stimuli])))
    return LabelIdManager(fold_keys, map_to_fold)


def make_fold_key_manager_20questions():
    map_to_fold = flags().map_stimulus_to_fold_key
    if map_to_fold is None:
        return None
    fold_keys = list(sorted(set([map_to_fold(s) for s in words_20questions])))
    return LabelIdManager(fold_keys, map_to_fold)


def make_word_len_numpy_label_manager_20questions():
    def map_to_key(stimulus):
        return stimulus.text.lower()
    return make_word_len_label_manager(words_20questions, map_to_key=map_to_key, is_numpy_only=True)


def make_word_len_label_manager(stimuli, map_to_key=None, is_numpy_only=False):
    if map_to_key is None:
        def lower_key(s):
            return s.lower()
        map_to_key = lower_key
    embedding_dict = dict()
    for stimulus in stimuli:
        embedding_dict[stimulus.lower()] = numpy.full((1,), fill_value=len(stimulus))
    return LabelEmbeddingManager(embedding_dict, map_to_key=map_to_key, is_numpy_only=is_numpy_only)


def word2vec_key(text):
    word_list = text.split()
    return '_'.join([s.lower() for s in word_list])


def make_word2vec_label_manager(word2vec_path, stimuli):
    import gensim
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    embedding_dict = dict()
    for stimulus in stimuli:
        key = word2vec_key(stimulus)
        embedding_dict[key] = word2vec[key]
    del word2vec
    return LabelEmbeddingManager(embedding_dict, map_to_key=word2vec_key)


def _lower_key(s):
    return s.lower()


def make_label_manager_from_npz(
        path,
        stimuli=None,
        vectors_name='vectors',
        stimuli_name='stimuli',
        features_name='features',
        excluded_features=None,
        make_key=_lower_key,
        rescale=None):
    loaded = numpy.load(path)
    if vectors_name not in loaded:
        raise ValueError('Unable to find vectors in loaded file: {}'.format(path))
    if stimuli_name not in loaded:
        raise ValueError('Unable to find stimuli in loaded file: {}'.format(path))
    vectors = loaded[vectors_name]
    features = loaded[features_name].tolist() if features_name in loaded else None

    if excluded_features is not None:
        if features is None:
            raise ValueError('Unable to find features in loaded file {}. Exclusion not possible'.format(path))
        indicator_excluded = numpy.full(len(features), False)
        excluded_features = set([s.lower() for s in excluded_features])
        final_features = list()
        for index_feature, feature in enumerate(features):
            if feature.lower() in excluded_features:
                indicator_excluded[index_feature] = True
            else:
                final_features.append(feature)
        if numpy.count_nonzero(indicator_excluded) > 0:
            vectors = vectors[:, ~indicator_excluded]
            features = final_features

    if rescale is not None:
        vectors = vectors * rescale

    embedding_dict = dict()

    loaded_stimuli = loaded[stimuli_name].tolist()
    if stimuli is None:
        for stimulus, vector in zip(loaded_stimuli, vectors):
            key = make_key(stimulus)
            embedding_dict[key] = vector
    else:
        stimuli2index = dict([(make_key(s), i) for i, s in enumerate(loaded_stimuli)])
        for stimulus in stimuli:
            key = make_key(stimulus)
            embedding_dict[key] = vectors[stimuli2index[key]]

    if features is None:
        return LabelEmbeddingManager(embedding_dict, map_to_key=make_key)
    else:
        return LabelEmbeddingManagerNamedComponent(embedding_dict, features, map_to_key=make_key)


class LabelEmbeddingManagerNamedComponent(LabelEmbeddingManager):

    def __init__(self, embedding_dict, component_names, map_to_key=None, sort_key=None, is_numpy_only=False):
        LabelEmbeddingManager.__init__(self, embedding_dict, map_to_key=map_to_key, sort_key=sort_key,
                                       is_numpy_only=is_numpy_only)
        self._component_names = component_names

    @property
    def component_names(self):
        return list(self._component_names)
