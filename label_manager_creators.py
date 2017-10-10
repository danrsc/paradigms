import numpy
from brain_gen import LabelEmbeddingManager


__all__ = [
    'word2vec_key',
    'make_word2vec_label_manager',
    'make_label_manager_from_npz',
    'LabelEmbeddingManagerNamedComponent']


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

    def __init__(self, embedding_dict, component_names, map_to_key=None, sort_key=None):
        LabelEmbeddingManager.__init__(self, embedding_dict, map_to_key=map_to_key, sort_key=sort_key)
        self._component_names = component_names

    @property
    def component_names(self):
        return list(self._component_names)
