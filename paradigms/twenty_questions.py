from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mne
from collections import namedtuple
from .stimulus import Stimulus, StimulusBuilder
from six import itervalues


__all__ = ['Word_EventType', 'Question_EventType', 'Response_EventType', 'EventTypes', 'WordAndResponse',
           'QuestionEvents', 'num_20questions_words', 'Event20Questions', 'WordGrouping', 'sort_key_20questions',
           'category_counts_20questions', 'create_stimuli_20questions', 'create_stimuli_60words',
           'read_events_20questions', 'read_events_60words', 'load_block_stimuli_20questions',
           'load_block_stimuli_60words', 'sudre_perceptual_features', 'make_compute_lower_upper_bounds_20questions',
           'words_20questions']

Word_EventType = 'word'
Question_EventType = 'question'
Response_EventType = 'response'
EventTypes = [Word_EventType, Question_EventType, Response_EventType]

WordAndResponse = namedtuple('WordAndResponse', 'word_event response_event')
QuestionEvents = namedtuple('QuestionEvents', 'question_event word_response_events')

sudre_perceptual_features = (
    'Word length',
    'White pixel count',
    'Internal details',
    'Verticality',
    'Horizontalness',
    'Left-diagonalness',
    'Right-diagonalness',
    'Aspect-ratio: skinny->fat',
    'Prickiliness',
    'Line curviness',
    '3D curviness'
)

__questions_in_id_order = (
    'Is it manmade?',
    'Is it made of metal?',
    'Is it hollow?',
    'Is it hard to catch?',
    'Does it grow?',
    'Was it ever alive?',
    'Could you fit inside it?',
    'Does it have at least one hole?',
    'Can you hold it?',
    'Is it bigger than a loaf of bread?',
    'Does it live in groups?',
    'Can it keep you dry?',
    'Is part of it made of glass?',
    'Is it bigger than a car?',
    'Can you hold it in one hand?',
    'Is it manufactured?',
    'Is it bigger than a microwave oven?',
    'Is it alive?',
    'Does it have feelings?',
    'Can you pick it up?'
)

__words_in_id_order = (
    'airplane',
    'ant',
    'apartment',
    'arch',
    'arm',
    'barn',
    'bear',
    'bed',
    'bee',
    'beetle',
    'bell',
    'bicycle',
    'bottle',
    'butterfly',
    'car',
    'carrot',
    'cat',
    'celery',
    'chair',
    'chimney',
    'chisel',
    'church',
    'closet',
    'coat',
    'corn',
    'cow',
    'cup',
    'desk',
    'dog',
    'door',
    'dress',
    'dresser',
    'eye',
    'fly',
    'foot',
    'glass',
    'hammer',
    'hand',
    'horse',
    'house',
    'igloo',
    'key',
    'knife',
    'leg',
    'lettuce',
    'pants',
    'pliers',
    'refrigerator',
    'saw',
    'screwdriver',
    'shirt',
    'skirt',
    'spoon',
    'table',
    'telephone',
    'tomato',
    'train',
    'truck',
    'watch',
    'window'
)

words_20questions = tuple(__words_in_id_order)
num_20questions_words = len(__words_in_id_order)


class Event20Questions:

    default_projector_delay_in_seconds = .037

    def __init__(self,
                 time_raw_in_samples_from_first_sample,
                 first_sample,
                 times,
                 text,
                 event_type,
                 code,
                 projector_delay_in_seconds=default_projector_delay_in_seconds):
        self.time_raw_in_samples_from_first_sample = time_raw_in_samples_from_first_sample
        self.time_raw_in_seconds = times[time_raw_in_samples_from_first_sample - first_sample]
        self.text = text
        self.type = event_type
        self.code = code
        self.projector_delay_in_seconds = projector_delay_in_seconds

    @property
    def time_corrected_in_seconds(self):
        return self.time_raw_in_seconds + self.projector_delay_in_seconds


class WordGrouping:

    __size = ['small', 'medium', 'large', 'extra_large']
    __manipulable = ['low', 'high']
    __animate = ['yes', 'no']
    __category = [
        'insect',
        'animal',
        'body_part',
        'vegetable',
        'kitchen',
        'tool',
        'vehicle',
        'furniture',
        'building',
        'building_part',
        'clothing',
        'man_made'
    ]

    def __init__(self, size, manipulable, animate, category):

        self.__size = size
        self.__manipulable = manipulable
        self.__animate = animate
        self.__category = category

        if self.__size not in WordGrouping.__size:
            raise ValueError('size must be one of {0}'.format(WordGrouping.__size))

        if self.__manipulable not in WordGrouping.__manipulable:
            raise ValueError('manipulable must be one of {0}'.format(WordGrouping.__manipulable))

        if self.__animate not in WordGrouping.__animate:
            raise ValueError('animate must be one of {0}'.format(WordGrouping.__animate))

        if self.__category not in WordGrouping.__category:
            raise ValueError('category must be one of {0}'.format(WordGrouping.__category))

    @property
    def size(self):
        return self.__size

    @property
    def manipulable(self):
        return self.__manipulable

    @property
    def animate(self):
        return self.__animate

    @property
    def category(self):
        return self.__category

    @property
    def sort_key(self):
        return (
            WordGrouping.__category.index(self.category),
            WordGrouping.__animate.index(self.animate),
            WordGrouping.__size.index(self.size),
            WordGrouping.__manipulable.index(self.manipulable))


def sort_key_20questions(word):
    if word not in __word_to_grouping:
        raise ValueError('Word is not a 20questions word: {0}'.format(word))

    key = list(__word_to_grouping[word].sort_key)
    key.append(word)

    return tuple(key)


def get_word_metadata(word):
    if word not in __word_to_grouping:
        raise ValueError('Word is not a 20questions word: {0}'.format(word))
    return __word_to_grouping[word]


def category_counts_20questions():

    counts = dict()
    for grouping in itervalues(__word_to_grouping):
        if grouping.category in counts:
            counts[grouping.category] += 1
        else:
            counts[grouping.category] = 1

    return counts


# these are just for me to group the items in the matrix
# to see if there is structure. These are not official ratings
__word_to_grouping = {
    'airplane': WordGrouping(size='extra_large', manipulable='low', animate='no', category='vehicle'),
    'ant': WordGrouping(size='small', manipulable='high', animate='yes', category='insect'),
    'apartment': WordGrouping(size='large', manipulable='low', animate='no', category='building'),
    'arch': WordGrouping(size='large', manipulable='low', animate='no', category='building_part'),
    'arm': WordGrouping(size='medium', manipulable='high', animate='yes', category='body_part'),
    'barn': WordGrouping(size='extra_large', manipulable='low', animate='no', category='building'),
    'bear': WordGrouping(size='large', manipulable='low', animate='yes', category='animal'),
    'bed': WordGrouping(size='medium', manipulable='low', animate='no', category='furniture'),
    'bee': WordGrouping(size='small', manipulable='high', animate='yes', category='insect'),
    'beetle': WordGrouping(size='small', manipulable='high', animate='yes', category='insect'),
    'bell': WordGrouping(size='medium', manipulable='high', animate='no', category='man_made'),
    'bicycle': WordGrouping(size='medium', manipulable='high', animate='no', category='vehicle'),
    'bottle': WordGrouping(size='small', manipulable='high', animate='no', category='kitchen'),
    'butterfly': WordGrouping(size='small', manipulable='high', animate='yes', category='insect'),
    'car': WordGrouping(size='large', manipulable='low', animate='no', category='vehicle'),
    'carrot': WordGrouping(size='small', manipulable='high', animate='no', category='vegetable'),
    'cat': WordGrouping(size='medium', manipulable='high', animate='yes', category='animal'),
    'celery': WordGrouping(size='small', manipulable='high', animate='no', category='vegetable'),
    'chair': WordGrouping(size='medium', manipulable='high', animate='no', category='furniture'),
    'chimney': WordGrouping(size='large', manipulable='low', animate='no', category='building_part'),
    'chisel': WordGrouping(size='small', manipulable='high', animate='no', category='tool'),
    'church': WordGrouping(size='extra_large', manipulable='low', animate='no', category='building'),
    'closet': WordGrouping(size='medium', manipulable='low', animate='no', category='building_part'),
    'coat': WordGrouping(size='medium', manipulable='high', animate='no', category='clothing'),
    'corn': WordGrouping(size='small', manipulable='high', animate='no', category='vegetable'),
    'cow': WordGrouping(size='large', manipulable='low', animate='yes', category='animal'),
    'cup': WordGrouping(size='small', manipulable='high', animate='no', category='kitchen'),
    'desk': WordGrouping(size='medium', manipulable='low', animate='no', category='furniture'),
    'dog': WordGrouping(size='medium', manipulable='low', animate='no', category='animal'),
    'door': WordGrouping(size='medium', manipulable='high', animate='no', category='building_part'),
    'dress': WordGrouping(size='medium', manipulable='high', animate='no', category='clothing'),
    'dresser': WordGrouping(size='large', manipulable='low', animate='no', category='furniture'),
    'eye': WordGrouping(size='small', manipulable='low', animate='yes', category='body_part'),
    'fly': WordGrouping(size='small', manipulable='high', animate='yes', category='insect'),
    'foot': WordGrouping(size='small', manipulable='high', animate='yes', category='body_part'),
    'glass': WordGrouping(size='small', manipulable='high', animate='no', category='kitchen'),
    'hammer': WordGrouping(size='small', manipulable='high', animate='no', category='tool'),
    'hand': WordGrouping(size='small', manipulable='high', animate='no', category='body_part'),
    'horse': WordGrouping(size='large', manipulable='high', animate='yes', category='animal'),
    'house': WordGrouping(size='extra_large', manipulable='low', animate='no', category='building'),
    'igloo': WordGrouping(size='large', manipulable='low', animate='no', category='building'),
    'key': WordGrouping(size='small', manipulable='high', animate='no', category='man_made'),
    'knife': WordGrouping(size='small', manipulable='high', animate='no', category='kitchen'),
    'leg': WordGrouping(size='small', manipulable='high', animate='yes', category='body_part'),
    'lettuce': WordGrouping(size='small', manipulable='high', animate='no', category='vegetable'),
    'pants': WordGrouping(size='medium', manipulable='high', animate='no', category='clothing'),
    'pliers': WordGrouping(size='small', manipulable='high', animate='no', category='tool'),
    'refrigerator': WordGrouping(size='medium', manipulable='low', animate='no', category='man_made'),
    'saw': WordGrouping(size='small', manipulable='high', animate='no', category='tool'),
    'screwdriver': WordGrouping(size='small', manipulable='high', animate='no', category='tool'),
    'shirt': WordGrouping(size='medium', manipulable='high', animate='no', category='clothing'),
    'skirt': WordGrouping(size='medium', manipulable='high', animate='no', category='clothing'),
    'spoon': WordGrouping(size='small', manipulable='high', animate='no', category='kitchen'),
    'table': WordGrouping(size='medium', manipulable='low', animate='no', category='furniture'),
    'telephone': WordGrouping(size='small', manipulable='high', animate='no', category='man_made'),
    'tomato': WordGrouping(size='small', manipulable='high', animate='no', category='vegetable'),
    'train': WordGrouping(size='extra_large', manipulable='low', animate='no', category='vehicle'),
    'truck': WordGrouping(size='large', manipulable='low', animate='no', category='vehicle'),
    'watch': WordGrouping(size='small', manipulable='high', animate='no', category='man_made'),
    'window': WordGrouping(size='medium', manipulable='high', animate='no', category='building_part')
}


categories = tuple(sorted(set([grouping.category for grouping in itervalues(__word_to_grouping)])))


def create_stimuli_60words(block_word_events, last_time):

    max_response_duration = 1
    min_word_event_buffer = .5  # word events are marked, there should be a 500ms fixation cross before these

    word_counts = dict()

    block_stimuli = list()
    for index_word, word_event in enumerate(block_word_events):

        word_stimulus = StimulusBuilder(Stimulus.word_level)
        word_stimulus[Stimulus.position_in_parent_attribute_name] = -1
        word_stimulus[Stimulus.position_in_root_attribute_name] = -1
        word_stimulus[Stimulus.master_stimulus_index_attribute_name] = word_event.code - 1
        word_stimulus[Stimulus.sort_key_attribute_name] = sort_key_20questions(word_event.text)
        word_stimulus[Stimulus.stratification_key_attribute_name] = \
            __word_to_grouping[word_event.text].category
        word_stimulus['category'] = __word_to_grouping[word_event.text].category
        word_stimulus[Stimulus.text_attribute_name] = word_event.text

        word_count = word_counts[word_event.text] if word_event.text in word_counts else 0
        word_count += 1
        word_counts[word_event.text] = word_count
        word_stimulus[Stimulus.stimulus_count_presentation_attribute_name] = word_count
        word_stimulus[Stimulus.word_count_presentation_attribute_name] = word_count

        presentation_stimulus = StimulusBuilder(Stimulus.presentation_level, word_stimulus.copy_attributes())
        presentation_stimulus[Stimulus.position_in_root_attribute_name] = 0
        presentation_stimulus[Stimulus.time_stamp_attribute_name] = word_event.time_corrected_in_seconds
        if index_word == len(block_word_events) - 1:
            presentation_stimulus[Stimulus.duration_attribute_name] = last_time - word_event.time_corrected_in_seconds
        else:
            next_event = block_word_events[index_word + 1]
            presentation_stimulus[Stimulus.duration_attribute_name] = (
                next_event.time_corrected_in_seconds - min_word_event_buffer -
                word_event.time_corrected_in_seconds)
        presentation_stimulus[Stimulus.duration_attribute_name] = min(
            presentation_stimulus[Stimulus.duration_attribute_name], max_response_duration)
        word_stimulus.add_child(presentation_stimulus)

        block_stimuli.append(word_stimulus.make_stimulus())

    return block_stimuli


def create_stimuli_20questions(block_question_events, last_time):

    max_response_duration = 1
    min_word_event_buffer = .5  # word events are marked, there should be a 500ms fixation cross before these

    stimulus_counts = dict()
    word_counts = dict()

    block_stimuli = list()
    for index_question, question_event in enumerate(block_question_events):

        for index_word, word_response in enumerate(question_event.word_response_events):

            word_stimulus = StimulusBuilder(Stimulus.word_level)
            word_stimulus[Stimulus.position_in_parent_attribute_name] = -1
            word_stimulus[Stimulus.position_in_root_attribute_name] = -1
            word_stimulus[Stimulus.master_stimulus_index_attribute_name] = word_response.word_event.code - 1
            word_stimulus[Stimulus.sort_key_attribute_name] = sort_key_20questions(word_response.word_event.text)
            word_stimulus[Stimulus.stratification_key_attribute_name] = \
                __word_to_grouping[word_response.word_event.text].category
            word_stimulus['category'] = __word_to_grouping[word_response.word_event.text].category
            word_stimulus[Stimulus.text_attribute_name] = word_response.word_event.text
            word_stimulus[Stimulus.question_text_attribute_name] = question_event.question_event.text

            stimulus_key = question_event.question_event.text + ':' + word_response.word_event.text
            stimulus_count = stimulus_counts[stimulus_key] if stimulus_key in stimulus_counts else 0
            stimulus_count += 1
            stimulus_counts[stimulus_key] = stimulus_count
            word_stimulus[Stimulus.stimulus_count_presentation_attribute_name] = stimulus_count

            word_count = \
                word_counts[word_response.word_event.text] if word_response.word_event.text in word_counts else 0
            word_count += 1
            word_counts[word_response.word_event.text] = word_count
            word_stimulus[Stimulus.word_count_presentation_attribute_name] = word_count

            presentation_stimulus = StimulusBuilder(Stimulus.presentation_level, word_stimulus.copy_attributes())
            presentation_stimulus[Stimulus.position_in_root_attribute_name] = 0
            presentation_stimulus[Stimulus.time_stamp_attribute_name] = \
                word_response.word_event.time_corrected_in_seconds
            if word_response.response_event is None:
                if index_word == len(question_event.word_response_events) - 1:
                    if index_question == len(block_question_events) - 1:
                        presentation_stimulus[Stimulus.duration_attribute_name] = (
                            last_time - word_response.word_event.time_corrected_in_seconds)
                    else:
                        next_event = block_question_events[index_question + 1].word_response_events[0].word_event
                        presentation_stimulus[Stimulus.duration_attribute_name] = (
                            next_event.time_corrected_in_seconds - min_word_event_buffer -
                            word_response.word_event.time_corrected_in_seconds)
                else:
                    presentation_stimulus[Stimulus.duration_attribute_name] = (
                        question_event.word_response_events[index_word + 1].word_event.time_corrected_in_seconds -
                        min_word_event_buffer - word_response.word_event.time_corrected_in_seconds)
                presentation_stimulus[Stimulus.duration_attribute_name] = min(
                    presentation_stimulus[Stimulus.duration_attribute_name], max_response_duration)
            else:
                presentation_stimulus[Stimulus.duration_attribute_name] = (
                    word_response.response_event.time_corrected_in_seconds -
                    word_response.word_event.time_corrected_in_seconds)
            word_stimulus.add_child(presentation_stimulus)

            if word_response.response_event is not None:
                response_stimulus = StimulusBuilder(Stimulus.presentation_level, word_stimulus.copy_attributes())
                response_stimulus[Stimulus.text_attribute_name] = word_response.response_event.text
                response_stimulus[Stimulus.position_in_parent_attribute_name] = 1
                response_stimulus[Stimulus.position_in_root_attribute_name] = 1
                response_stimulus[Stimulus.time_stamp_attribute_name] = \
                    word_response.response_event.time_corrected_in_seconds

                if index_word == len(question_event.word_response_events) - 1:
                    if index_question == len(block_question_events) - 1:
                        response_stimulus[Stimulus.duration_attribute_name] = (
                            last_time - word_response.response_event.time_corrected_in_seconds)
                    else:
                        next_event = block_question_events[index_question + 1].word_response_events[0].word_event
                        response_stimulus[Stimulus.duration_attribute_name] = (
                            next_event.time_corrected_in_seconds - min_word_event_buffer -
                            word_response.response_event.time_corrected_in_seconds)
                else:
                    response_stimulus[Stimulus.duration_attribute_name] = (
                        question_event.word_response_events[index_word + 1].word_event.time_corrected_in_seconds -
                        min_word_event_buffer - word_response.response_event.time_corrected_in_seconds)
                response_stimulus[Stimulus.duration_attribute_name] = min(
                    response_stimulus[Stimulus.duration_attribute_name], max_response_duration)
                word_stimulus.add_child(response_stimulus)

            block_stimuli.append(word_stimulus.make_stimulus())

    return block_stimuli


def read_events_60words(
        mne_raw_obj,
        projector_delay_in_seconds=Event20Questions.default_projector_delay_in_seconds,
        verbose=None):

    if verbose is None:
        # default to info since this was the old behavior
        verbose = 'INFO'

    if isinstance(mne_raw_obj, type('')):
        raw_obj = None
        try:
            try:
                raw_obj = mne.io.Raw(mne_raw_obj, add_eeg_ref=False, verbose=verbose)
            except TypeError: # in new version of mne, add_eeg_ref is gone
                raw_obj = mne.io.Raw(mne_raw_obj, verbose=verbose)
            return read_events_60words(
                raw_obj, projector_delay_in_seconds=projector_delay_in_seconds, verbose=verbose)
        finally:
            if raw_obj is not None:
                raw_obj.close()

    max_word_id = len(__words_in_id_order)

    # From mne documentation, return of find_events:
    # All events that were found. The first column contains the event time in samples and the third column contains the
    # event id. For output = 'onset' or 'step', the second column contains the value of the stim channel immediately
    # before the event/step. For output = 'offset', the second column contains the value of the stim channel after
    # the event offset.
    eve = mne.find_events(
        mne_raw_obj, stim_channel='STI101', shortest_event=1, uint_cast=True, min_duration=.005, verbose=verbose)
    index_time_column = 0
    index_event_column = 2

    event_list = list()

    for index_event in range(eve.shape[0]):

        if eve[index_event, index_event_column] > max_word_id:
            raise ValueError('Unknown event code: {}'.format(eve[index_event, index_event_column]))

        word_event = Event20Questions(
            time_raw_in_samples_from_first_sample=eve[index_event, index_time_column],
            first_sample=mne_raw_obj.first_samp,
            times=mne_raw_obj.times,
            text=__words_in_id_order[eve[index_event, index_event_column] - 1],
            event_type=Word_EventType,
            code=eve[index_event, index_event_column],
            projector_delay_in_seconds=projector_delay_in_seconds)

        event_list.append(word_event)

    return event_list


def read_events_20questions(
        mne_raw_obj,
        projector_delay_in_seconds=Event20Questions.default_projector_delay_in_seconds,
        verbose=None):

    if verbose is None:
        # default to info since this was the old behavior
        verbose = 'INFO'

    is_log_info = verbose == 'INFO' or verbose == 'DEBUG' or (isinstance(verbose, bool) and verbose)

    if isinstance(mne_raw_obj, type('')):
        raw_obj = None
        try:
            try:
                raw_obj = mne.io.Raw(mne_raw_obj, add_eeg_ref=False, verbose=verbose)
            except TypeError:  # in newer version of mne, add_eeg_ref is gone
                raw_obj = mne.io.Raw(mne_raw_obj, verbose=verbose)
            return read_events_20questions(
                raw_obj, projector_delay_in_seconds=projector_delay_in_seconds, verbose=verbose)
        finally:
            if raw_obj is not None:
                raw_obj.close()

    max_word_id = len(__words_in_id_order)

    # From mne documentation, return of find_events:
    # All events that were found. The first column contains the event time in samples and the third column contains the
    # event id. For output = 'onset' or 'step', the second column contains the value of the stim channel immediately
    # before the event/step. For output = 'offset', the second column contains the value of the stim channel after
    # the event offset.
    eve = mne.find_events(
        mne_raw_obj, stim_channel='STI101', shortest_event=1, uint_cast=True, min_duration=.005, verbose=verbose)
    index_time_column = 0
    index_event_column = 2

    index_event = 0
    question_event_list = list()

    this_question = None
    this_word_and_response = None
    words_left = set()

    while index_event < eve.shape[0]:
        if eve[index_event, index_event_column] > max_word_id:
            if this_word_and_response is None or this_question is None:
                raise ValueError('Response event occurred without a question id and word id set')

            if this_word_and_response is not this_question.word_response_events[-1]:
                raise RuntimeError('Code issue, expected this_word_and_response to be the '
                                   'last event in this_question.word_response_events')

            # hopefully there is a way to identify yes/no from the response code, in which case we
            # can replace the text here with 'yes' or 'no'
            this_word_and_response = WordAndResponse(
                word_event=this_word_and_response.word_event,
                response_event=Event20Questions(
                    time_raw_in_samples_from_first_sample=eve[index_event, index_time_column],
                    times=mne_raw_obj.times,
                    first_sample=mne_raw_obj.first_samp,
                    text='response',
                    event_type=Response_EventType,
                    code=eve[index_event, index_event_column],
                    projector_delay_in_seconds=projector_delay_in_seconds))
            this_question.word_response_events[-1] = this_word_and_response

            # skip all events that are response events between this response event and the next non-response event
            while (index_event < (eve.shape[0] - 1)) and eve[index_event + 1, index_event_column] > max_word_id:
                index_event += 1
        else:
            if len(words_left) == 0 or this_question is None:

                if this_question is not None:
                    if len(words_left) != 0:
                        raise ValueError('Question {0} ({1}) missing words {2}'.format(
                            this_question.question_event.code, this_question.question_event.text, words_left))

                    if is_log_info:
                        print('Done processing question {0}'.format(this_question.question_event.code))

                this_word_and_response = None
                if (eve[index_event, index_event_column] - 1 >= len(__questions_in_id_order)
                   or eve[index_event, index_event_column] - 1 < 0):
                    raise ValueError('Bad question id: {} at {} (first sample is {})'.format(
                        eve[index_event, index_event_column],
                        eve[index_event, index_time_column],
                        mne_raw_obj.first_samp))
                this_question = QuestionEvents(
                    question_event=Event20Questions(
                        time_raw_in_samples_from_first_sample=eve[index_event, index_time_column],
                        first_sample=mne_raw_obj.first_samp,
                        times=mne_raw_obj.times,
                        text=__questions_in_id_order[eve[index_event, index_event_column] - 1],
                        event_type=Question_EventType,
                        code=eve[index_event, index_event_column],
                        projector_delay_in_seconds=projector_delay_in_seconds),
                    word_response_events=list()
                )

                question_event_list.append(this_question)
                words_left = set(__words_in_id_order)
            else:
                this_word_and_response = WordAndResponse(
                    word_event=Event20Questions(
                        time_raw_in_samples_from_first_sample=eve[index_event, index_time_column],
                        first_sample=mne_raw_obj.first_samp,
                        times=mne_raw_obj.times,
                        text=__words_in_id_order[eve[index_event, index_event_column] - 1],
                        event_type=Word_EventType,
                        code=eve[index_event, index_event_column],
                        projector_delay_in_seconds=projector_delay_in_seconds),
                    response_event=None)
                try:
                    words_left.remove(this_word_and_response.word_event.text)
                except:
                    raise ValueError(
                        'Something went wrong. Word {0} ({1}) appeared twice in question {2} ({3})'.format(
                            this_word_and_response.word_event.code,
                            this_word_and_response.word_event.text,
                            this_question.question_event.code,
                            this_question.question_event.text))
                this_question.word_response_events.append(this_word_and_response)

        index_event += 1

    if is_log_info:
        print('Done processing question {0}'.format(this_question.question_event.code))

    return question_event_list


def _validate(question_event_list):

    for question in question_event_list:
        if len(question.word_response_events) != len(__words_in_id_order):
            raise ValueError('Words are missing from question {0} ({1})'.format(
                question.question_event.code,
                question.question_event.text))


def load_block_stimuli_60words(mne_raw, verbose=None):

    block_events = read_events_60words(mne_raw, verbose=verbose)
    return create_stimuli_60words(block_events, mne_raw.times[-1])


def load_block_stimuli_20questions(mne_raw, verbose=None):

    block_events = read_events_20questions(mne_raw, verbose=verbose)
    return create_stimuli_20questions(block_events, mne_raw.times[-1])


# noinspection PyUnusedLocal
def make_compute_lower_upper_bounds_20questions(recording_tuple):

    def compute_lower_upper_bounds(mne_raw):
        stimuli = load_block_stimuli_20questions(mne_raw, verbose=False)

        return [(
            stimulus,
            stimulus[Stimulus.time_stamp_attribute_name],
            stimulus[Stimulus.time_stamp_attribute_name] + stimulus[Stimulus.duration_attribute_name]
        ) for stimulus in stimuli]

    return compute_lower_upper_bounds
