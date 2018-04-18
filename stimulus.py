from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from brain_gen.core import zip_equal


__all__ = [
    'Event',
    'BlockEvents',
    'BlockStimuli',
    'StimulusEvents',
    'Stimulus',
    'StimulusBuilder',
    'add_word_stimuli_to_parent',
    'add_sentence_stimuli_to_parent',
    'make_root_stimulus_builder']


class Event(object):

    def __init__(self, stimulus, duration, trigger, time_stamp):
        self._stimulus = stimulus
        self._duration = duration
        self._trigger = trigger
        self._time_stamp = time_stamp

    @property
    def stimulus(self):
        return self._stimulus

    @property
    def duration(self):
        return self._duration

    @property
    def trigger(self):
        return self._trigger

    @property
    def time_stamp(self):
        return self._time_stamp

    def __str__(self):
        return 'Event(stimulus={stimulus}, time_stamp={time_stamp}, duration={duration}, trigger={trigger})'.format(
            stimulus=self.stimulus, time_stamp=self.time_stamp, duration=self.duration, trigger=self.trigger)


class BlockEvents(object):

    def __init__(self, block_id, events, times, data_path):
        self.block_id = block_id
        self.block_events = events
        self.times = times
        self.data_path = data_path


class BlockStimuli(object):

    def __init__(self, block_id, stimuli, times, data_path, modify_time):
        self.block_id = block_id
        self.stimuli = stimuli
        self.times = times
        self.data_path = data_path
        self.modify_time = modify_time


class StimulusEvents(object):

    def __init__(self, sub_stimuli, post_stimulus):
        self.sub_stimuli = sub_stimuli
        self.post_stimulus = post_stimulus


class Stimulus(object):

    passage_level = 'passage'
    sentence_level = 'sentence'
    word_level = 'word'
    presentation_level = 'presentation_response'

    auditory_modality = 'auditory'
    visual_modality = 'visual'

    text_attribute_name = 'text'
    part_of_speech_attribute_name = 'part_of_speech'
    position_in_parent_attribute_name = 'position_in_parent'
    position_in_root_attribute_name = 'position_in_root'
    master_stimulus_index_attribute_name = 'master_stimulus_index'
    sort_key_attribute_name = 'sort_key'
    stratification_key_attribute_name = 'stratification_key'
    stimulus_count_presentation_attribute_name = 'stimulus_count_presentation'
    word_count_presentation_attribute_name = 'word_count_presentation'
    time_stamp_attribute_name = 'time_stamp'
    duration_attribute_name = 'duration'
    modality_attribute_name = 'modality'
    question_text_attribute_name = 'question_text'  # for 20 questions data

    def __init__(self, level, attributes, children, parent):
        self._level = level
        self._attributes = dict(attributes) if attributes is not None else dict()
        self._parent = parent
        self._children = tuple(children) if children is not None else ()

    @property
    def text(self):
        return self[Stimulus.text_attribute_name] if Stimulus.text_attribute_name in self else None

    @property
    def level(self):
        return self._level

    @property
    def children(self):
        return self._children

    @property
    def parent(self):
        return self._parent

    def __contains__(self, item):
        return item in self._attributes

    def __iter__(self):
        return iter(self._attributes)

    def __getitem__(self, attribute_name):
        return self._attributes[attribute_name]

    def __str__(self):
        return self.text

    def num_nodes_at_level(self, level):
        count = 0
        for child in self._children:
            count += child.num_nodes_at_level(level)
        if level == self.level:
            count += 1
        return count

    def iter_level(self, level):
        if self._level == level:
            yield self
        else:
            for child in self._children:
                for yielded in child.iter_level(level):
                    yield yielded

    def copy(self):
        result = Stimulus(self.level, self._attributes, None, None)
        for key in result:
            if isinstance(result[key], Stimulus):
                result._attributes[key] = result[key].copy()
                # noinspection PyProtectedMember
                result[key]._parent = result
        children = list()
        for child in self.children:
            children.append(child.copy())
            # noinspection PyProtectedMember
            children[-1]._parent = result
        result._children = tuple(children)
        return result

    def copy_with_event_attributes(
            self,
            primary_stimulus_events,
            primary_stimulus_presentation_delay,
            primary_attributes_to_set,
            primary_attributes_to_remove,
            stimulus_count,
            normalize,
            word_counts,
            additional_matches,
            additional_stimulus_events,
            additional_stimulus_presentation_delays,
            additional_stimulus_attributes_to_set,
            additional_stimulus_attributes_to_remove):

        if self.parent is not None:
            raise RuntimeError('Can only call copy_with_event_attributes on root stimulus node')

        def _set_times(master_stimulus_copy, corresponding_events, presentation_delay):
            for stimulus_word, word_event in zip_equal(
                    master_stimulus_copy.iter_level(Stimulus.word_level), corresponding_events):
                # noinspection PyProtectedMember
                stimulus_word._attributes[Stimulus.time_stamp_attribute_name] = \
                    word_event.time_stamp + presentation_delay
                # noinspection PyProtectedMember
                stimulus_word._attributes[Stimulus.duration_attribute_name] = word_event.duration
            # noinspection PyProtectedMember
            master_stimulus_copy._propagate_time()

        match_keys = None
        if additional_matches is not None:
            match_keys = list()
            for match in additional_matches:
                if isinstance(match, tuple):
                    match_keys.append(match[0])
                else:
                    found = False
                    for attribute_name in self:
                        if self[attribute_name] == match:
                            match_keys.append(attribute_name)
                            found = True
                            break
                    if not found:
                        raise ValueError('Invalid match in additional_matches')

        root_copy = self.copy()
        root_copy._attributes[Stimulus.stimulus_count_presentation_attribute_name] = stimulus_count

        if match_keys is not None:
            for match_key, match, events, delay, to_remove, to_set in zip_equal(
                    match_keys, additional_matches, additional_stimulus_events,
                    additional_stimulus_presentation_delays, additional_stimulus_attributes_to_remove,
                    additional_stimulus_attributes_to_set):
                if isinstance(match, tuple):
                    # this is a Stimulus not currently in our attributes
                    stimulus = match[1].copy()
                    _set_times(stimulus, events, delay)
                    root_copy._attributes[match_key] = stimulus
                else:
                    _set_times(root_copy._attributes[match_key], events, delay)
                if to_remove is not None:
                    for attr in to_remove:
                        # noinspection PyProtectedMember
                        del root_copy._attributes[match_key]._attributes[attr]
                if to_set is not None:
                    # noinspection PyProtectedMember
                    root_copy._attributes[match_key]._attributes.update(to_set)

        _set_times(root_copy, primary_stimulus_events, primary_stimulus_presentation_delay)

        for index_word, word_stimulus in enumerate(root_copy.iter_level(Stimulus.word_level)):
            normalized_text = normalize(word_stimulus.text)
            word_count = word_counts[normalized_text] if normalized_text in word_counts else 0
            word_count += 1
            word_counts[normalized_text] = word_count
            word_stimulus._attributes[Stimulus.word_count_presentation_attribute_name] = word_count

        if primary_attributes_to_remove is not None:
            for attribute_name in primary_attributes_to_remove:
                del root_copy._attributes[attribute_name]

        if primary_attributes_to_set is not None:
            root_copy._attributes.update(primary_attributes_to_set)

        return root_copy

    def _propagate_time(self):
        if self.children is not None and len(self.children) > 0:
            for child in self.children:
                # noinspection PyProtectedMember
                child._propagate_time()

            start_time = self.children[0][Stimulus.time_stamp_attribute_name] \
                if Stimulus.time_stamp_attribute_name in self.children[0] else None
            end_time = self.children[-1][Stimulus.time_stamp_attribute_name] \
                if Stimulus.time_stamp_attribute_name in self.children[-1] else None
            end_duration = self.children[-1][Stimulus.duration_attribute_name] \
                if Stimulus.duration_attribute_name in self.children[-1] else None
            end_time = end_time + end_duration if end_time is not None and end_duration is not None else None
            duration = end_time - start_time if end_time is not None and start_time is not None else None
            if start_time is not None:
                self._attributes[Stimulus.time_stamp_attribute_name] = start_time
            if duration is not None:
                self._attributes[Stimulus.duration_attribute_name] = duration


class StimulusBuilder(Stimulus):

    def __init__(self, level, attributes=None):
        super(StimulusBuilder, self).__init__(level, attributes, None, None)

    def make_stimulus(self):
        result = Stimulus(self.level, self._attributes, None, None)
        for key in result:
            if isinstance(result[key], StimulusBuilder):
                result._attributes[key] = result[key].make_stimulus()
                # noinspection PyProtectedMember
                result[key]._parent = result
        children = list()
        for child in self.children:
            children.append(child.make_stimulus())
            # noinspection PyProtectedMember
            children[-1]._parent = result
        result._children = tuple(children)
        result._propagate_time()
        return result

    def __setitem__(self, key, value):
        # noinspection PyProtectedMember
        self._attributes[key] = value

    def update(self, __m, **kwargs):
        # noinspection PyProtectedMember
        self._attributes.update(__m, **kwargs)

    def copy_attributes(self):
        # noinspection PyProtectedMember
        return dict(self._attributes)

    def add_child(self, child):
        child[Stimulus.position_in_parent_attribute_name] = len(self.children)
        child._parent = self
        # noinspection PyProtectedMember
        self._children = self._children + (child,)
        return child[Stimulus.position_in_parent_attribute_name]


def add_word_stimuli_to_parent(parent, tagged_word_string):

    for w in tagged_word_string.split():
        w = w.strip()
        if len(w) == 0:
            continue
        parts = w.split('/')
        if len(parts) != 2:
            raise ValueError('Expected a \'/\' separated pair. Got {}'.format(w))
        word, tag = parts

        word_stimulus = StimulusBuilder(Stimulus.word_level, attributes={
            Stimulus.master_stimulus_index_attribute_name:
                parent[Stimulus.master_stimulus_index_attribute_name],
            Stimulus.text_attribute_name: word.strip(),
            Stimulus.part_of_speech_attribute_name: tag.strip()
        })

        parent.add_child(word_stimulus)

    parent[Stimulus.text_attribute_name] = ' '.join([c[Stimulus.text_attribute_name] for c in parent.children])


def add_sentence_stimuli_to_parent(parent, tagged_word_strings):

    for index_sentence, tagged_word_string in enumerate(tagged_word_strings):
        sentence_stimulus = StimulusBuilder(Stimulus.sentence_level, attributes={
            Stimulus.master_stimulus_index_attribute_name: parent[Stimulus.master_stimulus_index_attribute_name]
        })
        add_word_stimuli_to_parent(sentence_stimulus, tagged_word_string)
        parent.add_child(sentence_stimulus)

    parent[Stimulus.text_attribute_name] = ' '.join([c[Stimulus.text_attribute_name] for c in parent.children])


def make_root_stimulus_builder(root_level, index_stimulus):
    return StimulusBuilder(root_level, attributes={
        Stimulus.position_in_parent_attribute_name: -1,
        Stimulus.position_in_root_attribute_name: -1,
        Stimulus.master_stimulus_index_attribute_name: index_stimulus,
        Stimulus.sort_key_attribute_name: index_stimulus,
        Stimulus.stratification_key_attribute_name: index_stimulus})
