import itertools


__all__ = ['Event', 'BlockEvents', 'BlockStimuli', 'StimulusEvents', 'Stimulus']


class Event:

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


class BlockEvents:

    def __init__(self, block_id, events, times, data_path):
        self.block_id = block_id
        self.block_events = events
        self.times = times
        self.data_path = data_path


class BlockStimuli:

    def __init__(self, block_id, stimuli, times, data_path, modify_time):
        self.block_id = block_id
        self.stimuli = stimuli
        self.times = times
        self.data_path = data_path
        self.modify_time = modify_time


class StimulusEvents:

    def __init__(self, sub_stimuli, post_stimulus):
        self.sub_stimuli = sub_stimuli
        self.post_stimulus = post_stimulus


class Stimulus:

    passage_level = 'passage'
    sentence_level = 'sentence'
    word_level = 'word'
    presentation_level = 'presentation_response'

    text_attribute_name = 'word'  # not great naming, legacy of when we only did word experiments
    is_end_substimulus_attribute_name = 'isEndSubstimulus'
    position_in_parent_attribute_name = 'position_in_parent'
    position_in_root_attribute_name = 'position_in_root'
    master_stimulus_index_attribute_name = 'master_stimulus_index'
    sort_key_attribute_name = 'sort_key'
    stratification_key_attribute_name = 'stratification_key'
    stimulus_count_presentation_attribute_name = 'stimulus_count_presentation'
    word_count_presentation_attribute_name = 'word_count_presentation'
    post_stimulus_attribute_name = 'post_stimulus'
    time_stamp_attribute_name = 'time_stamp'
    duration_attribute_name = 'duration'
    question_text_attribute_name = 'question_text'  # for 20 questions data

    def __init__(self, level, attributes, parent=None):

        self.__level = level
        self.__text = attributes[Stimulus.text_attribute_name] if Stimulus.text_attribute_name in attributes else None
        self.__attributes = dict(attributes)
        self.__parent = parent
        if self.__parent is not None:
            self.__parent.__add_child(self)
        self.__children = list()

    @property
    def text(self):
        return self.__text

    @property
    def level(self):
        return self.__level

    def __getitem__(self, attribute_name):
        return self.__attributes[attribute_name]

    def __str__(self):
        return self.text

    def has_attribute(self, attribute_name):
        return attribute_name in self.__attributes

    def __add_child(self, child):
        if not child.has_attribute('position_in_parent'):
            raise ValueError('Cannot add a child without specifying position in parent')
        position_in_parent = child['position_in_parent']
        while len(self.__children) <= position_in_parent:
            self.__children.append(None)
        if self.__children[position_in_parent] is not None:
            raise ValueError('Child already exists at position in parent')
        self.__children[position_in_parent] = child

    def num_children(self):
        return len(self.__children)

    def child_at(self, index):
        return self.__children[index]

    def num_nodes_at_level(self, level):
        count = 0
        for child in self.__children:
            count += child.num_nodes_at_level(level)
        if level == self.level:
            count += 1
        return count

    @property
    def parent(self):
        return self.__parent

    def bounding_leaves(self):
        if self.__children is None or len(self.__children) == 0:
            return self, self
        return self.__children[0].bounding_leaves()[0], self.__children[-1].bounding_leaves()[1]

    def iter_level(self, level):
        if self.__level == level:
            yield self
        else:
            for child in self.__children:
                for yielded in child.iter_level(level):
                    yield yielded

    def __deep_copy(self, parent):
        copy = Stimulus(self.level, self.__attributes, parent)
        for child in self.__children:
            child.__deep_copy(copy)
        return copy

    def copy_with_event_attributes(self, stimulus_events, normalize, stimulus_count, word_counts):
        if self.parent is not None:
            raise RuntimeError('Can only call copy_with_event_attributes on root stimulus node')
        root_copy = self.__deep_copy(None)
        root_copy.__attributes[Stimulus.stimulus_count_presentation_attribute_name] = stimulus_count
        if stimulus_events.post_stimulus is not None:
            root_copy.__attributes[Stimulus.post_stimulus_attribute_name] = stimulus_events.post_stimulus

        # put all the sub_stimuli together
        word_events = None
        if stimulus_events.sub_stimuli is not None:
            word_events = list(filter(
                lambda event: event.trigger != 0, itertools.chain(*stimulus_events.sub_stimuli)))

        for index_word, word_stimulus in enumerate(root_copy.iter_level(Stimulus.word_level)):
            normalized_text = normalize(word_stimulus.text)
            word_count = word_counts[normalized_text] if normalized_text in word_counts else 0
            word_count += 1
            word_counts[normalized_text] = word_count
            word_stimulus.__attributes[Stimulus.word_count_presentation_attribute_name] = word_count
            if word_events is not None:
                if index_word > len(word_events):
                    raise ValueError('Mismatch in word events and master stimuli while merging: '
                                     'number of words does not agree')
                if normalize(word_events[index_word].stimulus) != normalized_text:
                    raise ValueError('Mismatch in word events and master stimuli while merging: '
                                     'expected {0}, got {1}'.format(
                                         normalized_text, normalize(word_events[index_word].stimulus)))
                word_stimulus.__attributes[Stimulus.time_stamp_attribute_name] = \
                    word_events[index_word].time_stamp
                word_stimulus.__attributes[Stimulus.duration_attribute_name] = word_events[index_word].duration

        # now for levels higher than word, add in time information
        Stimulus.__propagate_time(root_copy)
        return root_copy

    def update_time(self):
        if self.parent is not None:
            raise RuntimeError('Can only call update_time on root stimulus node')
        Stimulus.__propagate_time(self)

    @staticmethod
    def __propagate_time(node):
        if node.__children is None or len(node.__children) == 0:
            node_time = node.__attributes[Stimulus.time_stamp_attribute_name] \
                if Stimulus.time_stamp_attribute_name in node.__attributes else None
            node_duration = node.__attributes[Stimulus.duration_attribute_name] \
                if Stimulus.duration_attribute_name in node.__attributes else None
            return node_time, node_duration
        else:
            # noinspection PyUnresolvedReferences
            start_time = Stimulus.__propagate_time(node.__children[0])[0]
            end_time, end_duration = Stimulus.__propagate_time(node.__children[-1])
            end_time = end_time + end_duration if end_time is not None and end_duration is not None else None
            duration = end_time - start_time if end_time is not None and start_time is not None else None
            if start_time is not None:
                node.__attributes[Stimulus.time_stamp_attribute_name] = start_time
            if duration is not None:
                node.__attributes[Stimulus.duration_attribute_name] = duration
            return start_time, duration
