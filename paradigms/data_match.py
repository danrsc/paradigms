from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import inspect
from six import PY2

from .recording_tuple import RecordingTuple, ROITuple


__all__ = [
    'match_recordings',
    'make_filter',
    'make_filters',
    'make_matcher',
    'make_white_black_matcher',
    'filter_recordings',
    'split_recordings']


def match_recordings(root, recording_tuple_regex, experiment_on_unmatched=None):
    recordings = list()
    expected_type = None
    for current_directory, sub_directories, files in os.walk(root):
        for current_file in files:
            full_path = os.path.join(current_directory, current_file)
            match_object = recording_tuple_regex.match(full_path)
            if match_object is not None:
                group_dict = match_object.groupdict()
                if 'experiment' not in group_dict and experiment_on_unmatched is not None:
                    group_dict['experiment'] = experiment_on_unmatched
                if 'roi' in group_dict:
                    recordings.append(ROITuple(full_path=full_path, **group_dict))
                else:
                    recordings.append(RecordingTuple(full_path=full_path, **group_dict))
                if expected_type is None:
                    expected_type = type(recordings[-1])
                elif not isinstance(recordings[-1], expected_type):
                    raise ValueError('Mixed tuple types in recordings')

    return recordings


def _is_match(regex, part):
    match_obj = regex.match(part)
    return match_obj is not None and match_obj.end() - match_obj.start() == len(part)


def get_tuple_type_properties(tuple_type):
    property_names = [n for n, v in inspect.getmembers(tuple_type, lambda m: isinstance(m, property))]
    if PY2:
        # noinspection PyDeprecation
        init_kwargs = inspect.getargspec(tuple_type.__init__).args
    else:
        init_kwargs = inspect.getfullargspec(tuple_type.__init__).args
    property_names = [n for n in init_kwargs if n != 'full_path' and n in property_names]
    return property_names


def make_filter(property_names, filter_parts):

    if len(filter_parts) != len(property_names):
        raise ValueError('Expected {}'.format('$'.join(['<{}>'.format(n) for n in property_names])))

    name_regex_pairs = [(property_name, re.compile(part, re.IGNORECASE))
                        for property_name, part in zip(property_names, filter_parts)]

    def _filter(to_check):
        return all([_is_match(regex, getattr(to_check, name)) for name, regex in name_regex_pairs])

    return _filter


def make_filters(property_names, string_filter_list):
    return None if string_filter_list is None \
        else list(map(lambda s: make_filter(property_names, s), string_filter_list))


def make_matcher(property_names, string_filter_list, result_on_empty_filters):
    filters = make_filters(property_names, string_filter_list)

    def matches_any(recording_tuple):
        return result_on_empty_filters if filters is None else any(map(lambda f: f(recording_tuple), filters))

    return matches_any


def make_white_black_matcher(property_names, white_list=None, black_list=None):
    white_list_matcher = make_matcher(property_names, white_list, True)
    black_list_matcher = make_matcher(property_names, black_list, False)

    def is_match(recording_tuple):
        return white_list_matcher(recording_tuple) and not black_list_matcher(recording_tuple)

    return is_match


def filter_recordings(recording_tuples, white_list=None, black_list=None):
    property_names = get_tuple_type_properties(type(recording_tuples[0]))
    is_match = make_white_black_matcher(property_names, white_list, black_list)
    return list(filter(lambda r: is_match(r), recording_tuples))


def split_recordings(recording_tuples, second_set_white_list=None, second_set_black_list=None):

    if second_set_white_list is None and second_set_black_list is None:
        return recording_tuples, list()

    property_names = get_tuple_type_properties(type(recording_tuples[0]))
    is_match = make_white_black_matcher(property_names, second_set_white_list, second_set_black_list)
    first_set = list()
    second_set = list()
    for r in recording_tuples:
        if is_match(r):
            second_set.append(r)
        else:
            first_set.append(r)
    return first_set, second_set
