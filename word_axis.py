from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy

from brain_gen.core import copy_from_properties
from brain_gen.io import sort_regions, coalesce_lobes, to_lobe_names


__all__ = [
    'AxisWord',
    'AxisWords',
    'pass_act_3_short_passive_generic_axis',
    'pass_act_3_short_active_generic_axis',
    'pass_act_3_long_passive_generic_axis',
    'pass_act_3_long_active_generic_axis',
    'make_concrete_axis',
    'set_up_word_axis',
    'make_source_localized_region_word_grid_show']


class AxisWord(object):

    def __init__(self, text, align_keys=None, color=None, weight=None, time_delta_in_seconds=0.5):
        """
        Container class to help create an axis with word labels
        Args:
            text: The single word to display
            align_keys: e.g. 'noun1' or 'last'. If not None, these keys indicate the role of this word in the example,
                so that if the data is aligned to one of these keys, this word will be at time 0.
            color: The color to display this word in
            weight: The font-weight (e.g. semibold) to display this word in
            time_delta_in_seconds: The number of seconds between the onset of this word and
                the onset of the next word (if it exists)
        """
        self._text = text
        if align_keys is None:
            self._align_keys = {}
        elif isinstance(align_keys, str):
            self._align_keys = {align_keys}
        else:
            self._align_keys = set(align_keys)
        self._color = color
        self._weight = weight
        self._time_delta_in_seconds = time_delta_in_seconds

    @property
    def text(self):
        return self._text

    @property
    def align_keys(self):
        return set(self._align_keys)

    @property
    def color(self):
        return self._color

    @property
    def weight(self):
        return self._weight

    @property
    def time_delta_in_seconds(self):
        return self._time_delta_in_seconds


class AxisWords(object):

    def __init__(self, *axis_words):
        self._words = tuple(axis_words)

    @property
    def words(self):
        return self._words

    @property
    def texts(self):
        return [w.text for w in self.words]

    @property
    def colors(self):
        return [w.color for w in self.words]

    @property
    def weights(self):
        return [w.weight for w in self.words]

    def times_in_seconds(self, align_to, is_midpoints=False):
        time = 0
        result = list()
        alignment_time = None
        for index_word, word in enumerate(self.words):
            if is_midpoints:
                result.append(time + (word.time_delta_in_seconds / 2.))
            else:
                result.append(time)
            if align_to in word.align_keys:
                if alignment_time is not None:
                    raise ValueError('Multiple words match alignment key {}: {}'.format(align_to, self.texts))
                alignment_time = time
            time += word.time_delta_in_seconds
        if alignment_time is None:
            raise ValueError('No words match alignment key {}: {}'.format(align_to, self.texts))
        result = numpy.array(result)
        result -= alignment_time
        return result


def set_up_word_axis(
        ax, time_in_seconds, examples, align_to, axis='x', initial_offset=5, additional_offsets=15, tick_params=None,
        font_size=12, is_display_on_midpoints=True):

    is_x_axis = axis == 'x'
    if not is_x_axis and axis != 'y':
        raise ValueError('axis must be either \'x\' or \'y\'')

    if tick_params is None:
        tick_params = dict()
    else:
        tick_params = dict(tick_params)

    def _default(dict_, **defaults):
        for key in defaults:
            if key not in dict_:
                dict_[key] = defaults[key]

    if is_x_axis:
        _default(tick_params, labeltop='on', labelbottom='off', top='off', bottom='off')
    else:
        _default(tick_params, labelleft='on', labelright='off', left='off', right='off')

    current_offset = initial_offset
    axes = [ax]

    for _ in range(1, len(examples)):
        a = ax.twiny() if is_x_axis else ax.twinx()
        axes.append(a)
    for index_example, (example, current_axes) in enumerate(zip(examples, axes)):
        current_axes.set_xlim(ax.get_xlim()) if is_x_axis else current_axes.set_ylim(ax.get_ylim())
        limits = current_axes.get_xlim() if is_x_axis else current_axes.get_ylim()
        if index_example > 0:
            current_offset += additional_offsets
        tick_params['pad'] = current_offset
        current_axes.tick_params(axis=axis, **tick_params)
        example_onsets = example.times_in_seconds(align_to, is_midpoints=is_display_on_midpoints)
        delta = time_in_seconds[-1] + (time_in_seconds[-1] - time_in_seconds[-2]) - time_in_seconds[0]
        onset_fraction = (example_onsets - time_in_seconds[0]) / delta
        if onset_fraction[-1] < 0:
            # write a single tick with the name of the final word + the number of seconds post that word
            tick_labels = [example.texts[-1] + ' + {}s'.format(time_in_seconds[0] - example_onsets[-1])]
            tick_colors = [example.colors[-1]]
            tick_weights = [example.weights[-1]]
            ticks = [limits[0]]
        elif onset_fraction[0] > 1:
            # write a single tick with the name of the first word - the number of seconds before that word
            tick_labels = [example.texts[0] + ' - {}s'.format(example_onsets[0] - time_in_seconds[-1])]
            tick_colors = [example.colors[-1]]
            tick_weights = [example.weights[-1]]
            ticks = [limits[1]]
        else:
            ticks = list()
            tick_labels = list()
            tick_colors = list()
            tick_weights = list()
            for example_word, onset in zip(example.words, onset_fraction):
                if onset < 0 or onset > 1:
                    # this word is not in the window
                    continue
                ticks.append(onset * (limits[1] - limits[0]) + limits[0])
                tick_labels.append(example_word.text)
                tick_colors.append(example_word.color)
                tick_weights.append(example_word.weight)
        current_axes.set_xticks(ticks) if is_x_axis else current_axes.set_yticks(ticks)
        current_axes.set_xticklabels(tick_labels) if is_x_axis else current_axes.set_yticklabels(tick_labels)
        labels = current_axes.get_xticklabels() if is_x_axis else current_axes.get_yticklabels()
        for index_label, label in enumerate(labels):
            if font_size is not None:
                label.set_size(font_size)
            if tick_colors[index_label] is not None:
                label.set_color(tick_colors[index_label])
            if tick_weights[index_label] is not None:
                label.set_weight(tick_weights[index_label])
        tick_lines = current_axes.xaxis.get_ticklines() if is_x_axis else current_axes.yaxis.get_ticklines()
        for tick_line in tick_lines:
            tick_line.set_visible = False


pass_act_3_short_active_generic_axis = AxisWords(
    AxisWord('the', color='grey', align_keys=['first']),
    AxisWord('noun 1', weight='semibold', align_keys=['noun1', 'agent']),
    AxisWord('verb', weight='semibold', align_keys=['verb', 'last']))

pass_act_3_short_passive_generic_axis = AxisWords(
    AxisWord('the', color='grey', align_keys=['first']),
    AxisWord('noun 1', weight='semibold', align_keys=['noun1', 'patient']),
    AxisWord('was', color='grey'),
    AxisWord('verb', weight='semibold', align_keys=['verb', 'last']))

pass_act_3_long_active_generic_axis = AxisWords(
    AxisWord('the', color='grey', align_keys=['first']),
    AxisWord('noun 1', weight='semibold', align_keys=['noun1', 'agent']),
    AxisWord('verb', weight='semibold', align_keys=['verb']),
    AxisWord('the', color='grey'),
    AxisWord('noun 2', weight='semibold', align_keys=['noun2', 'patient', 'last']))

pass_act_3_long_passive_generic_axis = AxisWords(
    AxisWord('the', color='grey', align_keys=['first']),
    AxisWord('noun 1', weight='semibold', align_keys=['noun1', 'patient']),
    AxisWord('was', color='grey'),
    AxisWord('verb', weight='semibold', align_keys=['verb']),
    AxisWord('by', color='grey'),
    AxisWord('the', color='grey'),
    AxisWord('noun 2', weight='semibold', align_keys=['noun2', 'agent', 'last']))


def make_concrete_axis(template, noun_1, verb, noun_2):
    words = list()
    for word in template.words:
        if 'noun1' in word.align_keys:
            words.append(copy_from_properties(word, text=noun_1))
        elif 'verb' in word.align_keys:
            words.append(copy_from_properties(word, text=verb))
        elif 'noun_2' in word.align_keys:
            words.append(copy_from_properties(word, text=noun_2))
        else:
            words.append(word)
    return AxisWords(*words)


# e.g. to make a concrete example from the generic axes
# the_boy helped the girl = make_concrete_axis(pass_act_3_long_active_generic_axis, boy, helped, girl)


def make_source_localized_region_word_grid_show(title, data, time, regions, examples=None, align_to='first'):
    regions, sort_indices = sort_regions(regions)
    data = numpy.asarray(data)
    data = data[sort_indices]
    lobe_names, lobe_starts = coalesce_lobes(to_lobe_names(regions))

    def draw_grid_lines(ax):
        half_second = numpy.ceil(numpy.min(time) / 0.5) * 0.5
        while half_second < numpy.max(time):
            index = numpy.searchsorted(time, half_second)
            if time[index] < half_second + 0.001:
                index -= 0.5
            ax.axvline(index, color='red')
            half_second += 0.5
        for lobe_boundary in lobe_starts[1:]:
            ax.axhline(lobe_boundary - 0.5, color='black')

    def set_up_y_ticks(ax):
        lobe_bounds = numpy.concatenate((lobe_starts, numpy.expand_dims(len(regions), 0)), axis=0)
        lobe_ticks = lobe_bounds[:-1] + numpy.diff(lobe_bounds) // 2
        ax.set_yticks(lobe_ticks)
        ax.set_yticklabels(lobe_names)
        for label in ax.get_yticklabels():
            label.set_size(12)
        for tick_line in ax.yaxis.get_ticklines():
            tick_line.set_visible(False)

    def set_up_x_ticks(ax):
        if examples is not None:
            set_up_word_axis(
                ax, time, examples, align_to=align_to, tick_params={'labeltop': 'off', 'labelbottom': 'on'})
        else:
            ax.tick_params(axis='x', labeltop='off', labelbottom='on', top='off', bottom='off')
            word_boundary_spacing = 0.5
            tick_times = []
            indices_x_ticks = []
            first_tick_time = numpy.ceil(time[0] / word_boundary_spacing) * word_boundary_spacing
            max_time = time[-1] + (time[-1] - time[-2])
            limits = ax.get_xlim()

            def _to_tick(t):
                return (t - time[0]) / (max_time - time[0]) * (limits[1] - limits[0]) + limits[0]

            if max_time >= first_tick_time >= time[0]:
                tick_times.append(first_tick_time)
                indices_x_ticks.append(_to_tick(first_tick_time))
                while tick_times[-1] + word_boundary_spacing <= max_time:
                    tick_times.append(tick_times[-1] + word_boundary_spacing)
                    indices_x_ticks.append(_to_tick(tick_times[-1]))
            ax.set_xticks(indices_x_ticks)
            ax.set_xticklabels(time[indices_x_ticks])
            for label in ax.get_xticklabels():
                label.set_size(12)

    def render(ax, **kwargs):
        ax.set_title(title)
        if 'aspect' not in kwargs:
            h = ax.matshow(data, aspect='auto', **kwargs)
        else:
            h = ax.matshow(data, **kwargs)
        draw_grid_lines(ax)
        set_up_y_ticks(ax)
        set_up_x_ticks(ax)
        return h

    return render
