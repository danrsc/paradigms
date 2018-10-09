from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import inspect
from six.moves import zip_longest
from six import PY2


__all__ = ['zip_equal', 'copy_from_properties', 'get_keyword_properties', 'status_iter']


def zip_equal(*it):
    """
    Like zip, but raises a ValueError if the iterables are not of equal length
    Args:
        *it: The iterables to zip

    Returns:
        yields a tuple of items, one from each iterable
    """
    # wrap the iterators in an enumerate to guarantee that None is a legitimate sentinel
    iterators = [enumerate(i) for i in it]
    for item in zip_longest(*iterators):
        try:
            result = tuple(part[1] for part in item)
            yield result
        except TypeError:
            raise ValueError('Unequal number of elements in iterators')


def copy_from_properties(instance, **kwargs):
    """
    Returns a copy of instance by calling __init__ with keyword arguments matching the properties of instance.
    The values of these keyword arguments are taken from the properties of instance except where overridden by
    kwargs. Thus for a class Foo with properties [a, b, c], copy_from_properties(instance, a=7) is equivalent to
    Foo(a=7, b=instance.b, c=instance.c)
    Args:
        instance: The instance to use as a template
        **kwargs: The keyword arguments to __init__ that should not come from the current instance's properties

    Returns:
        A copy of instance modified according to kwargs
    """
    property_names = [n for n, v in inspect.getmembers(type(instance), lambda m: isinstance(m, property))]
    if PY2:
        # noinspection PyDeprecation
        init_kwargs = inspect.getargspec(type(instance).__init__).args
    else:
        init_kwargs = inspect.getfullargspec(type(instance).__init__).args

    def __iterate_key_values():
        for k in init_kwargs[1:]:
            if k in kwargs:
                yield k, kwargs[k]
            elif k in property_names:
                yield k, getattr(instance, k)

    return type(instance)(**dict(__iterate_key_values()))


def get_keyword_properties(instance, just_names=False):

    property_names = [n for n, v in inspect.getmembers(type(instance), lambda m: isinstance(m, property))]
    if PY2:
        # noinspection PyDeprecation
        init_kwargs = inspect.getargspec(type(instance).__init__).args
    else:
        init_kwargs = inspect.getfullargspec(type(instance).__init__).args

    if just_names:
        return [k for k in init_kwargs if k in property_names]

    return [(k, getattr(instance, k)) for k in init_kwargs if k in property_names]


def status_iter(
        iterable, freq=1, item_count=None, status_format=None, status_function=None, round_timedeltas_to_seconds=True):
    if status_format is None and status_function is None:
        maybe_count = ' of {count}' if item_count is not None else ''
        status_format = 'item {complete_count}' + maybe_count + ', avg: {average_time}'
        if item_count is not None:
            status_format += ', est: {remaining_time}'
    start_time = datetime.datetime.now()

    def _status(complete_count, current_item):
        now = datetime.datetime.now()
        duration = now - start_time
        average_time = duration // complete_count if complete_count > 0 else None

        remaining_time = None
        fraction_complete = None
        if item_count is not None:
            if item_count == 0:
                remaining_time = datetime.timedelta(seconds=0)
                fraction_complete = 1.0
            else:
                remaining_time = average_time * (item_count - complete_count)
                fraction_complete = float(complete_count) / item_count

        def _round_timedelta(t):
            return datetime.timedelta(seconds=t.seconds + t.days * 24 * 3600) if t is not None else None

        if round_timedeltas_to_seconds:
            duration = _round_timedelta(duration)
            average_time = _round_timedelta(average_time)
            remaining_time = _round_timedelta(remaining_time)

        kwargs = dict(
            complete_count=complete_count,
            item=current_item,
            count=item_count,
            fraction_complete=fraction_complete,
            start_time=start_time,
            now=now,
            duration=duration,
            average_time=average_time,
            remaining_time=remaining_time)

        if status_format is not None:
            print(status_format.format(**kwargs))

        if status_function is not None:
            status_function(kwargs)

    index = None
    for index, item in enumerate(iterable):
        yield item

        if index % freq == 0:
            _status(index + 1, item)

    if index is None or index % freq != 0:
        _status(index + 1 if index is not None else 0, None)
