from __future__ import print_function
from six import iteritems


__author__ = 'drschwar'


__all__ = ['read_commented_file', 'read_tagged_file']


def _is_line_pos_escaped(line, index):
    index -= 1
    is_escaped = False
    while index >= 0:
        if line[index] == '\\':
            is_escaped = not is_escaped
            index -= 1
        else:
            break
    return is_escaped


def _non_escaped_indices(line, char):
    indices = list()
    for index in range(len(line)):
        if line[index] == char and not _is_line_pos_escaped(line, index):
            indices.append(index)
    return indices


def _split_non_escaped(line, delim, is_keep_empty, is_unescape=False):
    indices = _non_escaped_indices(line, delim)
    tokens = list()
    if len(indices) == 0:
        tokens.append(line)
    else:
        for index_of_delim, index_delim in enumerate(indices):
            if index_of_delim == 0:
                tokens.append(line[:index_delim])
            else:
                index_start = indices[index_of_delim - 1] + 1
                tokens.append(line[index_start:index_delim])
        tokens.append(line[indices[-1] + 1:])
    if not is_keep_empty:
        tokens = list(filter(lambda token: len(token) > 0, tokens))
    if is_unescape:
        tokens = list(map(lambda token: _unescape(token), tokens))
    return tokens


# match the behavior of MATLAB code, even though we could
# use more sophisticated stuff here
def _unescape(s):

    is_escape = False
    escaped = None
    for index in range(len(s)):
        if s[index] == '\\':
            if escaped is None:  # first time we encounter escape
                escaped = s[:index]
            is_escape = not is_escape
            if not is_escape:
                escaped += '\\'
        else:
            is_escape = False
            if escaped is not None:
                escaped += s[index]
    return escaped if escaped is not None else s


def read_commented_file(path, comment_char='%'):

    def handle_comment(current_line):
        index_comment = -1
        for index in range(len(current_line)):
            if current_line[index] == comment_char and not _is_line_pos_escaped(current_line, index):
                index_comment = index
                break
        return line[:index_comment] if index_comment >= 0 else line

    with open(path, 'r') as commented_file:
        for index_line, line in enumerate(commented_file):
            line = handle_comment(line)
            if len(line) > 0:
                yield index_line, line


def _get_or_default(from_dict, key, default_val=None, set_on_not_found=False):
    if key in from_dict:
        return from_dict[key]
    if set_on_not_found:
        from_dict[key] = default_val
    return default_val


def _split_on_first(s, split_on):
    index_split = s.find(split_on)
    if index_split >= 0:
        return s[:index_split], s[index_split + len(split_on):]
    else:
        return s, None


def _map_type(s):
    if s == 'int16' or s == 'int32' or s == 'int64':
        return int
    elif s == 'uint8' or s == 'uint16' or s == 'uint32' or s == 'uint64':
        raise ValueError('Unsigned integer types not supported by Python')
    elif s == 'double' or s == 'single':
        return float
    elif s == 'bool':
        return bool
    elif s == 'string':
        return type('')
    else:
        raise ValueError('Unknown type: {0}'.format(s))


def _parse_bool(s):
    if s == '1' or s == 't' or s == 'T' or s == 'true' or s == 'True':
        return True
    elif s == '0' or s == 'f' or s == 'F' or s == 'false' or s == 'False':
        return False
    else:
        raise ValueError('bad format for boolean: {0}'.format(s))


def _canonical_value(name, value, allowed_field_values, field_types, field_value_remap):
    if name not in allowed_field_values:
        raise KeyError('Unable to find allowed_field_values for {0}'.format(name))
    field_allowed_values = allowed_field_values[name]
    if value is None or (field_allowed_values is not None and value not in field_allowed_values):
        raise ValueError('Value {0} is not in set of allowed values for {1}'.format(value, name))
    canonical_value = value
    if name in field_value_remap:
        value_remap = field_value_remap[name]
        if value in value_remap:
            canonical_value = value_remap[value]
    if name in field_types:
        field_type = field_types[name]
        if field_type == bool:
            canonical_value = _parse_bool(canonical_value)
        elif field_type == str:
            canonical_value = field_type(canonical_value)
    return canonical_value


def _handle_allowed_values(
        column_name,
        field_name,
        column_name_to_field_value_remap,
        column_name_to_allowed_field_values,
        column_name_to_field_values_copy_column,
        column_name_to_field_types,
        allowed_value_set,
        remap_operator):

    field_value_remap = _get_or_default(
        column_name_to_field_value_remap, column_name, default_val=dict(), set_on_not_found=True)
    allowed_field_values = _get_or_default(
        column_name_to_allowed_field_values, column_name, default_val=dict(), set_on_not_found=True)
    allowed_field_values_copy_column = _get_or_default(
        column_name_to_field_values_copy_column, column_name, default_val=dict(), set_on_not_found=True)
    field_types = _get_or_default(
        column_name_to_field_types, column_name, default_val=dict(), set_on_not_found=True)

    indices_semi = _non_escaped_indices(allowed_value_set, ';')
    if len(indices_semi) > 0:
        index_end_allowed_values = indices_semi[0]
    else:
        index_end_allowed_values = len(allowed_value_set)

    allowed_value_set = allowed_value_set[:index_end_allowed_values]
    field_type, allowed_values = _split_on_first(allowed_value_set, ':')
    field_type = field_type.strip()
    allowed_values = allowed_values.strip() if allowed_values is not None else ''
    if field_type == 'use_values_from':
        allowed_field_values_copy_column[field_name] = allowed_values
    else:
        field_types[field_name] = _map_type(field_type)

        if len(allowed_values) == 0:  # signals that this set is open
            # special case for boolean
            # noinspection PyPep8
            if field_types[field_name] == bool:
                allowed_field_values[field_name] = {'t', 'true', 'True', 'f', 'false', 'False'}
            else:
                allowed_field_values[field_name] = None
        else:
            if not allowed_values.startswith('{') or not allowed_values.endswith('}'):
                raise ValueError('Expected allowed values surrounded by curly braces: {0}'.format(allowed_values))
            allowed_values_split = _split_non_escaped(allowed_values[1:-1], ',', is_keep_empty=True)
            value_remap = None
            field_values_allowed = set()
            for allowed_value in allowed_values_split:
                file_form, struct_form = _split_on_first(allowed_value, remap_operator)
                file_form = _unescape(file_form.strip())
                struct_form = _unescape(struct_form.strip()) if struct_form is not None else None
                if struct_form is not None:
                    if value_remap is None:
                        value_remap = _get_or_default(
                            field_value_remap, field_name, default_val=dict(), set_on_not_found=True)
                    value_remap[file_form] = struct_form
                field_values_allowed.add(file_form)
            allowed_field_values[field_name] = field_values_allowed

    return index_end_allowed_values


def read_tagged_file(path, comment_char='%', delimiter='\t'):

    is_header = True
    end_header_line = '*****'
    remap_operator = '->'

    columns = list()

    column_name_to_column_type = dict()
    column_name_to_required_fields = dict()
    column_name_to_optional_fields = dict()
    column_name_to_allowed_field_values = dict()
    column_name_to_field_values_copy_column = dict()
    column_name_to_field_types = dict()
    column_name_to_field_name_remap = dict()
    column_name_to_field_value_remap = dict()
    column_name_to_default_field_values = dict()

    configuration = dict()
    items = list()

    for index_line, line in read_commented_file(path, comment_char):
        try:
            line = line.strip()
            if len(line) == 0:
                continue
            if is_header:
                if line == end_header_line:
                    is_header = False
                    if len(columns) == 0:
                        raise ValueError('Header ended before columns were specified')
                    for index_column, column_name in enumerate(columns):

                        allowed_field_values_copy_column = _get_or_default(
                            column_name_to_field_values_copy_column, column_name)

                        if allowed_field_values_copy_column is not None:
                            for field_name, copy_column_and_field in iteritems(allowed_field_values_copy_column):
                                copy_column, copy_field = _split_on_first(copy_column_and_field, '.')
                                if copy_column not in column_name_to_allowed_field_values:
                                    raise KeyError(
                                        'Column specified as master for allowed field '
                                        'values does not exist: {0}'.format(copy_column))
                                copy_allowed_field_values = column_name_to_allowed_field_values[copy_column]
                                if copy_field not in copy_allowed_field_values:
                                    raise KeyError('Field specified as master for allowed field '
                                                   'values does not exist in column {0}: {1}'.format(
                                                       copy_column, copy_field))
                                current_allowed_field_values = _get_or_default(
                                    column_name_to_allowed_field_values,
                                    column_name,
                                    default_val=dict(),
                                    set_on_not_found=True)
                                current_allowed_field_values[field_name] = copy_allowed_field_values[copy_field]

                        allowed_field_values = _get_or_default(
                            column_name_to_allowed_field_values, column_name, dict())
                        field_types = _get_or_default(column_name_to_field_types, column_name, dict())
                        field_value_remap = _get_or_default(column_name_to_field_value_remap, column_name, dict())
                        default_field_values = _get_or_default(
                            column_name_to_default_field_values, column_name, dict())
                        default_field_keys = default_field_values.keys()
                        for default_field_key in default_field_keys:
                            default_field_values[default_field_key] = _canonical_value(
                                default_field_key,
                                default_field_values[default_field_key],
                                allowed_field_values,
                                field_types,
                                field_value_remap
                            )

                    continue  # go to next line

                header_key, header_val = _split_on_first(line, '=')
                header_key = header_key.strip()

                if header_key == 'columns':
                    if len(columns) > 0:
                        raise KeyError('columns cannot be specified twice: {0}'.format(line))

                    column_specs = header_val.strip().split(',')
                    for column_spec in column_specs:
                        column_name, column_type = _split_on_first(column_spec, ':')
                        column_name = column_name.strip()
                        columns.append(column_name)
                        if column_type is not None:
                            column_name_to_column_type[column_name] = column_type.strip()
                else:
                    scope_key, scope_val = _split_on_first(header_key, ':')
                    if scope_val is None:
                        raise ValueError('lines in header must specify a column name, then specifier separated by '
                                         'a \':\'. Error in {0} at line {1}: {2}'.format(path, index_line, line))
                    scope_key = scope_key.strip()
                    scope_val = scope_val.strip()
                    if scope_key.startswith('keyval'):
                        is_list_type = False
                        if len(scope_key) > len('keyval'):
                            if scope_key[len('keyval')] != '_':
                                raise ValueError('Bad format in header: {0}'.format(scope_key))
                            val_type = scope_key[len('keyval_'):]
                            if val_type.endswith('_list'):
                                is_list_type = True
                                val_type = val_type[:-len('_list')]
                        else:
                            val_type = 'string'

                        val_type = _map_type(val_type)

                        if is_list_type:
                            to_convert = header_val.split(',')
                            if not isinstance(val_type, type('')):
                                to_convert = filter(lambda x: len(x) > 0, map(lambda con: con.strip(), to_convert))
                        elif not isinstance(val_type, type('')):
                            to_convert = [_unescape(header_val.strip())]
                        else:
                            to_convert = [_unescape(header_val)]

                        config_key = scope_val
                        config_vals = list()
                        for item in to_convert:
                            if isinstance(val_type, type('')):
                                config_vals.append(item)
                            elif isinstance(val_type, bool):
                                config_vals.append(_parse_bool(item))
                            else:
                                config_vals.append(val_type(item))
                        if not is_list_type:
                            configuration[config_key] = config_vals[0]
                        else:
                            configuration[config_key] = config_vals

                    else:

                        column_name = scope_key
                        spec_key, spec_val = _split_on_first(scope_val, ':')
                        if spec_val is None:
                            raise ValueError('lines in header must specify a column name, then specifier separated by '
                                             'a \':\'. error in {0} at {1}: {2}'.format(path, index_line, line))
                        spec_key = spec_key.strip()
                        spec_val = spec_val.strip()
                        if spec_key == 'unnamed_field':
                            allowed_value_set = header_val.strip() if header_val is not None else header_val
                            field_name = spec_val

                            if allowed_value_set is None or len(allowed_value_set) == 0:
                                raise ValueError('expected <field name>=<allowed value set>. error in {0} at {1}: {2}'.
                                                 format(path, index_line, line))

                            _handle_allowed_values(
                                column_name,
                                field_name,
                                column_name_to_field_value_remap,
                                column_name_to_allowed_field_values,
                                column_name_to_field_values_copy_column,
                                column_name_to_field_types,
                                allowed_value_set,
                                remap_operator
                            )

                            required_fields = _get_or_default(
                                column_name_to_required_fields, column_name, default_val=list(), set_on_not_found=True)
                            required_fields.append(field_name)

                        elif spec_key == 'named_field':
                            field_name_with_remap = spec_val
                            allowed_value_and_default = header_val.strip() if header_val is not None else header_val
                            if allowed_value_and_default is None or len(allowed_value_and_default) == 0:
                                raise ValueError('expected <field name>=<allowed value set>. error in {0} at {1}: {2}'.
                                                 format(path, index_line, line))
                            field_name, remapped_field_name = _split_on_first(field_name_with_remap, remap_operator)
                            if remapped_field_name is not None:
                                field_name_remap = _get_or_default(
                                    column_name_to_field_name_remap,
                                    column_name,
                                    default_val=dict(),
                                    set_on_not_found=True)
                                field_name_remap[field_name] = remapped_field_name

                            index_end_allowed_values = _handle_allowed_values(
                                column_name,
                                field_name,
                                column_name_to_field_value_remap,
                                column_name_to_allowed_field_values,
                                column_name_to_field_values_copy_column,
                                column_name_to_field_types,
                                allowed_value_and_default,
                                remap_operator
                            )

                            default_val_spec = allowed_value_and_default[index_end_allowed_values:].strip() \
                                if index_end_allowed_values < len(allowed_value_and_default) else ''

                            if len(default_val_spec) == 0 or default_val_spec[0] != ';':
                                raise ValueError('expected semi-colon after allowed values, followed by default value. '
                                                 'error in {0} at {1}: {2}'.format(path, index_line, line))
                            default_val_spec = default_val_spec[1:]  # remove ';'
                            default_key, default_val = _split_on_first(default_val_spec, '=')

                            default_key = default_key.strip()
                            default_val = _unescape(default_val.strip())

                            if default_key != 'default':
                                raise ValueError('expected default value specifier starting with \'default\'. '
                                                 'error in {0} at {1}: {2}'.format(path, index_line, line))

                            default_field_values = _get_or_default(
                                column_name_to_default_field_values,
                                column_name,
                                default_val=dict(),
                                set_on_not_found=True)

                            default_field_values[field_name] = default_val

                            optional_fields = _get_or_default(
                                column_name_to_optional_fields, column_name, default_val=list(), set_on_not_found=True)
                            optional_fields.append(field_name)

            else:  # not header
                item = dict()
                current_columns = _split_non_escaped(line, delimiter, is_keep_empty=True, is_unescape=False)
                if len(current_columns) != len(columns):
                    raise ValueError('mismatch between number of columns specified in header and number of columns in '
                                     'current line, {0}'.format(index_line))
                for index_column, column_name in enumerate(columns):
                    column_type = _get_or_default(column_name_to_column_type, column_name, default_val='string')
                    if column_type == 'spaceDelimTagged':
                        words_with_fields = current_columns[index_column].split()
                        tagged_words = list()
                        required_fields = _get_or_default(column_name_to_required_fields, column_name, list())
                        optional_fields = _get_or_default(column_name_to_optional_fields, column_name, list())
                        field_name_remap = _get_or_default(column_name_to_field_name_remap, column_name, dict())
                        field_value_remap = _get_or_default(column_name_to_field_value_remap, column_name, dict())
                        allowed_field_values = _get_or_default(
                            column_name_to_allowed_field_values, column_name, dict())
                        field_types = _get_or_default(column_name_to_field_types, column_name, dict())
                        default_field_values = _get_or_default(
                            column_name_to_default_field_values, column_name, dict())
                        for word_with_fields in words_with_fields:
                            word_fields = _split_non_escaped(
                                word_with_fields.strip(), '/', is_keep_empty=True, is_unescape=True)
                            tagged_word = dict()
                            for index_optional_field, struct_field_name in enumerate(optional_fields):
                                if struct_field_name in field_name_remap:
                                    struct_field_name = field_name_remap[struct_field_name]
                                    tagged_word[struct_field_name] = \
                                        default_field_values[optional_fields[index_optional_field]]
                            index_unnamed_field = 0
                            for word_field in word_fields:
                                name, value = _split_on_first(word_field, '=')
                                if value is None:
                                    value = name
                                    if index_unnamed_field > len(required_fields):
                                        raise ValueError('Too many unnamed fields in {0} at {1}: {2}'.format(
                                            path, index_line, word_field))
                                    name = required_fields[index_unnamed_field]
                                    struct_field_name = name
                                    index_unnamed_field += 1
                                else:
                                    if name not in default_field_values:
                                        raise ValueError(
                                            'Named field was not specified in header. Error in {0} at {1}: {2}'.format(
                                                path, index_line, name))
                                    struct_field_name = _get_or_default(field_name_remap, name, name)

                                canonical_value = _canonical_value(
                                    name, value, allowed_field_values, field_types, field_value_remap)

                                tagged_word[struct_field_name] = canonical_value

                            if index_unnamed_field < len(required_fields):
                                raise ValueError('Too few unnamed fields in {0} at {1}: {2}'.format(
                                    path, index_line, line))

                            tagged_words.append(tagged_word)

                        item[column_name] = tagged_words

                    else:

                        column_type = _map_type(column_type)
                        if isinstance(column_type, bool):
                            item[column_name] = _parse_bool(current_columns[index_column].strip())
                        elif isinstance(column_type, type('')):
                            item[column_name] = _unescape(current_columns[index_column])
                        else:
                            item[column_name] = column_type(current_columns[index_column].strip())

                items.append(item)

        except:
            print('Error near {0}:'.format(index_line))
            print(line)
            raise

    return items, configuration
