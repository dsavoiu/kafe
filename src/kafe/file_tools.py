'''
.. module:: file_tools
    :platform: Unix
    :synopsis: This submodule provides a set of helper functions for parsing
        files

.. moduleauthor:: Daniel Savoiu <danielsavoiu@gmail.com>
'''

import numpy as np

from string import split
from dataset import Dataset, build_dataset

# import main logger for kafe
import logging
logger = logging.getLogger('kafe')


def parse_column_data(file_to_parse, field_order='x,y', delimiter=' ',
                      cov_mat_files=None, title="Untitled Dataset"):
    '''
    Parses a file which contains measurement data in a one-measurement-per-row
    format. The field (column) order can be specified. It defaults to `x,y'.
    Valid field names are `x`, `y`, `xabsstat`, `yabsstat`, `xrelstat`,
    `yrelstat`. Another valid field name is `ignore` which can be used to skip
    a field.

    Every valid measurement data file *must* have an `x` and a `y` field.

    Additionally, a delimiter can be specified. If this is a whitespace
    character or omitted, any sequence of whitespace characters is assumed to
    separate the data.

    If the measurement errors and correlations are given as covariance matrices
    (in a separate file), these files can be specified using the
    `cov_mat_files` argument.

    **file_to_parse** : file-like object or string containing a file path
        The file to parse.

    *field_order* : string (optional)
        A string of comma-separated field names giving the order of the columns
        in the file. Defaults to ``'x,y'``.

    *delimiter* : string (optional)
        The field delimiter used in the file. Defaults to any whitespace.

    *cov_mat_files* : ``None`` or tuple of strings/file-like objects (optional)
        Files which contain x- and y-covariance matrices, in that order.
        Defaults to ``None``.

    **return** : `Dataset`
        A Dataset built from the parsed file.

    '''

    try:
        # try to read the lines of the file
        tmp_lines = file_to_parse.readlines()
        logger.info("Reading column data (%s) from file: %r"
                    % (field_order, file_to_parse))
        # this will fail if a file path string was passed, so alternatively:
    except AttributeError:
        # open the file pointed to by the path
        tmp_file = open(file_to_parse, 'r')
        logger.info("Reading column data (%s) from file: %s"
                    % (field_order, file_to_parse))
        # and then read the lines of the file
        tmp_lines = tmp_file.readlines()
        tmp_file.close()

    # define a dictionary of fields (lists) to populate
    fields = {'x': [],
              'y': [],
              'xabsstat': [],
              'yabsstat': [],
              'xrelstat': [],
              'yrelstat': [],
              'ignore': []}

    # define a list of axes
    axes = ('x', 'y')

    # Error handling in case of invalid field order
    if ',' in field_order:
        field_order_list = []
        for field in split(field_order, ','):  # go through the fields
            if field not in fields.keys():     # raise error for invalid fields
                raise SyntaxError("Supplied field order `%s' contains invalid \
                    field `%s'." % (field_order, field))
            elif field in field_order_list:    # error on repeated fields
                raise SyntaxError("Supplied field order `%s' contains \
                    repeated field `%s'." % (field_order, field))
            else:                              # validate field
                field_order_list.append(field)
        for axis in axes:
            if axis not in field_order_list:
                raise SyntaxError("Supplied field order `%s' \
                    missing mandatory field `%s'." % (field_order, axis))
    else:
        raise SyntaxError("Supplied field order `%s' is not a comma-separated \
            list of valid fields." % (field_order,))

    # handle delimiter
    if delimiter in ['', ' ', '\t']:  # if delimiter is a whitespace character
        delimiter = None              # set to None

    # actual file parsing
    for line in tmp_lines:  # go through the lines of the file

        if '#' in line:
            # ignore anything after a comment sign (#)
            line = split(line, '#')[0]

        if (not line) or (line.isspace()):  # ignore empty lines
            continue

        # get field contents by splitting lines
        if delimiter is None:
            tmp_fields = split(line)             # split line on whitespace
        else:
            tmp_fields = split(line, delimiter)  # split line on delimiter

        # append those contents to the right fields
        for idx, field_name in enumerate(field_order_list):
            fields[field_name].append(float(tmp_fields[idx]))

    # construct Dataset object
    dataset_kwargs = {}
    for key in fields.keys():
        if fields[key]:           # if the field is not empty
            # some syntax translation needed (x -> xdata)
            # for Dataset constructor
            if key in axes:
                dataset_kwargs[key+'data'] = np.asarray(fields[key])
            elif key == 'ignore':
                pass
            else:
                dataset_kwargs[key] = np.asarray(fields[key])

    dataset_kwargs.update({'title': title})

    # parse additional covariance matrix file, if necessary
    if cov_mat_files is not None:
        try:
            if len(cov_mat_files) != len(axes):
                print "Warning: cov_mat_files specification has \
                    invalid length (%d), ignoring... " % (len(cov_mat_files),)
        except:
            print "Warning: Invalid cov_mat_files specification, \
                ignoring... Expected 2-tuple of strings/file objects \
                instead of %r." % (cov_mat_files,)
        else:
            cov_mats = []
            for axis_id, axis_name in enumerate(axes):
                if cov_mat_files[axis_id] is not None:
                    # if cov mat is given, check for other
                    # errors on the same axis and warn
                    if (
                        set(dataset_kwargs) &
                        set([axis_name+'absstat', axis_name+'relstat'])
                    ):
                        print "Warning: Errors provided for axis `%s` \
                            along with covariance matrix. Ignoring \
                            errors..." % (axis_name)

                    # parse the given matrix file into cov mat
                    cov_mats.append(parse_matrix_file(cov_mat_files[axis_id]))
                else:
                    # don't load any cov mat for that axis
                    cov_mats.append(None)

        print cov_mats
        return Dataset(data=(dataset_kwargs['xdata'], dataset_kwargs['ydata']),
                       cov_mats=cov_mats,
                       title=title)

    else:   # if not covariance matrices given, build dataset without them

        return build_dataset(**dataset_kwargs)


def parse_matrix_file(file_like, delimiter=None):
    '''
    Read a matrix from a matrix file. The format of the matrix file should be::

        # comment row
        a_11  a_12  ...  a_1M
        a_21  a_22  ...  a_2M
        ...   ...   ...  ...
        a_N1  a_N2  ...  a_NM

    **file_like** : string or file-like object
        File path or file object to read matrix from.

    *delimiter* : ``None`` or string (optional)
        Column delimiter use in the matrix file. Defaults to ``None``,
        meaning any whitespace.
    '''

    # Read all lines from the file
    try:
        tmp_lines = file_like.readlines()
    except:
        tmp_f = open(file_like)
        tmp_lines = tmp_f.readlines()
        tmp_f.close()

    # actual file parsing
    result = []
    for line in tmp_lines:  # go through the lines of the file

        if '#' in line:
            # ignore anything after a comment sign (#)
            line = split(line, '#')[0]

        # ignore empty lines
        if (not line) or (line.isspace()):
            continue

        # get field contents by splitting lines
        if delimiter is None:
            tmp_fields = split(line)  # split line on whitespace
        else:
            tmp_fields = split(line, delimiter)  # split line on delimiter

        # turn them into floats
        tmp_fields = map(float, tmp_fields)

        # append those contents to the right fields
        result.append(tmp_fields)

    return np.asmatrix(result)  # return np.matrix as result
