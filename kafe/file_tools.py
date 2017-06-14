'''
.. module:: file_tools
    :platform: Unix
    :synopsis: This submodule provides a set of helper functions for parsing
        files

.. moduleauthor:: Daniel Savoiu <daniel.savoiu@cern.ch>
.. moduleauthor:: Guenter Quast <G.Quast@kit.edu>
'''

import numpy as np
import os, sys

from .dataset import Dataset
from .dataset_tools import build_dataset
from .fit import build_fit

# import main logger for kafe
import logging
logger = logging.getLogger('kafe')


def parse_column_data(file_to_parse, field_order='x,y', delimiter=' ',
                      cov_mat_files=None, title="Untitled Dataset",
                      basename=None, axis_labels=['x', 'y'],
                      axis_units=['', '']):
    '''
    Parses a file which contains measurement data in a one-measurement-per-row
    format. The field (column) order can be specified. It defaults to
    ``"x,y"``.
    Valid field names are `x`, `y`, `xabserr`, `yabserr`, `xrelerr`,
    `yrelerr`. Another valid field name is `ignore` which can be used to skip
    a field.

    A certain type of field can appear several times. If this is the case, all
    specified errors are added in quadrature:

    .. math::

        \\sigma_{\\text{tot}} = \\sqrt{{\\sigma_1}^2+{\\sigma_2}^2+\\dots}

    Every valid measurement data file *must* have an `x` and a `y` field.

    For more complex error models, errors and correlations may be specified as
    covariance matrices. If this is desired, then any number of covariance
    matrices (stored in separate files) may be specified for an axis by
    using the `cov_mat_files` argument.

    Additionally, a delimiter can be specified. If this is a whitespace
    character or omitted, any sequence of whitespace characters is assumed to
    separate the data.

    Parameters
    ----------

    **file_to_parse** : file-like object or string containing a file path
        The file to parse.

    *field_order* : string, optional
        A string of comma-separated field names giving the order of the columns
        in the file. Defaults to ``'x,y'``.

    *delimiter* : string, optional
        The field delimiter used in the file. Defaults to any whitespace.

    *cov_mat_files* : *several* (see below), optional
        This argument defaults to ``None``, which means no covariance matrices
        are used. If covariance matrices are needed, a tuple with two entries
        (the first for `x` covariance matrices, the second for `y`) must be
        passed.

        Each element of this tuple may be either ``None``, a file or file-like
        object, or an iterable containing files and file-like objects. Each
        file should contain a covariance matrix for the respective axis.

        When creating the :py:obj:`Dataset`, all given matrices are summed over.

    *title* : string, optional
        The title of the :py:obj:`Dataset`.

    *basename* : string or ``None``, optional
        A basename for the :py:obj:`Dataset`. All output files related to this dataset
        will use this as a basename. If this is ``None`` (default), the
        basename will be inferred from the filename.

    *axis_labels* : 2-tuple of strings, optional
        a 2-tuple containing the axis labels for the :py:obj:`Dataset`. This is
        relevant when plotting :py:obj:`Fits` of the :py:obj:`Dataset`, but is ignored when
        plotting more than one :py:obj:`Fit` in the same :py:obj:`Plot`.

    *axis_units* : 2-tuple of strings, optional
        a 2-tuple containing the axis units for the :py:obj:`Dataset`. This is
        relevant when plotting :py:obj:`Fits` of the :py:obj:`Dataset`, but is ignored when
        plotting more than one :py:obj:`Fit` in the same :py:obj:`Plot`.

    Returns
    -------

    ::py:class:`~kafe.dataset.Dataset`
        A `Dataset` built from the parsed file.

    '''
    # -- helper function to read matrix from file
    def parse_matrix_file(file_like, delimiter=None):
        '''
        Read a matrix from a matrix file. The format of the matrix file should be::

            # comment row
            a_11  a_12  ...  a_1M
            a_21  a_22  ...  a_2M
            ...   ...   ...  ...
            a_N1  a_N2  ...  a_NM

        Parameters
        ----------

        **file_like** : string or file-like object
            File path or file object to read matrix from.

        *delimiter* : ``None`` or string, optional
            Column delimiter use in the matrix file. Defaults to ``None``,
            meaning any whitespace.

        Returns
        -------

        *numpy.matrix*
            matrix read from file
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
                line = line.split('#')[0]

            # ignore empty lines
            if (not line) or (line.isspace()):
                continue

            # get field contents by splitting lines
            if delimiter is None:
                tmp_fields = line.split()
            else:
                tmp_fields = line.split(delimiter)

            # turn them into floats
            tmp_fields = list(map(float, tmp_fields))

            # append those contents to the right fields
            result.append(tmp_fields)

        return np.asmatrix(result)  # return np.matrix as result
    # -- end helper function

    try:
        # try to read the lines of the file
        tmp_lines = file_to_parse.readlines()
        logger.info("Reading column data (%s) from file: %r"
                    % (field_order, file_to_parse))
        # this will fail if a file path string was passed, so alternatively:
    except AttributeError:
        if basename is None:
            # get the basename from the path
            _basename = os.path.basename(file_to_parse)
            # remove the last extension (usually '.dat')
            basename = '.'.join(_basename.split('.')[:-1])
        # open the file pointed to by the path
        tmp_file = open(file_to_parse, 'r')
        logger.info("Reading column data (%s) from file: %s"
                    % (field_order, file_to_parse))
        # and then read the lines of the file
        tmp_lines = tmp_file.readlines()
        tmp_file.close()

    # if basename still unset, set it to 'untitled'
    if basename is None:
        basename = 'untitled'

    # define a dictionary of fields (lists) to populate
    fields = {'x': [],
              'y': [],
              'xabserr': [],
              'yabserr': [],
              'xrelerr': [],
              'yrelerr': [],
              'ignore': []}

    # define a list of axes
    axes = ('x', 'y')

    # Error handling in case of invalid field order
    if ',' in field_order:
        field_order_list = []
        for field in field_order.split(','):  # go through the fields
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
            line = line.split('#')[0]

        if (not line) or (line.isspace()):  # ignore empty lines
            continue

        # get field contents by splitting lines
        if delimiter is None:
            tmp_fields = line.split()
        else:
            tmp_fields = line.split(delimiter)

        # append those contents to the right fields
        for idx, field_name in enumerate(field_order_list):
            fields[field_name].append(float(tmp_fields[idx]))

    # gather kwargs for Dataset object
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

    dataset_kwargs.update({'title': title, 'basename': basename,
                           'axis_labels': axis_labels,
                           'axis_units': axis_units})

    # parse additional covariance matrix files, if necessary
    if cov_mat_files is not None:
        try:
            if len(cov_mat_files) != len(axes):
                raise SyntaxError(
                    "cov_mat_files tuple length (%d) doesn't match "
                    "number of axes, ignoring... " % (len(cov_mat_files),)
                )
        except:
            raise SyntaxError(
                "Invalid cov_mat_files specification, "
                "ignoring... Expected 2-tuple of strings/file objects "
                "instead of %r." % (cov_mat_files,)
            )
        else:
            cov_mats = []
            for axis_id, axis_name in enumerate(axes):
                if cov_mat_files[axis_id] is not None:
                    # if cov mat is given, check for iterability ( == more
                    # than one covariance matrix)
                    # (need to check explicitly if tuple or list,
                    # because strings and files are also iterable)

                    if (
                        isinstance(cov_mat_files[axis_id], tuple) or
                        isinstance(cov_mat_files[axis_id], list)
                    ):
                        # we have more than one cov_mat
                        current_cov_mat = None  # initialize to None, for now

                        # go through each matrix file for this axis
                        for tmp_mat_file in cov_mat_files[axis_id]:
                            # read the matrix
                            tmp_mat = parse_matrix_file(tmp_mat_file)
                            # add to previous cov_mats for this axis
                            if current_cov_mat is None:
                                current_cov_mat = tmp_mat
                            else:
                                current_cov_mat += tmp_mat
                    else:
                        # ony one cov_mat for the axis:
                        # parse the given matrix file into cov mat
                        current_cov_mat = parse_matrix_file(
                            cov_mat_files[axis_id]
                        )

                    # append to cov_mats to pass to :py:obj:`Dataset`
                    cov_mats.append(current_cov_mat)

                else:
                    # don't load any cov mat for that axis
                    cov_mats.append(None)

                dataset_kwargs['cov_mats'] = cov_mats

    #return dataset_kwargs
    return build_dataset(**dataset_kwargs)

def parse_general_inputfile(file_to_parse):
    '''
    This function can be used to specify `kafe`
    :py:class:`~kafe.dataset.Dataset` or :py:class:`~kafe.fit.Fit` objects
    in a single input file, thus requiring minimal Python code. Keywords as
    specified in a dictionary ``tokens`` specify all objects and parameters
    needed by the functions :py:func:`~kafe.dataset_tools.build_dataset` in
    module :py:mod:`~kafe.dataset` and :py:func:`~kafe.fit.build_fit` in
    module :py:mod:`~kafe.fit`.

    Parameters
    ----------

    **file_to_parse** : file-like object or string containing a file path
       The file to parse.

    Returns
    -------

    (dataset_kwargs, fit_kwargs)
          keyword lists to build a kafe :py:class:`~kafe.dataset.Dataset` or
          :py:class:`~kafe.fit.Fit` object with the helper functions
          `build_dataset` or `build_fit`


    **Input file format**

    The interpretation of the input data is driven by *keywords*. All data
    following a key must be of the same kind. A block of data ends when a
    new key is specified. Comments can be introduced by ``#``.

    Some keys only expect a single float or string-type value, given
    on the same line, separated by a space (``' '``)::

        <key> <value>

    Other keys require multiple lines of input. For instance, the keys
    ``*xData`` and ``*yData`` expect the following lines to be a table where
    the first column corresponds to the data values and the second column
    corresponds to the uncertainties::

        <key>
        <value1>  <uncertainty1>
        <value2>  <uncertainty2>
        ...
        <valueN>  <uncertaintyN>

    The column separator is space (``' '``). For more details about input
    data specification, see :ref:`below <specifying_input_data>`.

    **Specifying metadata**

        .. tabularcolumns:: |l|l|

        +---------------+------------------------+
        | **Key**       | **Description**        |
        +===============+========================+
        | ``*TITLE``    | name of the dataset    |
        +---------------+------------------------+
        | ``*BASENAME`` | name from which output |
        |               | file names is derived  |
        +---------------+------------------------+
        | ``*FITLABEL`` | fit label              |
        |               |                        |
        +---------------+------------------------+
        | ``*xLabel``   | x axis label           |
        +---------------+------------------------+
        | ``*xUnit``    | x axis unit            |
        +---------------+------------------------+
        | ``*yLabel``   | y axis label           |
        +---------------+------------------------+
        | ``*yUnit``    | y axis unit            |
        +---------------+------------------------+

    The fit label may be set using the key ``*FITLABEL``, followed by the
    desired name for the fit.

    .. _specifying_input_data:

    **Specifying input data**

    Input data are given as a list of values (one datapoint per row). For a
    simple uncertainty model (no correlations), the keys ``*xData`` and
    ``*yData`` are used. The second column indicates the uncertainty of the
    measurement::

        *xData
        1.2
        3.4
        6.9

        *yData
        2.1       0.2
        3.9       0.3
        8.2       0.5

    .. NOTE::
       Uncertainties always have to be specified for ``*yData``. For
       ``*xData``, they are optional.

    For input data with correlated uncertainties, the alternative keys
    ``*xData_COR`` and ``*yData_COR`` are provided. For these, additional
    columns must be given. The second and third column indicate the
    uncorrelated and correlated uncertainties, respectively. The subequent
    columns contain the correlation matrix (a lower triangular matrix
    containing the correlation coefficients)::

        *yData_COR
        # value  indep.uncert.  syst.uncert.  elements of corr. matrix.
        2.1      0.2            0.1
        3.9      0.3            0.2           1.0
        8.2      0.5            0.3           1.0        1.0

    .. NOTE::
       Only elements below the main diagonal of the correlation matrix have
       to be specified. Since the matrix is symmetric by construction, the
       elements above the main diagonal can be inferred from those below.
       Additionally, since the diagonal elements of a correlation matrix are
       always equal to 1 by definition, they are also omitted.

    As an alternative to specifying the correlation matrix, the covariance
    matrix may be specified directly. There are two ways to do this:

    The keys ``*xData_SCOV`` and ``*yData_SCOV`` allow specifying the
    covariance matrix by providing a correlated uncertainty (third column)
    and the square root of the elements below the main diagonal. This is
    useful if the pairwise covariance of two measurements cannot be
    expressed using the correlation coefficient and needs to be provided
    explicitly.

    In the example below, there is a correlation between the first two and
    the last two measurements, which is estimated under the assumption that
    the smaller of the two uncertainties represents a common error::

        *yData_SCOV
        # mH      err    syst   sqrt(cov)
        124.51   0.52    0.06
        125.60   0.40    0.20   0.06
        125.98   0.42    0.28   0.   0.
        124.70   0.31    0.15   0.   0.  0.15

    A second possibility is specifying the full covariance matrix directly.
    This is achieved using the ``*xData_COV`` and ``*yData_COV`` keywords.
    In this case, only the data values and the uncorrelated uncertainties
    (first and second columns, respectively) must be specified in addition
    to the covariance matrix (all other columns). All entries starting with
    the third column are assumed to be covariance matrix elements. The
    matrix is symmetric, so elements above the diagonal are omitted. Note
    that the diagonal must be specified and corresponds to the squares of
    the correlated errors::

        *yData_COV
        # mH      err    cov_ij
        124.51   0.52    0.0036
        125.60   0.40    0.0036  0.04
        125.98   0.42    0.      0.    0.0784
        124.70   0.31    0.      0.    0.0225  0.0225

    **Specifying additional uncertainties**

    In addition to the uncertainties already specified in the
    :ref:`input data table <specifying_input_data>`, other systematic
    uncertainties may be provided. These are assumed be fully correlated and
    common to all data points. This can be achieved by using the following
    keys:

        +---------------+---------------------------------------------------+
        | **Key**       | **Description**                                   |
        +===============+===================================================+
        | ``*xAbsCor``  | common fully correlated x-uncertainty (absolute)  |
        +---------------+---------------------------------------------------+
        | ``*yAbsCor``  | common fully correlated y-uncertainty (absolute)  |
        +---------------+---------------------------------------------------+
        | ``*xRelCor``  | common fully correlated x-uncertainty (relative)  |
        +---------------+---------------------------------------------------+
        | ``*yRelCor``  | common fully correlated y-uncertainty (relative)  |
        +---------------+---------------------------------------------------+

    **Specifying a fit function**

    To specify the fit function, the key ``*FitFunction`` is provided. This
    key should be followed by *Python* code::

          def fitf(x, ...):
              ...
              return ...

    .. NOTE::
       Only one Python function may be defined after the ``*FitFunction``
       keyword. Also, any function name can be used instead of ``fitf``.

       Additionally, the decorators ``@ASCII``, ``@LaTeX`` and
       ``@FitFunction`` are supported (see
       :py:class:`~kafe.function_tools.ASCII`,
       :py:class:`~kafe.function_tools.LaTeX` and
       :py:class:`~kafe.function_tools.FitFunction`)


    **Specifying initial values for parameters**

    Initial values for fit parameters may be set using the keyword
    ``*InitialParameters``. This keyword expects to be followed by a table
    with two columns containing floating-point values.

    Each line in the table corresponds to one fit parameter, in the order
    they are given in the fit function signature. The first column should
    contain the initial value of the parameters and the second column the
    "initial uncertainty", which controls the initial variation range of
    the parameter at the beginning of the fit::

        *InitialParameters
        <initial value par 1>  <initial uncert par 1>
        <initial value par 2>  <initial uncert par 2>
        ...
        <initial value par N>  <initial uncert par N>

    **Constraining parameters**

    If there is any prior knowledge about model parameters' values on
    uncertainties, these may be constrained during the fit.

    During the fit, model parameters can be constrained within their
    uncertainties if there is any prior knowledge about their values and
    uncertainties.

    This may be specified using the keyword ``*ConstrainedParameters``, followed
    by a table containing the parameter name, value and uncertainty for each parameter
    to be constrained::

          <parameter name>  <parameter value>  <parameter uncert.>,

    .. NOTE::
       The parameter name must be the one specified in the fit function definition.

    **Example**

    Here is an example of an input file to calculate the average of four
    partly correlated measurements (see :ref:`Example 8 <example_8>`)::


         #  Meta data for plotting
         *TITLE Higgs-mass measurements
         *xLabel number of measurement
         *yLabel $m_\mathrm{H}$
         *yUnit GeV/$c^2$

         #*xData  # commented out, as not needed for simple average

         *yData_SCOV  # assume that minimum of syst. errors is a common error
         # mH      err     syst as sqrt(cov)
         124.51   0.52    0.06
         125.60   0.40    0.20  0.06
         125.98   0.42    0.28  0.   0.
         124.70   0.31    0.15  0.   0.  0.15

         *FitFunction  # Python code of fit function

         #  kafe fit function decorators are supported
         @ASCII(expression='av')
         @LaTeX(name='f', parameter_names=('av'), expression='av')
         @FitFunction
         def fitf(x, av=1.0): # fit an average
             return av

         *FITLABEL Average

         *InitialParameters
         120. 1.

    .. sectionauthor::  Guenter Quast <G.Quast@kit.edu>
    '''
    #---------------------------------------------------------------------
    # modification log:
    #   G.Q., 27-JUL-14 <initial version>
    #   G.Q., 12-Aug-14 added tags `*Fiffunction`,`*InitialParameters`,
    #                     `*ConstrainedParameters`
    #   G.Q., 15-Aug-14 import of fitf via exec statement
    #   D.S.,  2-Jul-15 re-indent, pep8 conformity
    #---------------------------------------------------------------------

    # define a dictionary for tags from file
    #    values specify expected input type,
    #    are overwritten with read values by input from file
    tokens = {"*TITLE": '',      # title of plot (''=string)
              "*xLabel": '',     # x-axis label
              "*yLabel": '',     # y-axis label
              "*xUnit": '',      # x-axis unit
              "*yUnit": '',      # y-axis unit
              "*BASENAME": '',       # name for the data set
              "*xData": 'arr',       # x values, errors (arr=array)
              "*yData": 'arr',       # y values, errors
              "*xData_COR": 'arr',   # x values, errors, syst. & COR
              "*yData_COR": 'arr',   # y values, errors, syst. & COR
              "*xData_COV": 'arr',   # x values, errors, syst. & COV
              "*yData_COV": 'arr',   # y values, errors, syst. & COV
              "*xData_SCOV": 'arr',  # x values, errors, syst. & sqrt(COV)
              "*yData_SCOV": 'arr',  # y values, errors, syst. & sqrt(COV)
              "*xAbsCor": 'f',       # common x-error  (f=float32)
              "*yAbsCor": 'f',       # common y-error
              "*xRelCor": 'f',       # common, relative x-error
              "*yRelCor": 'f',       # common, relative y-error
              #
              "*FITLABEL": '',        # name for Fit
              "*FITNAME": '',         # name for the data set
              "*FitFunction": 'arr',  # read python code with function to fit
              "*InitialParameters": 'arr',  # initial values and range of pars
              "*ConstrainedParameters": 'arr'  # parameter constraints
              }

    setkeys = []  # remember specified keys

    # define character for comments
    ccomment = "#"
    # --- helpers for parse_general_inputfile -------------------------------

    def get_inputlines(f, comment_character='#'):
        # remove comments, empty lines and extra spaces from input file
        inputlines = []

        try:
            tmp_lines = f.readlines()
        except AttributeError:
            tmp_f = open(f, 'r')
            logger.info("Reading data from file: %s" % (f))
            tmp_lines = tmp_f.readlines()
            tmp_f.close()
        finally:
            logger.info("Reading data from file: %r" % (f))

        # pre-process each line
        for line in tmp_lines:
            # remove comments
            if comment_character in line:
                line = line.split(comment_character)[0]
            # skip empty lines
            if not line or line.isspace():
                continue
            else:
                inputlines.append(line)

        return inputlines

    def data_from_SCOV(flist):
        # decode data with covariance matrix
        # given as lower triangular matrix of sqrt of elements
        size = len(flist)
        dat = np.array([flist[i][0] for i in range(size)], np.float32)
        err = np.array([flist[i][1] for i in range(size)], np.float32)
        sys = np.array([flist[i][2] for i in range(size)], np.float32)
        cov = np.zeros((size, size), np.float64)
        for i in range(1, size):
            for j in range(0, i):
                cov[i, j] = np.float64(flist[i][j + 3]) ** 2
                cov[j, i] = cov[i, j]
        np.fill_diagonal(cov, sys * sys)
        return dat, err, cov

    def data_from_COV(flist):
        # decode data with full covariance matrix
        size = len(flist)
        dat = np.array([flist[i][0] for i in range(size)], np.float32)
        err = np.array([flist[i][1] for i in range(size)], np.float32)
        cov = np.zeros((size, size), np.float64)
        for i in range(1, size):
            for j in range(1, size):
                cov[i, j] = np.float64(flist[i][j + 2])
        return dat, err, cov

    def data_from_COR(flist):
        # decode data with covariance matrix given
        # given as lower triangular matrix of correlation coefficients
        size = len(flist)
        dat = np.array([flist[i][0] for i in range(size)], np.float32)
        err = np.array([flist[i][1] for i in range(size)], np.float32)
        sys = np.array([flist[i][2] for i in range(size)], np.float32)
        cov = np.zeros((size, size), np.float64)
        for i in range(1, size):
            for j in range(0, i):
                cov[i, j] = np.float64(flist[i][j + 3]) * sys[i] * sys[j]
                cov[j, i] = cov[i, j]
        np.fill_diagonal(cov, sys * sys)
        return dat, err, cov

    def parse_sanitize_fitf_code(code_string):
        '''Parse and sanitize Python code'''
        import tokenize
        import string
        try:
            from cStringIO import StringIO
        except ImportError:
            from io import StringIO

        # backwards compatibility: replace tildes by spaces
        code_string = code_string.replace('~', ' ')

        FORBIDDEN_TOKENS = ['import', 'exec', 'global', 'execfile']

        # parser flags
        _reading_decorator = False
        _reading_function_def = False
        _expect_function_name = True
        _reading_function_body = False
        _done_reading_function = False

        function_name = ""
        sanitized = []
        _tokens = tokenize.generate_tokens(StringIO(code_string).readline)   # tokenize the input
        for toknum, tokval, spos, epos, line_string  in _tokens:
            ##print "\tLine: '%s'" % (line_string[:-1],)
            # disallow forbidden tokens
            for _ftoken in FORBIDDEN_TOKENS:
                if tokval == _ftoken:
                    # encountered forbidden token -> throw error
                    _e = "Encountered forbidden token '%s' in user-entered code on line '%s'." % (tokval, line_string)
                    logger.error(_e)
                    raise ValueError(_e)

            # begin reading decorator
            if toknum == tokenize.OP and tokval == '@':
                _reading_decorator = True

            # read all tokens between decorator token and newline
            if _reading_decorator:
                #print 'TOKEN\t', tokenize.tok_name[toknum], tokval
                sanitized.append((toknum, tokval))
                if toknum == tokenize.NEWLINE:
                    _reading_decorator = False

            # check for function definition
            if tokval == 'def':
                if _done_reading_function:
                    # warn on additional function definitions
                    _e = "Already read fit function '%s'. No additional function definitions supported. Line: '%s'" % (function_name, line_string[:-1])
                    logger.warn(_e)
                    continue
                else:
                    _reading_function_def = True
                    _expect_function_name = True
                    _indent_level = 0  # set reference indent level to that of function def
                    sanitized.append((toknum, tokval))
                    # encountered 'def'. now expecting to read function name
                    continue
            elif not _reading_function_def and not _reading_function_body:
                # not encountered first 'def' statement yet
                continue

            # read function name
            if _expect_function_name:
                function_name = tokval
                # function nake should be <tokval>
                _expect_function_name = False

            # begin reading function body
            if _reading_function_def and toknum == tokenize.INDENT:
                # encountered INDENT while reading function def
                # now expecting to read function body...
                _reading_function_def = False
                _reading_function_body = True

            if not _done_reading_function:
                sanitized.append((toknum, tokval))

            # keep track of the indent level
            if toknum == tokenize.DEDENT:
                _indent_level -= 1
            elif toknum == tokenize.INDENT:
                _indent_level += 1

            # encountered end of function
            if _indent_level == 0 and _reading_function_body:
                _done_reading_function = True

        return function_name, tokenize.untokenize(sanitized)

    # --- end helpers for parse_general_inputfile ---------------------------

    # ---- parse input file

    # read file, skipping everything including and following comment character
    inputlines = get_inputlines(file_to_parse, comment_character=ccomment)

    # split input into words, separator is ' '
    current_key = '!'
    for i, line in enumerate(inputlines):
        # separate first token from the rest
        try:
            leading_token, coda = line.split(None, 1)
        except ValueError:
            # in case there is no coda
            leading_token, coda = line.split(None, 1)[0], ""

        # check if line provides new key or is continuation
        if leading_token in tokens:
            # new key -> new section
            current_key = leading_token
            setkeys.append(current_key)

            # handle keys based on expected continuation
            if tokens[current_key] == '':
                # expected input is string
                tokens[current_key] = ' '.join(coda.split())  # get rest of line
            elif tokens[current_key] == 'f':
                # expected input is float
                _vals = coda.split()
                _val = 0
                if not _vals:
                    _e = "Key '%s' expected a floating point value, but none found!" % (current_key,)
                    logger.error(_e)
                    raise ValueError(_e)
                elif len(_vals) > 1:
                    _w = "Key '%s' expected a single floating point value, but got: %r" % (current_key, _vals)
                    logger.warn(_w)
                    _val = _vals[0]
                else:
                    _val = _vals[0]
                tokens[current_key] = np.float32(_val)
            elif tokens[current_key] == 'arr':
                # expected input is array of data -> empty list
                tokens[current_key] = []
            #~ elif tokens[current_key] == None:
                #~ # expected input is txt (Python code) ->
                #~ pass
        elif leading_token[0] == '*':
            # leading token is unknown
            logger.warn("Unknown input token '%s' ignored" % (leading_token,))
        else:
            # line without key is data, belonging to last key given
            if current_key == '*FitFunction':   # expect continuous text
                # preserve space
                tokens[current_key].append(line) # join words to from line
            elif current_key == '*ConstrainedParameters':
                # expect string + 2 floats
                tokens[current_key].append(
                    [leading_token, float(line[1]), float(line[2])])
            else:
                # expect only floats
                tokens[current_key].append(list(map(float, line.split())))  # expect floats

    # ---- end parse input file

    # ---- now see what we got and decode into
    #            xdata, ydata, xerr, yerr, xcov, ycov
    xdat, xerr, xcov = [], [], []
    for key in setkeys:
        logger.info("valid key %s" % (key))
        if type(tokens[key]) == type([]):
            if (key == '*xData'):
                flist = tokens[key]
                rows = len(flist)
                cols = len(flist[0])
                xdat = np.array([flist[i][0] for i in range(rows)], np.float32)
                if(cols >= 2):
                    xerr = np.array([flist[i][1]
                                    for i in range(rows)], np.float32)
                else:
                    xerr = np.zeros(rows, np.float32)
                xcov = np.zeros((rows, rows), np.float64)
            elif (key == '*yData'):
                flist = tokens[key]
                rows = len(flist)
                cols = len(flist[0])
                ydat = np.array([flist[i][0] for i in range(rows)], np.float32)
                if(cols >= 2):
                    yerr = np.array([flist[i][1]
                                    for i in range(rows)], np.float32)
                else:
                    yerr = np.zeros(rows, np.float32)
                ycov = np.zeros((rows, rows), np.float64)
            elif (key == '*xData_SCOV'):
                xdat, xerr, xcov = data_from_SCOV(tokens[key])
            elif (key == '*yData_SCOV'):
                ydat, yerr, ycov = data_from_SCOV(tokens[key])
            elif (key == '*xData_COR'):
                xdat, xerr, xcov = data_from_COR(tokens[key])
            elif (key == '*yData_COR'):
                ydat, yerr, ycov = data_from_COR(tokens[key])
            elif (key == '*xData_COV'):
                xdat, xerr, xcov = data_from_COV(tokens[key])
            elif (key == '*yData_COV'):
                ydat, yerr, ycov = data_from_COV(tokens[key])
            elif (key == '*FitFunction'):
                #fitcode = text_from_input(tokens[key])
                fitf_name, fitcode = parse_sanitize_fitf_code('\n'.join(tokens[key]))
                logger.info("FitFunction name is '%s'" % (fitf_name,))
            elif (key == '*InitialParameters'):
                flist = tokens[key]
                rows = len(flist)
                cols = len(flist[0])
                parval = np.array([flist[i][0]
                                  for i in range(rows)], np.float32)
                if(cols >= 2):
                    parerr = np.array([flist[i][1]
                                      for i in range(rows)], np.float32)
                else:
                    parerr = None
            elif (key == '*ConstrainedParameters'):
                flist = tokens[key]
                rows = len(flist)
                cols = len(flist[0])
                cparnam = np.array([flist[i][0] for i in range(rows)], str)
                cparval = np.array([flist[i][1]
                                   for i in range(rows)], np.float32)
                cparerr = np.array([flist[i][2]
                                   for i in range(rows)], np.float32)
            else:
                logger.warn("invalid key %s" % (key))
                sys.exit("*==* parse_general_inputfile: unimplemented keyword")

    # some final additions to be made:
    #  x-data may not have been given (e.g. for calcualtion of average)
    if len(xdat) == 0:
        logger.warn("no xdata given - generated as an arange")
        xdat = np.arange(0., len(ydat), 1., np.float32)
        xerr = np.zeros(len(ydat), np.float32)
        xcov = np.zeros((len(ydat), len(ydat)), np.float64)

    # set a name for the data set, if not given yet
    if(tokens['*BASENAME'] == ''):
        if('str' in str(type(file_to_parse))):
        # get the basename from the path
            _basename = os.path.basename(file_to_parse)
             # remove the last extension (usually '.dat')
            tokens['*BASENAME'] = '.'.join(_basename.split('.')[:-1])
        else:
            tokens['*BASENAME'] = 'untitled'

    # if not fit name is given, set to 'None' -> automatic naming
    if tokens['*FITNAME'] == '':
        tokens['*FITNAME'] = None

    # check for additional, common errors and add in quadrature to cov matrices

    for key in setkeys:
        if key == '*xAbsCor':
            xcov += np.float64(tokens[key]) ** 2
        elif key == '*yAbsCor':
            ycov += np.float64(tokens[key]) ** 2
        elif key == '*xRelCor':
            xcov += np.float64(tokens[key]) ** 2 * np.outer(xdat, xdat)
        elif key == '*yRelCor':
            ycov += np.float64(tokens[key]) ** 2 * np.outer(ydat, ydat)

    # build dictionary for dataset
    dataset_kwargs = {}
    dataset_kwargs.update({
                          'xdata': xdat,
                          'ydata': ydat,
                          'xabserr': xerr,
                          'yabserr': yerr,
                          'cov_mats': [xcov, ycov],
                          'title': tokens['*TITLE'],
                          'basename': tokens['*BASENAME'],
                          'axis_labels': [tokens['*xLabel'], tokens['*yLabel']],
                          'axis_units': [tokens['*xUnit'], tokens['*yUnit']]
                          })

    # collect input for fit function
    fit_kwargs = None
    if '*FitFunction' in setkeys:
        prefix = 'from kafe.function_tools import FitFunction, LaTeX, ASCII\n'
        fullcode = prefix + fitcode  # add imports for decorators
        # execute code and place in a special scope
        scope = dict()
        exec(fullcode, scope)        # note: function name must be 'fitf'

        if '*InitialParameters' in setkeys:
            fitpars = parval, parerr
        else:
            fitpars = None
        if '*ConstrainedParameters' in setkeys:
            cpars = cparnam, cparval, cparerr
        else:
            cpars = None

        # build dictionary for build_fit
        fit_kwargs = {}
        fit_kwargs.update({
                          'fit_function': scope[fitf_name],
                          'fit_label': tokens['*FITLABEL'],
                          'fit_name': tokens['*FITNAME'],
                          'initial_fit_parameters': fitpars,
                          'constrained_parameters': cpars
                          })

    return dataset_kwargs, fit_kwargs

# ---- end parse_general_inputfile


def buildDataset_fromFile(file_to_parse):
    '''
    Build a kafe :py:class:`~kafe.dataset.Dataset` object from input file
    with key words and file format defined in
    :py:func:`~kafe.file_tools.parse_general_inputfile`

    Parameters
    ----------

    **file_to_parse** :  file-like object or string containing a file path
       The file to parse.

    Returns
    -------

    :py:class:`~kafe.dataset.Dataset`
       a :py:class:`~kafe.dataset.Dataset` object
       constructed with the help of the method
       :py:func:`kafe.dataset.Dataset.build_dataset`
    '''
    dataset_kwargs, dummy = parse_general_inputfile(file_to_parse)
    return build_dataset(**dataset_kwargs)


def buildFit_fromFile(file_to_parse):
    '''
    Build a kafe :py:class:`~kafe.fit.Fit` object from input file with
    keywords and file format defined in
    :py:func:`~kafe.file_tools.parse_general_inputfile`

    Parameters
    ----------

    **file_to_parse**:  file-like object or string containing a file path
       The file to parse.

    Returns
    -------

    :py:class:`~kafe.fit.Fit`
       a :py:class:`~kafe.fit.Fit` object
       constructed with the help of the methods
       :py:func:`~kafe.dataset.Dataset.build_dataset` and
       :py:func:`~kafe.fit.Fit.build_fit`

    '''
    dataset_kwargs, fit_kwargs = parse_general_inputfile(file_to_parse)
    if fit_kwargs is None:
        sys.exit("*==* no valid fit function specifed in input file")
    else:
        return build_fit(build_dataset(**dataset_kwargs), **fit_kwargs)
