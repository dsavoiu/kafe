'''
.. module:: file_tools
    :platform: Unix
    :synopsis: This submodule provides a set of helper functions for parsing
        files

.. moduleauthor:: Daniel Savoiu <danielsavoiu@gmail.com>
.. moduleauthor:: Guenter Quast <G.Quast@kit.edu>
'''

import numpy as np
import os, sys

from string import split, replace
from .dataset import Dataset
from .dataset_tools import build_dataset
from .fit import build_fit

#from importlib import import_module

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

    *field_order* : string (optional)
        A string of comma-separated field names giving the order of the columns
        in the file. Defaults to ``'x,y'``.

    *delimiter* : string (optional)
        The field delimiter used in the file. Defaults to any whitespace.

    *cov_mat_files* : *several* (see below, optional)
        This argument defaults to ``None``, which means no covariance matrices
        are used. If covariance matrices are needed, a tuple with two entries
        (the first for `x` covariance matrices, the second for `y`) must be
        passed.

        Each element of this tuple may be either ``None``, a file or file-like
        object, or an iterable containing files and file-like objects. Each
        file should contain a covariance matrix for the respective axis.

        When creating the :py:obj:`Dataset`, all given matrices are summed over.

    *title* : string (optional)
        The title of the :py:obj:`Dataset`.

    *basename* : string or ``None`` (optional)
        A basename for the :py:obj:`Dataset`. All output files related to this dataset
        will use this as a basename. If this is ``None`` (default), the
        basename will be inferred from the filename.

    *axis_labels* : 2-tuple of strings (optional)
        a 2-tuple containing the axis labels for the :py:obj:`Dataset`. This is
        relevant when plotting :py:obj:`Fits` of the :py:obj:`Dataset`, but is ignored when
        plotting more than one :py:obj:`Fit` in the same :py:obj:`Plot`.

    *axis_units* : 2-tuple of strings (optional)
        a 2-tuple containing the axis units for the :py:obj:`Dataset`. This is
        relevant when plotting :py:obj:`Fits` of the :py:obj:`Dataset`, but is ignored when
        plotting more than one :py:obj:`Fit` in the same :py:obj:`Plot`.
        
    Returns
    -------

    :py:class:`~kafe.dataset.Dataset`
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

        *delimiter* : ``None`` or string (optional)
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
    module :py:mod:`~kake.fit`.
        
    Parameters
    ----------

    **file_to_parse**:  file-like object or string containing a file path
       The file to parse.

    **return** : dataset_kwargs, fit_kwargs
          keyword lists to build a kafe :py:class:`~kafe.dataset.Dataset` or
          :py:class:`~kafe.fit.Fit` object with the helper functions
          `build_dataset` or `build_fit`


    **Description of the format of the input file**

    The interpretation of the input data is driven by keywords.
    All data following a key must be of the same kind, a block of
    data ends when a new key is specified.

    Some keys only expect a single float or string-tpye value, given
    on the same line, separated by a space (``' '``)::

        <key> <value>

    For multiple input, i.e. data, uncertainties and covariance or
    correlation matrices, the format is::

        <key>
        <xval>  <xerr>  [<xsyst>  <elements of cov/cor matrix>]

        ...

        <xval>  <xerr>  [<xsyst>  <elements of cov/cor matrix>]

    The field separator is space (``' '``). Note that the number of input
    values in each line must correspond to the specified format of the
    (correlated) uncertainties.

    The currently implemented keys are:

    * for metadata:

      - ``*TITLE``     <name of the data set>
      - ``*BASENAME``  <name from which output file names are derived>
      - ``*xLabel``    <x axis label>
      - ``*yLabel``    <y axis label>
      - ``xUnit``      <x axis unit>
      - ``yUnit``      <y axis unit>

    * for input data:

      - ``*xData``     `x data and, optionally, uncertainties`
          <xval>  [<x-uncert.>]
           ...
      - ``*yData``     `y data and uncertainties`
          <yval>  <y uncert.>
           ...
    * `x` or `y` data, independent and correlated uncertainties and elements of
      correlation matrix, given as as a lower triangular matrix with no
      diagonal:

      - ``*xData_COR``
      - ``*yData_COR``
         <x/y val>  <indep. x/y uncert.>  <x/y syst>  <elements of cor matrix>
          ...

    * `x` or `y` data, independent and correlated uncertainties and sqrt of
      elements of covariance matrix, given as as a lower triangular matrix
      with no diagonal:

      - ``*xData_SCOV``
      - ``*yData_SCOV``
          <x/y val>  <idep. x/y uncert.>  <x/y syst>  <sqrt of elements of cov matrix>
           ...

    * `x` or `y` data, independent uncertainties and full covariance matrix (note
      that the correlated uncertainties are contained in the diagonal of the
      matrix in this case, i.e. the field <xsyst> is to be omitted):

      - ``*xData_COV``
      - ``*yData_COV``
          <x/y val>  <indep. x/y ucert.>  <elements of cov matrix>
           ...

    * Additional keys allow to specify correlated absolute or relative
      uncertainties:

      - ``*xAbsCor <common abs. x uncert.>``
      - ``*yAbsCor <common abs. y uncert.>``
      - ``*xRelCor <common rel. x uncert.>``
      - ``*yRelCor <common rel. y uncert.>``


    * To specify the fit function, the defined keywords are:

      - ``*FitFunction``  followed by python code (note: blanks for line
        indent must be replaced by '~')::

          def fitf(x, ...):
          ~~~~...
          ~~~~return ...

        The name `fitf` is mandatory. The kafe decorator functions
        ``@ASCII``, ``@LATEX`` and ``@FitFunction``
        are suppoted.

      - ``*FITLABEL`` <the name for the fit>
      - ``*InitialParameters`` -  followed by two columns of float values
        for the initial values of the parameters and their range, one line
        per fit parameter is mandatory

          <initial value>  <range>


    * Model parameters can be constrained within their uncertainties, if prior
      knowledge on the value(s) and uncertainty(ies) of parameters are
      to be accounted for in the fit. This option is specified via the
      keyword:

      - ``*ConstrainedParameters`` followed by one or more lines with
        the fields::

          <parameter name>  <parameter value>  <parameter uncert.>,

        where `parameter name` is the name of the parameter in the fit
        function specification.


    Here is an example of an input file to calculate the average
    of correlated measurements::

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

         # set Python code of fit function
         ### there are some restrictions:
         ##     function name must be 'fitf'
         ##     blanks must be replaced by '~'
         #  kafe fit function decorators are supported
         *FitFunction
         @ASCII(expression='av')
         @LaTeX(name='f', parameter_names=('av'), expression='av')
         @FitFunction
         def fitf(x,av=1.): # fit an average
         ~~~~return av
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
              "*BASENAME": '',      # name for the data set
              "*xData": 'arr',      # x values, errors (arr=array)
              "*yData": 'arr',      # y values, errors
              "*xData_COR": 'arr',  # x values, errors, syst. & COR
              "*yData_COR": 'arr',  # y values, errors, syst. & COR
              "*xData_COV": 'arr',  # x values, errors, syst. & COV
              "*yData_COV": 'arr',  # y values, errors, syst. & COV
              "*xData_SCOV": 'arr',  # x values, errors, syst. & sqrt(COV)
              "*yData_SCOV": 'arr',  # y values, errors, syst. & sqrt(COV)
              "*xAbsCor": 'f',      # common x-error  (f=float32)
              "*yAbsCor": 'f',      # common y-error
              "*xRelCor": 'f',      # common, relative x-error
              "*yRelCor": 'f',      # common, relative y-error
              #
              "*FITLABEL": '',        # name for Fit
              "*FitFunction": 'arr',  # read python code with function to fit
              "*InitialParameters": 'arr',  # initial values and range of pars
              "*ConstrainedParameters": 'arr'  # parameter constraints
              }

    setkeys = []  # remember specified keys

    # define character for comments
    ccomment = "#"
# --- helpers for parse_general_inputfile

    def get_inputlines(f, cc='#'):
        # remove comments, emty lines and extra spaces from input file
        inputlines = []
        # try to read from open file
        try:
            tmp_lines = f.readlines()
            logger.info("Reading data from file: %r" % (f))
        except AttributeError:
            # take argument as file name and try to open it
            tmpfile = open(f, 'r')
            logger.info("Reading data from file: %s" % (f))
            tmp_lines = tmpfile.readlines()
        # remove comment character and everything following it
        for line in tmp_lines:
            if cc in line:
                line = split(line, cc)[0]
            line = line.strip()
            if(not(line == '' or line == '\n')):
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

    def text_from_input(lines):
        # join lines to for a textstring, replace '~' by blanks
        return replace('\n'.join(lines), '~', ' ')

    # --- end helpers for parse_general_inputfile ---------------------------

    # read file, skipping everything including and following comment character
    inputlines = get_inputlines(file_to_parse, ccomment)
    xdat, xerr, xcov = [], [], []
    # split input into words, separator is ' '
    words = map(split, inputlines)
    for i in range(len(inputlines)):
        if words[i][0] in tokens:
            curkey = words[i][0]
            setkeys.append(curkey)
            if tokens[curkey] == '':                  # expected input is string
                tokens[curkey] = ' '.join(words[i][1:])  # get rest of line
            elif(tokens[curkey]) == 'f':   # expected input is float
                tokens[curkey] = np.float32(words[i][1])
            elif(tokens[curkey]) == 'arr':  # expected input is array of data
                tokens[curkey] = []
            elif(tokens[curkey]) == None:  # expected input is txt (python code)
                pass
        else:
            if(words[i][0][0] == '*'):
                logger.warn("unknown input token %s ignored" % (words[i][0]))
                continue
           # line without key is data, belonging to last key given
            if curkey == '*FitFunction':   # expect continuous text
                tokens[curkey].append(
                    ' '.join(words[i][:]))  # join words to from line
            elif curkey == '*ConstrainedParameters':   # expect string + 2 floats
                tokens[curkey].append(
                    [words[i][0], float(words[i][1]), float(words[i][2])])
            else:  # expect only floats
                tokens[curkey].append(map(float, words[i]))  # expect floats

    # ---- now see what we got and decode into
    #            xdata, ydata, xerr, yerr, xcov, ycov
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
                fitcode = text_from_input(tokens[key])
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
        # execute code and place in global scope
        exec fullcode in globals()  # note: function name must be 'fitf'

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
                          'fitfunc': fitf,
                          'fitlabel': tokens['*FITLABEL'],
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

    **returns** : an instance of the :py:class:`~kafe.dataset.Dataset` class,
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

    **returns** :  an instance of the :py:class:`~kafe.fit.Fit` class,
       constructed with the help of the methods
       :py:func:`~kafe.dataset.Dataset.build_dataset` and
       :py:func:`~kafe.fit.Fit.build_fit`

    '''
    dataset_kwargs, fit_kwargs = parse_general_inputfile(file_to_parse)
    if fit_kwargs is None:
        sys.exit("*==* no valid fit function specifed in input file")
    else:
        return build_fit(build_dataset(**dataset_kwargs), **fit_kwargs)
