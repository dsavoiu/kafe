'''
.. module:: stream
    :platform: Unix
    :synopsis: A submodule containing an object for simultaneous output to file
        and to ``sys.stdout``.
..  moduleauthor:: Daniel Savoiu <danielsavoiu@gmail.com>
'''

import sys
from time import gmtime, strftime


class StreamDup(object):
    '''
    Object for simultaneous logging to stdout and files.
    This object provides a file/like object for the outout to be written to.
    Writing to this object will write to stdout (usually the console) and to
    a file.

    **out_file** : file path or file-like object or list of file paths ...
        File(s) to which to log the output, along with stdout. If a file exists
        on disk, it will be appended to.

    *suppress_stdout* : boolean
        Whether to log to stdout simultaneously (``False``) or suppress output
        to stdout (``True``). Default to ``False``.
    '''
    def __init__(self, out_file, suppress_stdout=False):
        self.suppress_stdout = suppress_stdout

        # make the out_file a list if this is not the case
        if not isinstance(out_file, list):
            out_file = [out_file]

        self.out_file = []
        for file_like in out_file:
            try:
                file_like.write("")
            except AttributeError:
                self.out_file.append(open(file_like, 'a'))
            else:
                self.out_file.append(file_like)
        self.closed = False

    def close(self):
        for _file in self.out_file:
            if not _file.closed:
                _file.close()
        self.closed = True

    def write(self, message):
        # write to log file(s) AND to stdout
        for _file in self.out_file:
            _file.write(message)
        if not self.suppress_stdout:
            sys.stdout.write(message)

    def write_to_file(self, message):
        for _file in self.out_file:
            _file.write(message)

    def write_to_stdout(self, message, check_if_suppressed=False):
        '''
        Explicitly write to stdout. This method will not check by default
        whether ``suppress_stdout`` is set for this `StreamDup`. If
        ``check_if_suppressed`` is explicitly set to ``True``, then this
        check occurs.
        '''
        if not check_if_suppressed:
            sys.stdout.write(message)
        else:
            if not self.suppress_stdout:
                sys.stdout.write(message)

    def _write_timestamp_to_file(self, prefix, write_to):
        write_to.write('\n')
        write_to.write('#'*(len(prefix)+4+20))
        write_to.write('\n')
        write_to.write("# %s " % (prefix,) +
                       strftime("%Y-%m-%d %H:%M:%S #\n", gmtime()))
        write_to.write('#'*(len(prefix)+4+20))
        write_to.write('\n\n')

    def write_timestamp(self, prefix):
        for _file in self.out_file:
            self._write_timestamp_to_file(prefix, _file)

    def fileno(self):
        '''Returns the file handler id of the main (first) output file.'''
        return self.out_file[0].fileno()

    def flush(self):
        sys.stdout.flush()
        for _file in self.out_file:
            _file.flush()
