'''
.. module:: stream
   :platform: Unix
   :synopsis: A submodule containing an object for simultaneous output to file and to ``sys.stdout``.
.. moduleauthor:: Daniel Savoiu <danielsavoiu@gmail.com>
'''


import sys
from time import gmtime, strftime

class StreamDup(object):
    '''
    Object for simultaneous logging to stdout and a file.
    '''
    def __init__(self, out_file):
        try:
            out_file.write()
        except:
            self.out_file = open(out_file, 'a')
        else:
            self.out_file = out_file
        
    def write(self, message):
        # write to log file AND to stdout
        self.out_file.write(message)
        sys.stdout.write(message)
    
    def write_to_file(self, message):
        self.out_file.write(message)
    
    def write_to_stdout(self, message):
        sys.stdout.write(message)
    
    def write_timestamp(self, prefix):
        self.out_file.write('\n')
        self.out_file.write('#'*(len(prefix)+4+20))
        self.out_file.write('\n')
        self.out_file.write("# %s " % (prefix,) + strftime("%Y-%m-%d %H:%M:%S #\n", gmtime()))
        self.out_file.write('#'*(len(prefix)+4+20))
        self.out_file.write('\n\n')
    
    def fileno(self):
        return self.out_file.fileno()
    
    def flush(self):
        sys.stdout.flush()
        self.out_file.flush()
