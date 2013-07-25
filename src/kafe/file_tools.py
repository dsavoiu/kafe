'''
.. module:: file_tools
   :platform: Unix
   :synopsis: This submodule provides a set of helper functions for parsing files

.. moduleauthor:: Daniel Savoiu <daniel.savoiu@ekp.kit.edu>
'''

from string import join, split 
from dataset import Dataset, build_dataset
import numpy as np

def parse_column_data(file_to_parse, field_order='x,y', delimiter=' '):
    '''
    Parses a file which contains measurement data in a one-measurement-per-row format.
    The field (column) order can be specified. It defaults to `x,y'. Valid field names are
    `x`, `y`, `xabsstat`, `yabsstat`, `xrelstat`, `yrelstat`, `xabssyst`. Another
    valid field name is `ignore` which can be used to skip a field.
    
    Every valid measurement data file *must* have an `x` and a `y` field.
    
    Additionally, a delimiter can be specified. If this is a whitespace character or omitted, any
    sequence of whitespace characters is assumed to separate the data.
    
    **file_to_parse** : file-like object or string containing a file path
        The file to parse.
    
    *field_order* : string (optional) 
        A string of comma-separated field names giving the order of the columns in the file. Defaults to ``'x,y'``.
    
    *delimiter* : string (optional)
        The field delimiter used in the file.
    
    '''
    
    try:
        tmp_lines = file_to_parse.readlines()   # try to read the lines of the file
    except AttributeError:                      # this will fail if a file path string was passed, so alternatively:
        tmp_file = open(file_to_parse, 'r')     # open the file pointed to by the path
        tmp_lines = tmp_file.readlines()        # and then read the lines of the file
        tmp_file.close()
    
    # define a dictionary of fields (lists) to populate
    fields = {'x':[],
              'y':[],
              'xabsstat':[],
              'yabsstat':[],
              'xrelstat':[],
              'yrelstat':[],
              'ignore':[]
              }
    
    # Error handling in case of invalid field order
    if ',' in field_order:
        field_order_list = []
        for field in split(field_order, ','):   # go through the fields
            if field not in fields.keys():      # raise error for invalid fields
                raise SyntaxError, "Supplied field order `%s' contains invalid field `%s'." % (field_order,field)
            elif field in field_order_list:     # raise error for repeated fields
                raise SyntaxError, "Supplied field order `%s' contains repeated field `%s'." % (field_order,field)
            else:                               # validate field
                field_order_list.append(field)
        for axis in ('x','y'):
            if axis not in field_order_list:
                raise SyntaxError, "Supplied field order `%s' missing mandatory field `%s'." % (field_order,axis)
    else:
        raise SyntaxError, "Supplied field order `%s' is not a comma-separated list of valid fields." % (field_order,)
    
    # handle delimiter
    if delimiter in ['', ' ', '\t']: # if delimiter is a whitespace character
        delimiter=None               # set to None
        
    # actual file parsing
    for line in tmp_lines:                      # go through the lines of the file
        
        if '#' in line:
            line = split(line, '#')[0]          # ignore anything after a comment sign (#)
            
        if (not line) or (line.isspace()):      # ignore empty lines
            continue
        
        # get field contents by splitting lines
        if delimiter is None:
            tmp_fields = split(line)            # split line on whitespace
        else:
            tmp_fields = split(line, delimiter) # split line on delimiter
    
        # append those contents to the right fields
        for idx in range(len(field_order_list)):
            fields[field_order_list[idx]].append( float(tmp_fields[idx]) )
            
    # construct Dataset object
    dataset_kwargs = {}
    for key in fields.keys():
        if fields[key]:           # if the field is not empty
            if key in ('x', 'y'): # some syntax translation needed (x -> xdata) for Dataset constructor
                dataset_kwargs[key+'data'] = np.asarray(fields[key])
            else:
                dataset_kwargs[key] = np.asarray(fields[key])

    return build_dataset(**dataset_kwargs)

if __name__ == '__main__':
    myDataset = parse_column_data('../examples/linear_2par/linear_data_xyxerryerr.ssv', 'x,y,xabsstat,yabsstat')
    print myDataset.get_formatted()