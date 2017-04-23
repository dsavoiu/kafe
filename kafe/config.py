'''
.. module:: config
   :platform: Unix
   :synopsis: A submodule for working with config files and the filesystem.

.. moduleauthor:: Daniel Savoiu <daniel.savoiu@cern.ch>
'''

import sys
import os
try:
    import ConfigParser
except ImportError:
    import configparser as ConfigParser
import kafe

# import main logger for kafe
import logging
logger = logging.getLogger('kafe')

# initialize a file object pointing to /dev/null (for scrapping output)
DEV_NULL_FILE_OBJECT = open(os.devnull, 'w')

def null_file():
    global DEV_NULL_FILE_OBJECT
    if DEV_NULL_FILE_OBJECT.closed:
        logger.warn("Output dump file '%s' has been closed before program "
                    "exit. Reopening..." % (os.devnull,))
        DEV_NULL_FILE_OBJECT = open(os.devnull, 'w')
    return DEV_NULL_FILE_OBJECT

def log_file(file_relative_path):
    '''Returns correct location for placing log files.'''
    _hdir = '.kafe'

    # create '.kafe' directory, if not present
    if not os.path.exists(_hdir):
        os.makedirs(_hdir)
    elif not os.path.isdir(_hdir):  # this shouldn't happen, but if it does...
        logger.error('Error finding log file path for `%s\'. Node `%s\' is not '
                     'a directory.' % (file_relative_path, _hdir))
        return file_relative_path  # write to parent path instead (!)

    # prepend '.kafe' to given file path
    return os.path.join(_hdir, file_relative_path)

def create_config_file(config_type, force=False):
    """
    Create a kafe config file.

    **config_type** : 'user' or 'local'
        Create a 'user' config file in '~/.config/kafe' or a
        'local' one in the current directory.

    *force* : boolean (optional)
        If true, overwrites existing files.

    """

    global kafe_configs, logger

    if config_type not in ('user', 'local'):
        raise ValueError("config_type is not 'user' or 'local'")

    _dir, _filename = kafe_configs[config_type]

    # create a user-level config directory
    if config_type == 'user':
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        else:
            logger.warning("User config directory '%s' already exists." \
                           % (_dir,))

    _file = os.path.join(_dir, _filename)
    if force or (not os.path.exists(_file)):
        # copy the default config file
        import shutil
        kafe_default_config_file = \
            os.path.join(kafe.__path__[0], 'config', 'kafe.default.conf')
        shutil.copyfile(kafe_default_config_file, _file)
        logger.debug("Successfully copied default config "
                     "from '%s' to '%s'." \
                     % (kafe_default_config_file, _file))
    else:
        raise Exception("Cannot create config file at '%s': file exists"
                        % (_file,))


# initialize config parser
cp = ConfigParser.ConfigParser()

# Check for kafe.conf in local directory
if os.path.isfile("kafe.conf"):
    logger.info("Local 'kafe.conf' found in '%s'." % (os.getcwd()))
    cp.read("kafe.conf")

# List of places to look for config files
kafe_configs = {
    'local':  (os.path.abspath(sys.path[0]),             'kafe.conf'),
    'user':   (os.path.expanduser('~/.config/kafe'),     'kafe.conf'),
    'system': (os.path.join(kafe.__path__[0], 'config'), 'kafe.default.conf')
}

# Raise Error is no default config file found
if not os.path.exists(os.path.join(*kafe_configs['system'])):
    raise IOError("No default config file for kafe was "
                  "found on the system but there should "
                  "be one at '%s'. Please check your "
                  "installation. "
                   % (os.path.join(*kafe_configs['system']),))

# Load all found config files into the ConfigParser
kafe_config_file = None
for _type in ('system', 'user', 'local'):
    _dir, _filename = kafe_configs[_type]
    if os.path.exists(_dir):
        _file = os.path.join(_dir, _filename)
        if os.path.isfile(_file):
            cp.read(_file)
            logger.debug("Read '%s' config file at '%s'." % (_type, _file))

# Import "constants" from the configuration

G_PADDING_FACTOR_X = cp.getfloat('Plot', 'padding_factor_x')
G_PADDING_FACTOR_Y = cp.getfloat('Plot', 'padding_factor_y')
G_PLOT_POINTS = cp.getint('Plot', 'plot_points')

M_MINIMIZER_TO_USE = cp.get('Minuit', 'minimizer_to_use')
M_TOLERANCE = cp.getfloat('Minuit', 'tolerance')
M_MAX_ITERATIONS = cp.getint('Minuit', 'max_iterations')
M_MAX_X_FIT_ITERATIONS = cp.getint('Minuit', 'max_x_fit_iterations')

F_SIGNIFICANCE_LEVEL = cp.getfloat('Fit', 'hyptest_significance')

FORMAT_ERROR_SIGNIFICANT_PLACES = cp.getint('Formatting', 'significant_error_places')

D_DEBUG_MODE = cp.getboolean('Debug', 'debug_mode')

G_FIT_INFOBOX_TITLE = cp.get('Plot', 'fit_infobox_title')

G_MATPLOTLIB_BACKEND = cp.get('Plot', 'matplotlib_backend')
