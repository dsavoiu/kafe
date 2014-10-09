'''
.. module:: config
   :platform: Unix
   :synopsis: A submodule for working with config files.

.. moduleauthor:: Daniel Savoiu <danielsavoiu@gmail.com>
'''

import os
import ConfigParser
import kafe

# import main logger for kafe
import logging
logger = logging.getLogger('kafe')

kafe_config_dir = os.path.expanduser('~/.config/kafe')

# create the config directory if not
if not os.path.exists(kafe_config_dir):
    logger.debug("Directory '%s' not found." % kafe_config_dir)
    os.makedirs(kafe_config_dir)
    logger.debug("Successfully created directory '%s'." % kafe_config_dir)

kafe_config_file = os.path.join(kafe_config_dir, 'kafe.conf')

if not os.path.isfile(kafe_config_file):
    logger.debug("Config file '%s' not found." % kafe_config_file)
    # copy the default config file to the user's conf directory
    import shutil
    kafe_default_config_file = \
        os.path.join(kafe.__path__[0], 'config', 'kafe.default.conf')
    shutil.copyfile(kafe_default_config_file, kafe_config_file)
    logger.debug("Successfully copied default config from '%s' to '%s'." % (kafe_default_config_file, kafe_config_file))

cp = ConfigParser.ConfigParser()
cp.read(kafe_config_file)

# Check for kafe.conf in local directory
if os.path.isfile("kafe.conf"):
    logger.info("Local 'kafe.conf' found in '%s'." % (os.getcwd()))
    cp.read("kafe.conf")

# Import "constants" from the configuration

G_PADDING_FACTOR_X = cp.getfloat('Plot', 'padding_factor_x')
G_PADDING_FACTOR_Y = cp.getfloat('Plot', 'padding_factor_y')
G_PLOT_POINTS = cp.getint('Plot', 'plot_points')

M_TOLERANCE = cp.getfloat('Minuit', 'tolerance')
M_MAX_ITERATIONS = cp.getint('Minuit', 'max_iterations')
M_MAX_X_FIT_ITERATIONS = cp.getint('Minuit', 'max_x_fit_iterations')

F_SIGNIFICANCE_LEVEL = cp.getfloat('Fit', 'hyptest_significance')

FORMAT_ERROR_SIGNIFICANT_PLACES = cp.getint('Formatting', 'significant_error_places')

D_DEBUG_MODE = cp.getboolean('Debug', 'debug_mode')

G_FIT_INFOBOX_TITLE = cp.get('Plot', 'fit_infobox_title')
