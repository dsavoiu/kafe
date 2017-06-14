
# import main logger for kafe
import logging
import kafe
import copy
logger = logging.getLogger('kafe')

class Multiplot(object):
    '''
    Object starting a multiplot. Takes a Multifit object and plots the fit objects stored in multifit.

    Parameters
    ----------
    **Multifit**: Multifit.py object
        The Multifit.py object which will be plotted. Only one Multifit can be given.
    '''
    def __init__(self,Multifit):
        self.fits = []
        self.subplots = []
        for fit in Multifit.fit_list:
            self.fits.append(fit)
            self.subplots.append(kafe.Plot(fit))

    def plot_all(self, show_info=True, show_band_for='meaningful'):
        '''
        Plots all fits stored in the Multifit.py object.
        '''
        if show_info:
            show_info_for='all'
        else:
            show_info_for = None

        for plot in self.subplots:
            plot.plot_all(show_info_for=show_info_for, show_data_for='all',
                          show_function_for='all', show_band_for=show_band_for)

    def plot(self,id,  show_info=True, show_band_for='meaningful'):
        '''
         Plot the `Fit` object with the number `p_id` from Multifit to its figure.
        '''
        if show_info:
            show_info_for='all'
        else:
            show_info_for = None

        self.subplots[id].plot_all(show_info_for=show_info_for, show_data_for='all',
                                    show_function_for='all', show_band_for=show_band_for)

    def save(self, id, output_file):
        self.subplots[id].save(output_file)

    def save_all(self, *output_files):
        if len(output_files) != len(self.subplots):
            raise ValueError("Multiplot has {} subplots: need {} output files but "
                             "got {}!".format(len(self.subplots), len(self.subplots), len(output_files)))

        for i, filename in enumerate(output_files):
            self.save(i, filename)
