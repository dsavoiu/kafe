'''
.. module:: plot
   :platform: Unix
   :synopsis: A submodule for plotting `Dataset`s with ``matplotlib``.
.. moduleauthor:: Daniel Savoiu <danielsavoiu@gmail.com>
.. moduleauthor:: Guenter Quast <G.Quast@kit.edu>

'''
# -----------------------------------------------------------------
# Changes:
# GQ 140304: inserted argument "borderpad" replacing "pad" in legend
# GQ 140731: white spaces as beginning of line cause trouble in
#            ver. 1.3.1 - reformatted parameter box
# GQ 140807: parameter errors are 0. for fixed parameters; this
#             needs a fix when determining `derivative_spacing`
#             in `plot()`; now depends on individual errors
# DS 141001: stylistic and esthetic changes
#               - font choice now defaults to Palatino (serif)
#                 and Helvetica (sans-serif)
#               - axis labels moved to end of axis
#               - fit info box now adjusts in size to be the same
#                 width as the legend box
# GQ 160116    two fixes in self.fitinfotext (compatibility with matplotlib 1.5)
# GQ 160319    new kw argument "plotstyle" in Plot() allows using
#               user-defined instance of PlotStyle()
# GQ 160516   derive figure name from fit_label of fits[0]
# -----------------------------------------------------------------

import numpy as np


import matplotlib as mpl

import matplotlib.pyplot as plt
from .function_tools import outer_product
from .numeric_tools import extract_statistical_errors

from .config import (G_PADDING_FACTOR_X, G_PADDING_FACTOR_Y,
                     G_PLOT_POINTS, G_FIT_INFOBOX_TITLE)
import re  # regular expressions
from string import split, join, lower, replace

# import main logger for kafe
import logging
logger = logging.getLogger('kafe')


def label_to_latex(label):
    '''
    Generates a simple LaTeX-formatted label from a plain-text label.
    This treats isolated characters and words beginning with a backslash
    as mathematical expressions and surround them with $ signs accordingly.
        
    Parameters
    ----------

    **label** : string
        Plain-text string to convert to LaTeX.
    '''

    tokens = split(label)
    for token_id, token in enumerate(tokens):
        if len(token) == 1 or token[0] == '\\':
            if lower(token[-1]) not in "abcdefghijklmnopqrstuvwxyz":
                # surround isolated chars with $ (omit last)
                tokens[token_id] = '$%s$%s' % (token[:-1], token[-1])
            else:
                # surround isolated chars with $
                tokens[token_id] = '$%s$' % (token,)
    return join(tokens)


def pad_span_log(span, pad_coeff=1, additional_pad=None, base=10):

    try:
        len(span)
    except:
        raise TypeError("Span passed to pad_span must be an iterable type"
                        "(got %s)" % type(span))

    if len(span) != 2:
        raise IndexError("Span passed to pad_span must be an iterable"
                         "of length 2 (got %d)" % (len(span),))

    if (np.asarray(span)<0).any():
        raise ValueError("Span passed to pad_span_log must be "
                         "on the real positive axis (got %r)" % (span,))

    _logspan = np.log(span) / np.log(base)
    _logspan2 = np.asarray(pad_span(_logspan, pad_coeff, additional_pad))
    _span = np.exp(_logspan2 * np.log(base))
    return list(_span)

def pad_span(span, pad_coeff=1, additional_pad=None):
    '''
    Enlarges the interval `span` (list of two floats) symmetrically around
    its center to length `pad_coeff`. Optionally, an `additional_pad` argument
    can be specified. The returned span is then additionally enlarged by that
    amount.

    `additional_pad` can also be a list of two floats which specifies an
    asymmetric amount by which to enlarge the span. Note that in this case,
    positive entries in `additional_pad` will enlarge the span (move the
    interval end away from the interval center) and negative amounts will
    shorten it (move the interval end towards the interval center).
    '''

    try:
        len(span)
    except:
        raise TypeError("Span passed to pad_span must be an iterable type"
                        "(got %s)" % type(span))

    if len(span) != 2:
        raise IndexError("Span passed to pad_span must be an iterable"
                         "of length 2 (got %d)" % (len(span),))

    if additional_pad is None:
        additional_pad = [0, 0]
    else:
        try:
            # check if the additional_pad argument value is iterable
            # (list or array)
            # by trying to get its iterator.
            iter(additional_pad)
        except TypeError:
            # if this fails, then this object is not iterable
            # turn into list of identical elements
            additional_pad = [additional_pad, additional_pad]
        else:
            if len(additional_pad) != 2:
                raise IndexError(
                    "Additional pad passed to pad_span "
                    "is not an iterable of length 2 (got %d)"
                    % (len(additional_pad),)
                )

    # store the interval width in an explicit variable
    width = span[1] - span[0]

    if width == 0:
        return list(span)  # Zero - width intervals cannot be padded

    return list((
        0.5*(span[0] + span[1] - width*pad_coeff) - additional_pad[0],
        0.5*(span[0] + span[1] + width*pad_coeff) + additional_pad[1]
    ))


class PlotStyle:
    '''
    Class for specifying a style for a specific plot. This object stores a
    progression of marker and line types and colors, as well as preferences
    relating to point size and label size. These can be overriden by
    overwriting the instance variables directly. A series of `get_...` methods
    are provided which go through these lists cyclically.
    '''

    def __init__(self):
        '''
        Construct a default plot style.
        '''

        # Define a progression of matplotlib
        # marker and line styles, as well as colors
        self.markers = ['o', '^', 's', 'D', 'v', 'h', '*', '+', 'x']
        self.lines = ['-', '--', '-.', ':']
        self.markercolors = ['r', 'b', 'g', 'c', 'm', 'k']
        self.linecolors = ['r', 'b', 'g', 'c', 'm', 'k']

        # Define a progression of point sizes
        self.pointsizes = [7]

        # Set other style properties
        self.labelsize = 12     # make labels large
        self.grid = True        # use a grid per default
        self.usetex = True      # tell matplotlib to use TeX

        # Some default styles and offsets
        self.axis_label_styles = ('italic', 'italic')
        self.axis_label_align = ('right', 'right')
        #self.axis_label_coords = ((.5, -.1), (-.12, .5))
        self.axis_label_coords = ((1.02, -.1), (-.12, 1.02))
        self.axis_label_pad = (7, 7)

        # Set legend parameters
        self.legendparams_kw = {
            'ncol': 1,
            #'fancybox': True,
            'shadow': False,
            'numpoints': 1,
            'bbox_to_anchor': (1.05, 1.),
            'borderaxespad': 0.,
            #'borderpad': 0.05
        }

        # Default keyword arguments to pass to rc('font',...)
        self.rcfont_kw = {
            'family': 'sans-serif',
            'serif':  ['Palatino', 'cm', 'CMU Classical Serif'],
            'sans-serif':  ['Helvetica', 'CMU Bright'],
            'monospace':  ['Monospace', 'CMU Typewriter Text']
        }
        self.rcparams_kw = {
            'axes.labelsize': 20,
            'font.size': 14,
            'legend.fontsize': 18,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'text.usetex': True,
            #'text.latex.preamble': [r"\usepackage{sansmath}"],
            'axes.unicode_minus': True,
            #'legend.loc': 'best',
            'legend.loc': 'upper left',
            'figure.figsize': (12, 6)
        }

    def get_marker(self, idm):
        '''Get a specific marker type. This runs cyclically through the defined
        defaults.'''
        return self.markers[idm % len(self.markers)]

    def get_line(self, idm):
        '''Get a specific line type. This runs cyclically through the defined
        defaults.'''
        return self.lines[idm % len(self.lines)]

    def get_markercolor(self, idm):
        '''Get a specific marker color. This runs cyclically through the
        defined defaults.'''
        return self.markercolors[idm % len(self.markercolors)]

    def get_linecolor(self, idm):
        '''Get a specific line color. This runs cyclically through the defined
        defaults.'''
        return self.linecolors[idm % len(self.linecolors)]

    def get_pointsize(self, idm):
        '''Get a specific point size. This runs cyclically through the defined
        defaults.'''
        return self.pointsizes[idm % len(self.pointsizes)]


class Plot(object):
    '''
    The constuctor accepts a series of `Fit` objects as positional
    arguments. Some keyword arguments can be provided to override
    the defaults.
    '''

    def __init__(self, *fits, **kwargs):
        # store the child Fit objects in an instance variable
        #: list of :py:obj:`Fit` objects to plot
        self.fits = list(fits)

        # set the default style as the current plot style
        #: plot style
        self.plot_style = kwargs.get('plotstyle', PlotStyle())

        # Update matplotlib's rcParams with those from plot_style
        self._update_rcParams()

        self.plot_range = {'x': None, 'y': None}    #: plot range

        #: whether to show the plot legend (``True``) or not (``False``)
        self.show_legend = kwargs.get('show_legend', True)

        if len(fits) == 1:
            # inherit axis labels from Fit's Dataset
            #: axis labels
            self.axis_labels = map(label_to_latex,
                                   self.fits[0].dataset.axis_labels)

            # set unit in brackets (if available)
            for label_id, _ in enumerate(self.axis_labels):
                unit = self.fits[0].dataset.axis_units[label_id]
                if unit:
                    self.axis_labels[label_id] += " [\\textrm{%s}]" % (unit,)
        elif len(fits) > 1:
            logger.warning("More than one Fit in the same Plot "
                           "detected. In this case, axis labels default to "
                           "`x` and `y` and must be set explicitly if "
                           "something different is desired.")
            self.axis_labels = ('$x$', '$y$')  # set default axis names
        else:
            # Plot with no Fits -> just set axis names
            self.axis_labels = ('$x$', '$y$')  # set default axis names        
            
        self.init_plots(**kwargs)              # initialize the plots

    def _update_rcParams(self):
        '''
        Set matplotlib plotting parameters to those from self.plot_style
        '''
        # Update matplotlib's rcParams with those from plot_style
        mpl.rcParams.update(self.plot_style.rcparams_kw)
        mpl.rc('font', **self.plot_style.rcfont_kw)

    def init_plots(self, **kwargs):
        '''
        Initialize the plots for each fit.
        '''

        self.figure = plt.figure(self.fits[0].fit_name)
        """A matplotlib figure object."""
        self.axes = self.figure.add_subplot(121)
        """A matplotlib axes object. Use this to modify the `axes` object
           from matplotlib directly."""
        #self.axes = self.figure.add_axes([.07, .15, .6, .8])

        self.figure.subplots_adjust(
            wspace=.2,  # default .2
            hspace=.2,  # default .2
            top=.92,  # default .9
            bottom=.12,  # default .1
            left=.11,  # default .125
            right=.8  # default .9
        )

        box = self.axes.get_position()
        self.axes.set_position([box.x0, box.y0, box.width * 1.5, box.height])

        # set axis scales for both axes
        self.axis_scales = ['linear', 'linear']
        self.axis_scale_logbases = [10, 10]
        for _axis in ('x', 'y'):
            _scale_spec = _axis + 'scale'
            _scale_base_spec = _axis + 'scalebase'

            _scale = kwargs.get(_scale_spec, 'linear')
            _scale_base = kwargs.get(_scale_base_spec, 10)

            scale_kwargs = {'base'+_axis: _scale_base}
            self.set_axis_scale(_axis, _scale, **scale_kwargs)

        self.compute_plot_range()

        # attach the custom callback function to the draw event
        self.figure.canvas.mpl_connect('draw_event', self.on_draw)

    def set_axis_scale(self, axis, scale_type, **kwargs):
        '''
        Set the scale for an axis.
        
        Parameters
        ----------

        **axis** : ''x'' or ''y''
            Axis for which to set the scale.

        **scale_type** : ''linear'' or ''log''
            Type of scale to set for the axis.

        Keyword Arguments
        -----------------

        **basex** : int
            Base of the ''x'' axis scale logarithm. Only relevant for log
            scales.

        **basey** : int
            Base of the ''y'' axis scale logarithm. Only relevant for log
            scales.
        '''

        if scale_type not in ('linear', 'log'):
            raise ValueError("Unknown scale `%s'. Use 'linear' or 'log'." % (scale_type,))
        if axis == 'x':
            self.axes.set_xscale(scale_type, **kwargs)
            if scale_type == 'log':
                _base = kwargs.get('basex', 10)
                self.axis_scales[0] = 'log'
                self.axis_scale_logbases[0] = _base
            elif scale_type == 'linear':
                self.axis_scales[0] = 'linear'
                self.axis_scale_logbases[1] = 10    # irrelevant -> set to 10
        elif axis == 'y':
            self.axes.set_yscale(scale_type, **kwargs)
            if scale_type == 'log':
                _base = kwargs.get('basey', 10)
                self.axis_scales[1] = 'log'
                self.axis_scale_logbases[1] = _base
            elif scale_type == 'linear':
                self.axis_scales[1] = 'linear'
                self.axis_scale_logbases[1] = 10    # irrelevant -> set to 10
        else:
            raise SyntaxError("Unknown axis `%s'. Use 'x' and 'y'." % (axis,))

    def plot_all(self, show_info_for='all', show_data_for='all',
                 show_function_for='all', show_band_for='meaningful'):
        '''
        Plot every `Fit` object to its figure.
        '''

        # Update matplotlib's rcParams with those from plot_style
        self._update_rcParams()

        _show_data_flags = []
        _show_function_flags = []
        for p_id, _ in enumerate(self.fits):
            if show_data_for != 'all':
                try:
                    iter(show_data_for)
                except:
                    show_data_for = (show_data_for,)  # wrap value in tuple

                if p_id in show_data_for:
                    _show_data_flags.append(True)
                    #self.plot(p_id, show_data=True)
                else:
                    _show_data_flags.append(False)
                    #self.plot(p_id, show_data=False)
            else:
                _show_data_flags.append(True)
                #self.plot(p_id, show_data=True)

            if show_function_for != 'all':
                try:
                    iter(show_function_for)
                except:
                    # wrap value in tuple
                    show_function_for = (show_function_for,)

                if p_id in show_function_for:
                    _show_function_flags.append(True)
                else:
                    _show_function_flags.append(False)
            else:
                    _show_function_flags.append(True)

            if show_band_for not in ('all', 'meaningful'):
                try:
                    iter(show_band_for)
                except:
                    # wrap value in tuple
                    show_band_for = (show_band_for,)

                if p_id in show_band_for:
                    self.plot(p_id, show_data=_show_data_flags[p_id],
                      show_function=_show_function_flags[p_id], show_band=True)
                else:
                    self.plot(p_id, show_data=_show_data_flags[p_id],
                      show_function=_show_function_flags[p_id], show_band=False)
            else:
                _draw_band = True
                if show_band_for == 'meaningful':
                    # only draw error band for datasets with error models
                    # since otherwise it is meaningless
                    _draw_band = self.fits[p_id].dataset.has_errors()

                self.plot(p_id, show_data=_show_data_flags[p_id],
                  show_function=_show_function_flags[p_id], show_band=_draw_band)

        if self.show_legend:
            self.draw_legend()

        self.draw_fit_parameters_box(show_info_for)

    def on_draw(self, event):
        '''
        Function to call when a draw event occurs.
        '''
        fig = event.canvas.figure
        if not fig == self.figure:
            return
        else:
            # update rcParams to match plot style
            self._update_rcParams()

            if hasattr(self, 'fitinfobox'):
                # disable callbacks to avoid recursion when redrawing
                tmp_callbacks = fig.canvas.callbacks.callbacks[event.name]
                fig.canvas.callbacks.callbacks[event.name] = {}

                # get the legend bounding box and extract some dimension info
                legend_bbox = self.legend.legendPatch.get_bbox().inverse_transformed(
                    self.axes.transAxes
                )

                viewport_limits = self.axes.transAxes.inverted().transform(
                        self.figure.transFigure.transform((1, 1))
                    )

                fig.canvas.draw()
                #print fig.subplotpars.__dict__
                self.figure.subplots_adjust(right=1.1)
                self.fitinfobox.set_width(legend_bbox.width)

                if self.fitinfobox.get_height() > 1. - legend_bbox.height - 0.05:
                    self.fitinfobox.set_height(1 - legend_bbox.height - 0.05)

                # redraw
                fig.canvas.draw()

                # reenable callbacks
                fig.canvas.callbacks.callbacks[event.name] = tmp_callbacks

    def draw_legend(self):
        '''Draw the plot legend to the canvas'''
        # show plot legend
        self.legend = self.axes.legend(**self.plot_style.legendparams_kw)
        self.legend.draggable()

    def draw_fit_parameters_box(self, plot_spec=0,
                                force_show_uncertainties=False):
        '''Draw the parameter box to the canvas
        
        Parameters
        ----------

        *plot_spec* : int, list of ints, string or None (optional, default: 0)
            Specify the plot id of the plot for which to draw the parameters.
            Passing 0 will only draw the parameter box for the first plot, and
            so on. Passing a list of ints will only draw the parameters for
            plot ids inside the list. Passing ``'all'`` will print parameters
            for all plots. Passing ``None`` will return immediately doing
            nothing.

        *force_show_uncertainties* : boolean (optional, default: False)
            If ``True``, shows uncertainties even for Datasets without error
            data. Note that in that case these values are meaningless!
        '''
        if plot_spec is None:
            return

        self.figure.canvas.draw()

        # get the offset to subtract for each axes transformation
        offset = self.axes.transAxes.transform((0, 0))

        # get the legend bounding box and extract some dimension info
        legend_bbox = self.legend.legendPatch.get_bbox().inverse_transformed(
            self.axes.transAxes
        )

        legend_size = (legend_bbox.width, legend_bbox.height)

        # fit parameters box size
        textbox_size = (legend_size[0], 1. - legend_size[1] - 0.05)
        textbox_size_px = self.axes.transAxes.transform(textbox_size) - offset

        # pad text in fit parameters box by this amount
        pad_amount = .05  #self.plot_style.legendparams_kw['pad']
        pad_amount_px = (self.axes.transAxes.transform(
            (pad_amount, 0)
        ) - offset)[0]

        text_content = "\\textbf{%s}\n" % (G_FIT_INFOBOX_TITLE,)

        try:
            fit = self.fits[plot_spec]  # try to find the specified plot
        except TypeError:
            if plot_spec == "all":   # if not found, check for keyword 'all'
                # go through all fits
                fits_with_parameter_box = self.fits
            else:
                try:
                    fits_with_parameter_box = []
                    for p_id in plot_spec:
                        fits_with_parameter_box.append(self.fits[p_id])
                except:
                    raise Exception("Cannot parse plot specification %r."
                                    % (plot_spec,))
        else:
            fits_with_parameter_box = [fit]

        for fit in fits_with_parameter_box:
            if fit.fit_label is not None:
                text_content += "~\n%s" % fit.fit_label
                text_content += "~\n%s:\n" \
                    % fit.fit_function.get_function_equation('latex', 'full')
            else:
                # if no expression is provided for the function,
                # just display the name
                if fit.fit_function.latex_expression is not None:
                    text_content += "~\n%s:\n" \
                        % fit.fit_function.get_function_equation('latex',
                                                                 'full')
                else:
                    text_content += "~\n%s:\n" \
                        % fit.fit_function.get_function_equation('latex',
                                                                 'name')
            for idx, _ in enumerate(fit.parameter_names):
                parname = fit.latex_parameter_names[idx]
                parval = fit.get_parameter_values(rounding=True)[idx]
                if force_show_uncertainties or fit.dataset.has_errors():
                    if fit.parameter_is_fixed(idx):
                        # parameter is fixed -> show in infobox
                        text_content += ('$%s=%g$ (fixed)\n' % (parname, parval))
                    else:
                        parerrs = fit.get_parameter_errors(rounding=True)[idx]
                        text_content += ('$%s=%g\pm%g$\n' % (parname, parval, parerrs))
                else:
                    text_content += ('$%s=%g$\n' % (parname, parval))


        # replace scientific notation with power of ten
        text_content = re.sub(r'(-?\d*\.?\d+?)0*e\+?(-?[0-9]*[1-9]?)',
                              r'\1\\times10^{\2}', text_content)

        # BUG: matplotlib LaTeX cannot handle certain substrings:
        #      '\n~\n' = no text rendered
        #      '\n\n'  = RuntimeError
        # WORKAROUND: replace '\n~\n' with '\n|\n'
        # (the pipe '|' renders as a dash)

        text_content = replace(text_content, '\n~\n', '\n|\n')

        self.fitinfobox = self.axes.add_patch(
            mpl.patches.Rectangle((legend_bbox.xmin, 0.00),
                textbox_size[0], textbox_size[1], facecolor='w',
                transform=self.axes.transAxes, clip_on=False
            )
        )

        # TODO: find cause of MPL error when passing 'xy'/'width'
        self.fitinfotext = self.axes.text(
            legend_bbox.xmin+pad_amount/2, 0.00+pad_amount/2,
            text_content[:-1],
            transform=self.axes.transAxes,
            fontsize=self.plot_style.rcparams_kw['legend.fontsize'],
            verticalalignment='bottom',
            horizontalalignment='left',
            bbox={
                #'xy': (legend_bbox.xmin, 0.00),  # GQ: remove for compatibility with matplotlib 1.5.1
                'facecolor': (1., .9, .9, 0.),
                'edgecolor': (1., .9, .9, 0.),
                #'width': textbox_size_px[0],  # GQ: remove for compatibility with matplotlib 1.5.1
                #'height': textbox_size_px[1],
                'pad': pad_amount_px
            },
            clip_on=False
        )

        self.figure.canvas.draw()

    def compute_plot_range(self, include_error_bars=True):
        '''
        Compute the span of all child datasets and sets the plot range to that
        '''

        # default plot range None (undefined)
        self.plot_range = {'x': None, 'y': None}

        # initialize plot spans for axes
        for current_fit in self.fits:
            xspan = current_fit.dataset.get_data_span('x', include_error_bars)
            # choose pad function by x scale
            if self.axis_scales[0] == 'log':
                xspan = pad_span_log(xspan, G_PADDING_FACTOR_X,
                                     base=self.axis_scale_logbases[0])
            else:
                xspan = pad_span(xspan, G_PADDING_FACTOR_X)
            self.extend_span('x', xspan)

            yspan = current_fit.dataset.get_data_span('y', include_error_bars)
            # choose pad function by y scale
            if self.axis_scales[1] == 'log':
                yspan = pad_span_log(yspan, G_PADDING_FACTOR_Y,
                                     base=self.axis_scale_logbases[1])
            else:
                yspan = pad_span(yspan, G_PADDING_FACTOR_Y)
            self.extend_span('y', yspan)

    def extend_span(self, axis, new_span):
        '''
        Expand the span of the current plot.

        This method extends the current plot span to include `new_span`
        '''

        # make sure the axis specification is valid
        if axis not in ('x', 'y'):
            raise SyntaxError("Unknown axis `%s'" % (axis,))

        # move the minimum down and maximum up, if necessary
        if self.plot_range[axis] is not None:
            self.plot_range[axis][0] = min(self.plot_range[axis][0],
                                           new_span[0])
            self.plot_range[axis][1] = max(self.plot_range[axis][1],
                                           new_span[1])
        else:
            # if plot range in None (undefined), take the new values directly
            self.plot_range[axis] = new_span

    def plot(self, p_id, show_data=True, show_function=True, show_band=True):
        '''
        Plot the `Fit` object with the number `p_id` to its figure.
        '''

        current_fit = self.fits[p_id]

        # set the current style
        # for the data points
        _pdata_kw = {
            'marker': self.plot_style.get_marker(p_id),
            'linestyle': 'None',
            'markerfacecolor': self.plot_style.get_markercolor(p_id),
            'markeredgecolor': self.plot_style.get_markercolor(p_id),
            'color': self.plot_style.get_markercolor(p_id),
            'label': current_fit.dataset.data_label,
            'ms': self.plot_style.get_pointsize(p_id),
            'capsize': 0,
            'zorder': 1
        }

        # for the fit function
        _fdata_kw = {
            'marker': 'None',
            'linestyle': self.plot_style.get_line(p_id),
            'color': self.plot_style.get_linecolor(p_id),
            'label': current_fit.fit_label,
            'zorder': 2
        }

        # take care of empty fit labels (replace with full function equation)
        if current_fit.fit_label is None:
            _fdata_kw['label'] = \
                current_fit.fit_function.get_function_equation('latex', 'full')

        # current error bar data defaults to None
        error_bar_data = {'x': None, 'y': None}

        # set current error bar data
        for axis in ('x', 'y'):
            # only set error bar data if errors are explicitly given,
            # either in form of a user-defined covariance matrix
            # or error lists
            if current_fit.dataset.has_errors(axis):
                # use the covmat's diagonal for the error bars
                error_bar_data[axis] = extract_statistical_errors(
                    current_fit.dataset.get_cov_mat(axis)
                )

        # compute the function data
        ############################

        # sample a fixed number of evenly-spaced points within the x span
        fxdata = np.linspace(self.plot_range['x'][0], self.plot_range['x'][1],
                             G_PLOT_POINTS)

        # apply the current fit function to every point in fxdata => fydata
        fydata = np.asarray(map(current_fit.get_current_fit_function(),
                                fxdata))

        # compute the confidence band around the function
        ##################################################

        # go through each data point and calculate the confidence interval
        if (show_band and current_fit.par_cov_mat is not None):
          confidence_band_data = current_fit.get_function_error(fxdata)
          # store upper and lower confidence band limits
          lower_cb = fydata - confidence_band_data
          upper_cb = fydata + confidence_band_data

        else:
           show_band = False

        # Do the actual plotting
        #########################

        # set some properties inherited from plot_style
        self.axes.grid(self.plot_style.grid)
        xlabel = self.axes.set_xlabel('\\bfseries '+self.axis_labels[0],
                                      style=self.plot_style.axis_label_styles[0],
                                      ha=self.plot_style.axis_label_align[0])

        ylabel = self.axes.set_ylabel('\\bfseries '+self.axis_labels[1],
                                      style=self.plot_style.axis_label_styles[1],
                                      ha=self.plot_style.axis_label_align[1])

        self.axes.xaxis.set_label_coords(*self.plot_style.axis_label_coords[0])
        self.axes.yaxis.set_label_coords(*self.plot_style.axis_label_coords[1])

        self.axes.tick_params(axis='x', pad=self.plot_style.axis_label_pad[0])
        self.axes.tick_params(axis='y', pad=self.plot_style.axis_label_pad[1])

        # plot data points, unless otherwise specified by the caller
        if show_data:
            pplot = self.axes.errorbar(current_fit.dataset.get_data('x'),
                                       current_fit.dataset.get_data('y'),
                                       yerr=error_bar_data['y'],
                                       xerr=error_bar_data['x'], **_pdata_kw)

        # plot fit function and confidence band
        if show_function:
            # shade confidence band
            if show_band:
                cband = self.axes.fill_between(fxdata, lower_cb, upper_cb,
                                              alpha='0.1',
                                              color=_fdata_kw['color'])

            # plot fit function
            fplot = self.axes.plot(fxdata, fydata, **_fdata_kw)

        # set the plot range
        self.axes.set_xlim(self.plot_range['x'])
        self.axes.set_ylim(self.plot_range['y'])

        # draw everything so positions are known
        self.figure.canvas.draw()

    def show(self):
        '''
        Show graphics in one or more matplotlib interactive windows.

        Note
        ----
            This shows all figures/plots generated before it is called. Because
            of the way ``matplotlib`` handles some plotting parameters
            (``matplotlib.rcParams``) these cannot be set individually for each
            figure before it is displayed. This means that all figures will be
            shown with the same plot style: that of the `Plot` object from
            which show() is called.
        '''
        self._update_rcParams()
        plt.show(self.axes)

    def save(self, output_file):
        '''
        Save the `Plot` to a file.
        '''

        self.figure.savefig(output_file)
