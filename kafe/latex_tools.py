'''
.. module:: latex_tools
    :platform: Unix
    :synopsis: This submodule contains several useful tools for handling
        :math:`LaTeX` expressions.

.. moduleauthor:: Daniel Savoiu <daniel.savoiu@cern.ch>

'''

L_ESCAPE_FOR_MATH_MODE = {'^': "\\^{}",
                          '_': "\\_{}"}


def ascii_to_latex_math(str_ascii, monospace=True, ensuremath=True):
    r"""
    Escapes certain characters in an ASCII input string so that the result
    can be included in math mode without error.

    **str_ascii** : string
        A plain-text string containing characters to be escaped for
        :math:`LaTeX` math mode.

    *monospace* : boolean (optional)
        Whether to render the whole expression as monospace. Defaults to
        ``True``.

    *ensuremath* : boolean (optional)
        If this is ``True``, the resulting formula is wrapped in
        an ``\ensuremath{}`` tag. Defaults to ``True``.
    """
    result = str_ascii

    result = result.replace("{", "(")  # transform braces
    result = result.replace("}", ")")  # transform braces

    result = result.replace("\\", "")  # remove slashes

    for from_ascii, to_latex in L_ESCAPE_FOR_MATH_MODE.items():
        result = result.replace(from_ascii, to_latex)

    if monospace:
        result = "\\texttt{%s}" % (result,)

    if ensuremath:
        result = "\\ensuremath{%s}" % (result,)

    return result
