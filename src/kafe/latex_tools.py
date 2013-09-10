'''
.. module:: latex_tools
    :platform: Unix
    :synopsis: This submodule contains several useful tools for handling
        :math:`LaTeX` expressions.

.. moduleauthor:: Daniel Savoiu <danielsavoiu@gmail.com>

'''

L_ESCAPE_FOR_MATH_MODE = {'^': "\\^{}",
                          '_': "\\_{}"}


def ascii_to_latex_math(str_ascii):
    r"""
    Escapes certain characters in an ASCII input string so that the result
    can be included in math mode without error.
    """
    result = str_ascii

    result = result.replace("{", "(")  # transform braces
    result = result.replace("}", ")")  # transform braces

    result = result.replace("\\", "")  # remove slashes

    for from_ascii, to_latex in L_ESCAPE_FOR_MATH_MODE.iteritems():
        result = result.replace(from_ascii, to_latex)

    result = "\\texttt{%s}" % (result,)

    return result
