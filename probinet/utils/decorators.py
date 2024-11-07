"""
This module contains utility decorators.
"""


def inherit_docstring(cls, from_init=True):
    """
    Decorator to inherit the docstring from the parent class.

    Parameters
    ----------
    cls : class
        The parent class from which to inherit the docstring.
    from_init : bool, optional
        If True, inherit the docstring from the __init__ method of the parent class.
        If False, inherit the class-level docstring from the parent class.
    """

    # Based on the implementation from the scikit-tensor library:
    # https://github.com/mnick/scikit-tensor/blob/master/sktensor/pyutils.py

    def decorator(func):
        if from_init:
            # Inherit the docstring from the __init__ method
            parent_init = getattr(cls, "__init__", None)
            if parent_init and parent_init.__doc__:
                func.__doc__ = parent_init.__doc__
        else:
            # Inherit the class-level docstring
            if class_docstring := cls.__doc__:
                func.__doc__ = class_docstring

        return func

    return decorator
