"""
Custom Exception Handling for Hoki - based off astropy.utils.exceptions (thanks!!)
"""


# Basics


class HokiWarning(Warning):
    """
    The base warning class from which all Hoki warnings should inherit.
    """


class HokiUserWarning(UserWarning, HokiWarning):
    """
    One of the most common warnings
    """


class HokiDeprecationWarning(HokiWarning):
    """
    A warning class to indicate a deprecated feature.
    """


# Formatting


class HokiFormatWarning(UserWarning, HokiWarning):
    """
    A very important warning name since Hoki relies on specific formats
    """


class HokiFormatError(Exception):
    """
    A very important error message since Hoki relies on specific formats
    """


class HokiFatalError(Exception):
    """
    An error raised when user does something that they shouldn't and hoki will stop
    """

class HokiKeyError(KeyError):
    """
    Hoki Key error
    """

class HokiTypeError(KeyError):
    """
    Hoki Type error
    """

