'''
Created on Aug 28, 2017

@author: optas
'''

import functools
import types


def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


def bind_new_method_to_instance(new_method, instance):
    '''
    To an existing instance of a class`instance` add a bound method  `new_method`.
    '''
    instance.__setattr__(new_method.__name__, types.MethodType(new_method, instance))
