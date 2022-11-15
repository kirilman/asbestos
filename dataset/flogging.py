import logging, sys
from typing import Any, Callable, TypeVar, cast

FuncType = Callable[..., Any]
F = TypeVar('F', bound = FuncType)

def debug_load_img(function :F)->F:
    def logged_function(*arg, **kw):
        logging.debug("%s(%r,%r)", function.__name__, arg, kw)
        result = function(*arg, **kw)
        logging.debug("result of %s = %r, shape = %s", function.__name__, result, result.shape)
        return result
    return cast(F, logged_function)