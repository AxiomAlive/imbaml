import inspect
import logging
import pprint
import traceback
from typing import Callable

logger = logging.getLogger(__name__)


class Decorators:
    @staticmethod
    def log_exception(f: Callable) -> Callable:
        def _log_exception(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as exc:
                logger.error(pprint.pformat(traceback.format_exception(type(exc), exc, exc.__traceback__   )))
                return None
        return _log_exception

    @staticmethod
    def remove_unnecessary_hp(hp: str, parent_level: int = 1):
        def _decorate(f) -> Callable:
            def _remove_unnecessary_hp(*args, **kwargs):
                for arg in args:
                    if inspect.isclass(arg):
                        if getattr(arg, hp, None) is not None:
                            delattr(arg.__mro__[parent_level], hp)
                        return f(*args, **kwargs)
            return _remove_unnecessary_hp
        return _decorate
