import logging
import pprint
import traceback
from typing import Callable


logger = logging.getLogger(__name__)


class ExceptionWrapper:
    @staticmethod
    def log_exception(f: Callable) -> Callable:
        def _log_exception(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as exc:
                logger.error(pprint.pformat(traceback.format_exception(type(exc), exc, exc.__traceback__   )))
                return None
        return _log_exception