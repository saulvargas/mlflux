import functools
import inspect

import mlflow


def log_input_params(f):
    @functools.wraps(f)
    def wrapper(**kwargs):
        default_params = {
            k: v.default for k, v in inspect.signature(f).parameters.items()
        }
        params = {**default_params, **kwargs}
        mlflow.log_params(params)

        return f(**kwargs)

    return wrapper
