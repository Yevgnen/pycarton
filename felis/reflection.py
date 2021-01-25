# -*- coding: utf-8 -*-

import copy
import importlib


def reflect(module_specs):
    modules = []
    for module_spec in module_specs:
        try:
            module = importlib.import_module(module_spec)
        except ModuleNotFoundError:
            module = None

        if module is not None:
            modules += [module]

    def _constructor(*args, **kwargs):
        kwargs = copy.deepcopy(kwargs)
        class_name = kwargs.pop("class")
        cls = None
        for module in modules:
            try:
                cls = getattr(module, class_name)
                break
            except AttributeError:
                pass

        if cls is None:
            raise ValueError(
                "class name `%s` not found in modules: [%r]"
                % (class_name, module_specs)
            )

        return cls(*args, **kwargs)

    return _constructor
