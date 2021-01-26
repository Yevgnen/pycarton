# -*- coding: utf-8 -*-

import copy
import json
from typing import Mapping, Optional, Type, TypeVar

import pytoml
import yaml

_T = TypeVar("_T", bound="Params")


class Params(dict):
    def copy(self) -> _T:
        return copy.deepcopy(self)

    def merge(self, default: Mapping) -> None:
        for key, value in copy.deepcopy(default).items():
            self.setdefault(key, value)

    def to_json_string(self, **kwargs) -> str:
        if not kwargs:
            kwargs = {"ensure_ascii": False}

        return json.dumps(self, **kwargs)

    def to_json(
        self, filename: str, file_kwargs: Optional[Mapping] = None, **kwargs
    ) -> None:
        if not file_kwargs:
            file_kwargs = {"mode": "w"}
        kwargs.setdefault("ensure_ascii", False)
        kwargs.setdefault("indent", 4)

        with open(filename, **file_kwargs) as f:
            json.dump(self, f, **kwargs)

    dumps = to_json_string
    dump = to_json

    def to_yaml(
        self, filename: str, file_kwargs: Optional[Mapping] = None, **kwargs
    ) -> None:
        if not file_kwargs:
            file_kwargs = {"mode": "w"}

        if not kwargs:
            kwargs = {"explicit_start": True, "indent": 2}

        with open(filename, **file_kwargs) as f:
            yaml.safe_dump(dict(self), f, **kwargs)

    def to_toml(
        self, filename: str, file_kwargs: Optional[Mapping] = None, **kwargs
    ) -> None:
        if not file_kwargs:
            file_kwargs = {"mode": "w"}

        with open(filename, **file_kwargs) as f:
            pytoml.dump(self, f, **kwargs)

    @classmethod
    def from_json_string(cls: Type[_T], json_string: str, **kwargs) -> _T:
        return cls(json.loads(json_string, **kwargs))

    @classmethod
    def from_json(
        cls: Type[_T], filename: str, file_kwargs: Optional[Mapping] = None, **kwargs
    ) -> _T:
        if not file_kwargs:
            file_kwargs = {"mode": "r"}

        with open(filename, **file_kwargs) as f:
            return cls(json.load(f, **kwargs))

    loads = from_json_string
    load = from_json

    @classmethod
    def from_toml(
        cls: Type[_T], filename: str, file_kwargs: Optional[Mapping] = None, **kwargs
    ) -> _T:
        if not file_kwargs:
            file_kwargs = {"mode": "r"}

        with open(filename, **file_kwargs) as f:
            return cls(pytoml.load(f, **kwargs))

    @classmethod
    def from_yaml(
        cls: Type[_T], filename: str, file_kwargs: Optional[Mapping] = None, **kwargs
    ) -> _T:
        if not file_kwargs:
            file_kwargs = {"mode": "r"}

        with open(filename, **file_kwargs) as f:
            return cls(yaml.safe_load(f, **kwargs))
