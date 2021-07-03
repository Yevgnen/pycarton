# -*- coding: utf-8 -*-

import datetime
import numbers
from typing import NewType, Optional, Union

import dateutil
import dateutil.parser

DateTime = NewType("DateTime", Union[str, numbers.Real, datetime.datetime])


def ensure_datetime(dt: DateTime) -> datetime.datetime:
    if isinstance(dt, datetime.datetime):
        return dt

    if isinstance(dt, str):
        return dateutil.parser.parse(dt)

    if isinstance(dt, numbers.Real):
        return datetime.datetime.fromtimestamp(dt)

    raise ValueError(f"Invalid datetime type: {type(dt)}")


def utcstr_to_datetime(
    string: str, timezone: Optional[Union[datetime.tzinfo, str]] = dateutil.tz.tzlocal()
) -> datetime.datetime:
    if isinstance(timezone, str):
        timezone = dateutil.tz.gettz(timezone)

    return (
        dateutil.parser.parse(string)
        .replace(tzinfo=dateutil.tz.tzutc())
        .astimezone(timezone)
        .replace(tzinfo=None)
    )


def timestamp13_to_datetime(timestamp: float) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(timestamp / 1000)


def timestamp10_to_datetime(timestamp: float) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(timestamp)


def timestamp_to_datetime(timestamp: float) -> datetime.datetime:
    return timestamp10_to_datetime(timestamp)


def date(
    offset: int = 0, return_date: bool = False
) -> Union[datetime.datetime, datetime.date]:
    _date = datetime.datetime.now() + datetime.timedelta(days=offset)
    if return_date:
        return _date.date()

    return datetime.datetime.fromordinal(_date.toordinal())


def today(return_date: bool = False) -> Union[datetime.datetime, datetime.date]:
    return date(offset=0, return_date=return_date)


def yesterday(return_date: bool = False) -> Union[datetime.datetime, datetime.date]:
    return date(offset=-1, return_date=return_date)


def tomorrow(return_date: bool = False) -> Union[datetime.datetime, datetime.date]:
    return date(offset=1, return_date=return_date)
