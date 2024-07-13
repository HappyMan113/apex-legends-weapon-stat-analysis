import operator
from typing import Any, Dict, Sized, Tuple, Type, TypeVar

T = TypeVar('T')


def check_type(allowed_types: type | tuple[type, ...], optional: bool = False, **values: Any):
    for allowed_type in (
            (allowed_types,)
            if not isinstance(allowed_types, tuple)
            else allowed_types):
        if allowed_type is type(None):
            raise ValueError('allowed_types cannot contain type(None). Use optional flag instead.')

    for name, value in values.items():
        if not (optional and value is None) and not isinstance(value, allowed_types):
            allowed_types_name = (allowed_types.__name__
                                  if not isinstance(allowed_types, tuple)
                                  else tuple(map(lambda t: t.__name__, allowed_types)))
            raise TypeError(f'{name} ({value}) must be an instance of {allowed_types_name}')


def check_float(optional: bool = False,
                min_value: float | None = None,
                min_is_exclusive: bool = False,
                **values: float):
    check_type((float, int), optional=optional, **values)
    for name, value in values.items():
        if value is None:
            continue

        if (min_value is not None and
                (operator.le if min_is_exclusive else operator.lt)(value, min_value)):
            must_be = 'greater than' if min_is_exclusive else 'at least'
            raise ValueError(f'{name} ({value}) must be {must_be} {min_value}.')


def check_int(optional: bool = False, min_value: int | None = None, **values: int):
    check_type(int, optional=optional, **values)
    for name, value in values.items():
        if value is None:
            continue

        if min_value is not None and value < min_value:
            raise ValueError(f'{name} ({value}) must be at least {min_value}.')


def check_bool(**values: bool):
    check_type(bool, **values)


def check_str(allow_blank: bool = True, optional: bool = False, **values: str):
    check_type(str, optional=optional, **values)
    for name, value in values.items():
        if value is None:
            continue

        if not allow_blank and len(value) == 0:
            raise ValueError(f'{name} cannot be blank.')


def to_kwargs[T](**tuples: Tuple['T', ...]) -> Dict[str, T]:
    return {f'{name}[{idx}]': value
            for name, tup in tuples.items()
            for idx, value in enumerate(tup)}


def check_equal_length(**sized_objects: Sized):
    if len(sized_objects) < 2:
        raise ValueError('Must specify at least two sized objects to compare.')
    iterable = iter(sized_objects.items())
    first_name, first_value = next(iterable)
    for name, value in sized_objects.items():
        if len(first_value) != len(value):
            raise ValueError(f'len({name}) ({len(value)}) did not equal len({first_name}) ('
                             f'{len(first_value)})')


def check_tuple(allowed_element_types: Type | Tuple[Type, ...],
                allow_empty: bool = True,
                **single_or_tuple: Tuple):
    check_type(tuple, **single_or_tuple)
    check_type(allowed_element_types, **to_kwargs(**single_or_tuple))
    for name, tup in single_or_tuple.items():
        if not allow_empty and len(tup) == 0:
            raise ValueError(f'{name} cannot be empty.')
