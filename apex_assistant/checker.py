import operator
from typing import Dict, Iterable, Mapping, Optional, Sized, Tuple, Type, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


def check_type(allowed_types: Type[T] | Tuple[Type, ...], optional: bool = False, **values: T):
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


def check_float_vec(min_value: float | None = None, **values: NDArray[np.float64]):
    for name, value in values.items():
        if not isinstance(value, np.ndarray):
            raise TypeError(f'{name} must be an NDArray.')

        if (not np.issubdtype(value.dtype, np.floating) and
                not np.issubdtype(value.dtype, np.integer)):
            raise TypeError(f'{name} had a dtype of {value.dtype} instead of np.floating or '
                            'np.integer')

        if value.ndim != 1:
            raise ValueError(f'{name} ({value}) must have one dimension.')

        if min_value is not None and value.min(initial=min_value) < min_value:
            raise ValueError(f'{name} ({value}) must be at least {min_value}.')


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


def to_kwargs(**tuples: Iterable[T]) -> Dict[str, T]:
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


def check_iterable(allowed_element_types: Type[T] | Tuple[Type[T], ...],
                   allowed_iterable_types: Type[T] | Tuple[Type[T], ...] = Iterable,
                   values_optional: bool = False,
                   allow_empty: bool = True,
                   **iterable: Iterable[T]):
    check_type(allowed_iterable_types, **iterable)
    check_type(allowed_element_types, optional=values_optional, **to_kwargs(**iterable))
    for name, tup in iterable.items():
        if not allow_empty and next(iter(tup), None) is None:
            raise ValueError(f'{name} cannot be empty.')


def check_tuple(allowed_element_types: Type[T] | Tuple[Type[T], ...],
                values_optional: bool = False,
                allow_empty: bool = True,
                **tup: Tuple[T, ...]):
    check_iterable(allowed_element_types=allowed_element_types,
                   allowed_iterable_types=tuple,
                   values_optional=values_optional,
                   allow_empty=allow_empty,
                   **tup)


def check_mapping(allowed_key_types: Optional[Union[Type[K], Tuple[Type[K], ...]]] = None,
                  keys_optional: bool = False,
                  allowed_value_types: Optional[Union[Type[V], Tuple[Type, ...]]] = None,
                  values_optional: bool = False,
                  allow_empty: bool = True,
                  **dictionaries: Mapping[K, V]):
    check_type(Mapping, **dictionaries)
    if allowed_key_types is not None:
        check_type(allowed_key_types,
                   optional=keys_optional,
                   **{f'Element {idx} in {dict_name}': elt_key
                      for dict_name, dictionary in dictionaries.items()
                      for idx, elt_key in enumerate(dictionary)})
    if allowed_value_types is not None:
        check_type(allowed_value_types,
                   optional=values_optional,
                   **{f'{dict_name}[{elt_key}]': elt_value
                      for dict_name, dictionary in dictionaries.items()
                      for elt_key, elt_value in dictionary.items()})
    for dict_name, dictionary in dictionaries.items():
        if not allow_empty and len(dictionary) == 0:
            raise ValueError(f'{dict_name} cannot be empty.')
