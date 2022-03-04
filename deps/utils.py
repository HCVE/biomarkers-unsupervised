import json
from functools import singledispatch
from typing import List, Any, Dict

# noinspection PyUnresolvedReferences
import swifter

from deps.arguments import get_params


def get_class_attributes(cls: Any) -> List[str]:
    # noinspection PyTypeChecker
    return list(
        (k for k in cls.__annotations__.keys() if not k.startswith('__')))


def get_tabulate_format():
    tablefmt = 'fancy_grid'

    return dict(tablefmt=tablefmt, floatfmt=".3f")


def load_dictionary():
    with open("%s/output/dictionary.json" % get_params('data_folder'),
              "r",) as f:
        return json.load(f)


def get_feature_category(feature_name):
    return get_feature_category_from_dictionary(
        feature_name,
        load_dictionary(),
    )


def get_feature_category_from_dictionary(feature_name, dictionary):
    return dictionary[feature_name]["category"]


def assert_equals(variable1: Any, variable2: Any) -> None:
    if variable1 == variable2:
        return
    else:
        raise AssertionError(
            f'Does not equal\nLeft side: {str(variable1)}\nRight side: {str(variable2)}'
        )


def get_object_attributes(something: object) -> List[str]:
    return [
        key for key in something.__dict__.keys() if not key.startswith("_")
    ]


@singledispatch
def object2dict(obj) -> Dict:
    if hasattr(obj, '__dict__'):
        return object2dict(obj.__dict__)
    else:
        return obj
