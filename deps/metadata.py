import json
import re
from functools import partial
from typing import Dict

from toolz import groupby, valmap

from hcve_lib.functional import statements, raise_exception, pipe, flatten, merge_by, map_tuples


def get_biomarkers_metadata():
    with open('./biomarkers_mapping.json', 'r') as f:
        input_data = json.load(f)

    return pipe(
        input_data,
        parse_data,
        group_by_feature,
    )


def parse_data(categories: Dict) -> Dict:
    return pipe(
        categories,
        partial(
            valmap,
            partial(
                map, lambda biomarker_str: statements(
                    match := re.match(r'(.*) \((.*)\) \((.*)\)', biomarker_str
                                      ),
                    {
                        'name':
                        match.group(1),
                        'feature':
                        (match.group(2).strip().replace('IL-18', 'IL18').
                         replace('MMP-12', 'MMP12').replace('MMP-7', 'MMP7').
                         replace('FGF21', 'FGF-21').replace('-', '_').replace(
                             ' ', '_').upper()),
                        'feature_olink':
                        match.group(2),
                        'code':
                        match.group(3),
                    } if match else raise_exception(Exception('Bad format')),
                ))))


def group_by_feature(categories: Dict) -> Dict:
    return pipe(
        categories.items(),
        partial(
            map_tuples, lambda category, biomarker_list: [{
                **biomarker, 'categories': [category]
            } for biomarker in biomarker_list]), flatten,
        partial(groupby, lambda biomarker: biomarker['feature']),
        partial(
            valmap,
            partial(
                merge_by, lambda x, y: {
                    **x, 'categories': [*x['categories'], *y['categories']]
                })))
