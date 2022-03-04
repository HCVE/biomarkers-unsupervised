import pandas

from deps.utils import assert_equals

data_biomarkers = pandas.read_csv(
    '../data/LV_inflamm.csv')

data_biomarkers.columns = [
    column.upper() for column in data_biomarkers.columns
]

data_biomarkers_19_X = data_biomarkers[[
    'TNFRSF10A', 'PGF', 'KIM1', 'GAL_9', 'CTSL1', 'TRAIL_R2', 'TIE2', 'PRSS8',
    'TNFRSF11A', 'RAGE', 'MMP7', 'ADAM_TS13', 'IL6', 'GH', 'AMBP', 'SPON2',
    'IL16', 'GDF_2', 'ACE2'
]]

data_biomarkers_13_X = data_biomarkers[[
    'ACE2', 'AMBP', 'IL16', 'GAL_9', 'MMP7', 'PRSS8', 'KIM1', 'CTSL1', 'PGF',
    'TNFRSF11A', 'TRAIL_R2', 'IL6', 'TNFRSF10A'
]]

data_biomarkers_25_X = data_biomarkers[[
    'TNFRSF10A', 'PGF', 'KIM1', 'GAL_9', 'CTSL1', 'TRAIL_R2', 'TIE2', 'PRSS8',
    'TNFRSF11A', 'RAGE', 'MMP7', 'ADAM_TS13', 'IL6', 'GH', 'AMBP', 'SPON2',
    'IL16', 'GDF_2', 'ACE2', 'DCN', 'IGG_FC_RECEPTOR_II_B', 'IL_1RA', 'CD84',
    'CA5A', 'BNP'
]]

assert_equals(len(data_biomarkers_13_X.columns), 13)

data_biomarkers_all_X = data_biomarkers.drop(
    axis='columns',
    labels=[
        "CPNBRX",
        "RWT",
        "LVMI",
        "RMVEA",
        "REEM",
        "GS",
        "LA_GS",
        "RWT_ADJ",
        "LVMI_ADJ",
        "RMVEA_ADJ",
        "REEM_ADJ",
        "GS_ADJ",
        "LA_GS_ADJ",
        "LVCR",
        "LVH",
        "LV_REMODELING",
        "LVDDF",
        "LVSDF",
        "LA_DYSF",
    ],
)

y_labels = ('LVDDF', 'LV_REMODELING', 'LA_DYSF')

data_biomarkers_all_y = {key: data_biomarkers[key] for key in y_labels}
