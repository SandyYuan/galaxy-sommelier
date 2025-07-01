"""
Standard 26-Feature Mapping for Galaxy Sommelier
All models must use this exact feature order for fair comparison.
Generated from four_survey_feature_mapping.csv
"""

# Standard feature order (index -> feature info)
STANDARD_26_FEATURES = [
    {
        'index': 0,
        'feature_name': 'smooth_or_features_smooth',
        'sdss_column': 't01_smooth_or_features_a01_smooth_fraction',
        'decals_column': 'smooth-or-featured_smooth_fraction',
        'hst_column': 't01_smooth_or_features_a01_smooth_fraction',
        'ukidss_column': 'smooth-or-featured-ukidss_smooth_fraction'
    },
    {
        'index': 1,
        'feature_name': 'smooth_or_features_features_disk',
        'sdss_column': 't01_smooth_or_features_a02_features_or_disk_fraction',
        'decals_column': 'smooth-or-featured_featured-or-disk_fraction',
        'hst_column': 't01_smooth_or_features_a02_features_or_disk_fraction',
        'ukidss_column': 'smooth-or-featured-ukidss_featured-or-disk_fraction'
    },
    {
        'index': 2,
        'feature_name': 'smooth_or_features_artifact',
        'sdss_column': 't01_smooth_or_features_a03_star_or_artifact_fraction',
        'decals_column': 'smooth-or-featured_artifact_fraction',
        'hst_column': 't01_smooth_or_features_a03_star_or_artifact_fraction',
        'ukidss_column': 'smooth-or-featured-ukidss_artifact_fraction'
    },
    {
        'index': 3,
        'feature_name': 'edge_on_yes',
        'sdss_column': 't02_edgeon_a04_yes_fraction',
        'decals_column': 'disk-edge-on_yes_fraction',
        'hst_column': 't02_edgeon_a01_yes_fraction',
        'ukidss_column': 'disk-edge-on-ukidss_yes_fraction'
    },
    {
        'index': 4,
        'feature_name': 'edge_on_no',
        'sdss_column': 't02_edgeon_a05_no_fraction',
        'decals_column': 'disk-edge-on_no_fraction',
        'hst_column': 't02_edgeon_a02_no_fraction',
        'ukidss_column': 'disk-edge-on-ukidss_no_fraction'
    },
    {
        'index': 5,
        'feature_name': 'bar_yes',
        'sdss_column': 't03_bar_a06_bar_fraction',
        'decals_column': 'bar_yes_fraction',
        'hst_column': 't03_bar_a01_bar_fraction',
        'ukidss_column': 'bar-ukidss_yes_fraction'
    },
    {
        'index': 6,
        'feature_name': 'bar_no',
        'sdss_column': 't03_bar_a07_no_bar_fraction',
        'decals_column': 'bar_no_fraction',
        'hst_column': 't03_bar_a02_no_bar_fraction',
        'ukidss_column': 'bar-ukidss_no_fraction'
    },
    {
        'index': 7,
        'feature_name': 'spiral_arms_yes',
        'sdss_column': 't04_spiral_a08_spiral_fraction',
        'decals_column': 'has-spiral-arms_yes_fraction',
        'hst_column': 't04_spiral_a01_spiral_fraction',
        'ukidss_column': 'has-spiral-arms-ukidss_yes_fraction'
    },
    {
        'index': 8,
        'feature_name': 'spiral_arms_no',
        'sdss_column': 't04_spiral_a09_no_spiral_fraction',
        'decals_column': 'has-spiral-arms_no_fraction',
        'hst_column': 't04_spiral_a02_no_spiral_fraction',
        'ukidss_column': 'has-spiral-arms-ukidss_no_fraction'
    },
    {
        'index': 9,
        'feature_name': 'bulge_size_none',
        'sdss_column': 't05_bulge_prominence_a10_no_bulge_fraction',
        'decals_column': 'bulge-size_none_fraction',
        'hst_column': 't05_bulge_prominence_a01_no_bulge_fraction',
        'ukidss_column': 'bulge-size-ukidss_no_fraction'
    },
    {
        'index': 10,
        'feature_name': 'bulge_size_obvious',
        'sdss_column': 't05_bulge_prominence_a12_obvious_fraction',
        'decals_column': 'bulge-size_obvious_fraction',
        'hst_column': 't05_bulge_prominence_a03_obvious_fraction',
        'ukidss_column': 'bulge-size-ukidss_obvious_fraction'
    },
    {
        'index': 11,
        'feature_name': 'bulge_size_dominant',
        'sdss_column': 't05_bulge_prominence_a13_dominant_fraction',
        'decals_column': 'bulge-size_dominant_fraction',
        'hst_column': 't05_bulge_prominence_a04_dominant_fraction',
        'ukidss_column': 'bulge-size-ukidss_dominant_fraction'
    },
    {
        'index': 12,
        'feature_name': 'rounded_completely_round',
        'sdss_column': 't07_rounded_a16_completely_round_fraction',
        'decals_column': 'how-rounded_completely_fraction',
        'hst_column': 't07_rounded_a01_completely_round_fraction',
        'ukidss_column': 'how-rounded-ukidss_round_fraction'
    },
    {
        'index': 13,
        'feature_name': 'rounded_in_between',
        'sdss_column': 't07_rounded_a17_in_between_fraction',
        'decals_column': 'how-rounded_in-between_fraction',
        'hst_column': 't07_rounded_a02_in_between_fraction',
        'ukidss_column': 'how-rounded-ukidss_in-between_fraction'
    },
    {
        'index': 14,
        'feature_name': 'rounded_cigar_shaped',
        'sdss_column': 't07_rounded_a18_cigar_shaped_fraction',
        'decals_column': 'how-rounded_cigar-shaped_fraction',
        'hst_column': 't07_rounded_a03_cigar_shaped_fraction',
        'ukidss_column': 'how-rounded-ukidss_cigar_fraction'
    },
    {
        'index': 15,
        'feature_name': 'spiral_winding_tight',
        'sdss_column': 't10_arms_winding_a28_tight_fraction',
        'decals_column': 'spiral-winding_tight_fraction',
        'hst_column': 't10_arms_winding_a01_tight_fraction',
        'ukidss_column': 'spiral-winding-ukidss_tight_fraction'
    },
    {
        'index': 16,
        'feature_name': 'spiral_winding_medium',
        'sdss_column': 't10_arms_winding_a29_medium_fraction',
        'decals_column': 'spiral-winding_medium_fraction',
        'hst_column': 't10_arms_winding_a02_medium_fraction',
        'ukidss_column': 'spiral-winding-ukidss_medium_fraction'
    },
    {
        'index': 17,
        'feature_name': 'spiral_winding_loose',
        'sdss_column': 't10_arms_winding_a30_loose_fraction',
        'decals_column': 'spiral-winding_loose_fraction',
        'hst_column': 't10_arms_winding_a03_loose_fraction',
        'ukidss_column': 'spiral-winding-ukidss_loose_fraction'
    },
    {
        'index': 18,
        'feature_name': 'spiral_count_1',
        'sdss_column': 't11_arms_number_a31_1_fraction',
        'decals_column': 'spiral-arm-count_1_fraction',
        'hst_column': 't11_arms_number_a01_1_fraction',
        'ukidss_column': 'spiral-arm-count-ukidss_1_fraction'
    },
    {
        'index': 19,
        'feature_name': 'spiral_count_2',
        'sdss_column': 't11_arms_number_a32_2_fraction',
        'decals_column': 'spiral-arm-count_2_fraction',
        'hst_column': 't11_arms_number_a02_2_fraction',
        'ukidss_column': 'spiral-arm-count-ukidss_2_fraction'
    },
    {
        'index': 20,
        'feature_name': 'spiral_count_3',
        'sdss_column': 't11_arms_number_a33_3_fraction',
        'decals_column': 'spiral-arm-count_3_fraction',
        'hst_column': 't11_arms_number_a03_3_fraction',
        'ukidss_column': 'spiral-arm-count-ukidss_3_fraction'
    },
    {
        'index': 21,
        'feature_name': 'spiral_count_4',
        'sdss_column': 't11_arms_number_a34_4_fraction',
        'decals_column': 'spiral-arm-count_4_fraction',
        'hst_column': 't11_arms_number_a04_4_fraction',
        'ukidss_column': 'spiral-arm-count-ukidss_4_fraction'
    },
    {
        'index': 22,
        'feature_name': 'spiral_count_more_than_4',
        'sdss_column': 't11_arms_number_a36_more_than_4_fraction',
        'decals_column': 'spiral-arm-count_more-than-4_fraction',
        'hst_column': 't11_arms_number_a05_more_than_4_fraction',
        'ukidss_column': 'spiral-arm-count-ukidss_more-than-4_fraction'
    },
    {
        'index': 23,
        'feature_name': 'bulge_shape_rounded',
        'sdss_column': 't09_bulge_shape_a25_rounded_fraction',
        'decals_column': 'edge-on-bulge_rounded_fraction',
        'hst_column': 't09_bulge_shape_a01_rounded_fraction',
        'ukidss_column': 'bulge-shape-ukidss_round_fraction'
    },
    {
        'index': 24,
        'feature_name': 'bulge_shape_boxy',
        'sdss_column': 't09_bulge_shape_a26_boxy_fraction',
        'decals_column': 'edge-on-bulge_boxy_fraction',
        'hst_column': 't09_bulge_shape_a02_boxy_fraction',
        'ukidss_column': 'bulge-shape-ukidss_boxy_fraction'
    },
    {
        'index': 25,
        'feature_name': 'bulge_shape_no_bulge',
        'sdss_column': 't09_bulge_shape_a27_no_bulge_fraction',
        'decals_column': 'edge-on-bulge_none_fraction',
        'hst_column': 't09_bulge_shape_a03_no_bulge_fraction',
        'ukidss_column': 'bulge-shape-ukidss_no-bulge_fraction'
    },
]

# Quick access mappings
FEATURE_NAMES = [f['feature_name'] for f in STANDARD_26_FEATURES]
SDSS_COLUMNS = [f['sdss_column'] for f in STANDARD_26_FEATURES] 
DECALS_COLUMNS = [f['decals_column'] for f in STANDARD_26_FEATURES]
HST_COLUMNS = [f['hst_column'] for f in STANDARD_26_FEATURES]
UKIDSS_COLUMNS = [f['ukidss_column'] for f in STANDARD_26_FEATURES]

def get_feature_order():
    """Returns the standard 26-feature order for model outputs"""
    return FEATURE_NAMES

def get_survey_columns(survey):
    """Get column names for a specific survey in standard order"""
    survey_map = {
        'sdss': SDSS_COLUMNS,
        'decals': DECALS_COLUMNS, 
        'hst': HST_COLUMNS,
        'ukidss': UKIDSS_COLUMNS
    }
    return survey_map.get(survey.lower(), [])
