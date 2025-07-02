"""
Model Feature Registry for Galaxy Sommelier
Maps different model types/paths to their specific feature indexing systems.
Works with standard_26_features.py to provide complete feature mapping.
"""

from standard_26_features import STANDARD_26_FEATURES, get_survey_columns
import os

# Define the 9 key morphological features used for benchmarking
# These are the same features across all models, just at different indices
BENCHMARK_9_FEATURES = [
    't01_smooth_or_features_a01_smooth_fraction',
    't01_smooth_or_features_a02_features_or_disk_fraction', 
    't01_smooth_or_features_a03_star_or_artifact_fraction',
    't02_edgeon_a04_yes_fraction',
    't02_edgeon_a05_no_fraction',
    't03_bar_a06_bar_fraction',
    't03_bar_a07_no_bar_fraction',
    't04_spiral_a08_spiral_fraction',
    't04_spiral_a09_no_spiral_fraction'
]

# Model type definitions with their feature mapping characteristics
MODEL_TYPES = {
    'standard_26': {
        'description': 'New models using standard 26-feature mapping',
        'output_dimensions': 26,
        'feature_indices': list(range(9)),  # [0, 1, 2, 3, 4, 5, 6, 7, 8]
        'feature_order': 'consecutive',
        'detection_patterns': [
            'mixed',
            'triple_mixed',
            'standard_26'
        ],
        'config_indicators': [
            'standard_features.use_standard_mapping: true',
            'use_standard_mapping: True'
        ]
    },
    'legacy_scattered': {
        'description': 'Old models with scattered feature indices',
        'output_dimensions': [52, 74],  # Variable based on dataset
        'feature_indices': [11, 17, 23, 29, 35, 41, 47, 53, 59],  # Scattered indices
        'feature_order': 'scattered',
        'detection_patterns': [
            'pretrained',
            'finetuned',
            'headtrained'
        ],
        'config_indicators': [
            'standard_features.use_standard_mapping: false',
            'use_standard_mapping: False'
        ]
    }
}

def detect_model_type(model_path=None, config=None, model_name=None):
    """
    Detect model type based on various indicators.
    
    Args:
        model_path: Path to model directory
        config: Model configuration dict
        model_name: Model name/identifier
        
    Returns:
        str: Model type key ('standard_26' or 'legacy_scattered')
    """
    
    # Method 1: Check config for explicit indicators
    if config:
        # Check for standard features flag
        if config.get('standard_features', {}).get('use_standard_mapping'):
            return 'standard_26'
        
        # Check top-level flag
        if config.get('use_standard_mapping'):
            return 'standard_26'
            
        # If explicitly false, it's legacy
        if (config.get('standard_features', {}).get('use_standard_mapping') == False or
            config.get('use_standard_mapping') == False):
            return 'legacy_scattered'
    
    # Method 2: Check model path/name patterns
    search_text = ""
    if model_path:
        search_text += model_path.lower()
    if model_name:
        search_text += " " + model_name.lower()
    
    # Check for standard_26 patterns first (more specific)
    for pattern in MODEL_TYPES['standard_26']['detection_patterns']:
        if pattern in search_text:
            return 'standard_26'
    
    # Check for legacy patterns
    for pattern in MODEL_TYPES['legacy_scattered']['detection_patterns']:
        if pattern in search_text:
            return 'legacy_scattered'
    
    # Default: if uncertain, assume new standard format
    return 'standard_26'

def get_model_feature_mapping(model_path=None, config=None, model_name=None, survey='sdss'):
    """
    Get complete feature mapping for a model.
    
    Args:
        model_path: Path to model directory
        config: Model configuration dict  
        model_name: Model name/identifier
        survey: Target survey for column names ('sdss', 'ukidss', etc.)
        
    Returns:
        dict: Complete feature mapping with indices, names, and survey columns
    """
    
    model_type = detect_model_type(model_path, config, model_name)
    type_info = MODEL_TYPES[model_type]
    
    # Get survey-specific column names for the 9 benchmark features
    survey_columns = get_survey_columns(survey)
    
    # Map the 9 benchmark features to their indices and survey columns
    feature_mapping = []
    for i, benchmark_feature in enumerate(BENCHMARK_9_FEATURES):
        # Find this feature in the standard mapping
        std_feature = None
        for feat in STANDARD_26_FEATURES:
            if feat['sdss_column'] == benchmark_feature:
                std_feature = feat
                break
        
        if std_feature:
            feature_mapping.append({
                'benchmark_index': i,
                'model_output_index': type_info['feature_indices'][i],
                'feature_name': std_feature['feature_name'],
                'sdss_column': std_feature['sdss_column'],
                'survey_column': std_feature.get(f'{survey}_column', std_feature['sdss_column'])
            })
    
    return {
        'model_type': model_type,
        'model_info': type_info,
        'features': feature_mapping,
        'output_dimensions': type_info['output_dimensions'],
        'feature_indices': type_info['feature_indices']
    }

def get_model_feature_indices(model_path=None, config=None, model_name=None):
    """
    Get just the feature indices for a model (for quick lookups).
    
    Returns:
        list: Feature indices in model output
    """
    model_type = detect_model_type(model_path, config, model_name)
    return MODEL_TYPES[model_type]['feature_indices']

def get_model_feature_columns(model_path=None, config=None, model_name=None, survey='sdss'):
    """
    Get just the survey column names for a model (for quick lookups).
    
    Returns:
        list: Survey-specific column names for the 9 benchmark features
    """
    mapping = get_model_feature_mapping(model_path, config, model_name, survey)
    return [feat['survey_column'] for feat in mapping['features']]

# Convenience function for backwards compatibility
def get_benchmark_features_for_model(model_path=None, config=None, model_name=None):
    """
    Get the 9 benchmark feature names (SDSS column names) for any model.
    This maintains compatibility with existing evaluation scripts.
    
    Returns:
        list: The 9 SDSS column names for benchmark features
    """
    return BENCHMARK_9_FEATURES.copy()

# Registry for specific known models (can be extended)
KNOWN_MODELS = {
    '/pscratch/sd/s/sihany/galaxy-sommelier-data/models/mixed': 'standard_26',
    '/pscratch/sd/s/sihany/galaxy-sommelier-data/models/triple_mixed': 'standard_26',
    'models/pretrained_baseline': 'legacy_scattered',
    'models/finetuned_model_6': 'legacy_scattered',
    'models/headtrained_model_6': 'legacy_scattered'
}

def register_known_model(model_path, model_type):
    """Register a specific model path with its type."""
    KNOWN_MODELS[model_path] = model_type

def get_registered_model_type(model_path):
    """Get model type from registry if known."""
    return KNOWN_MODELS.get(model_path)

if __name__ == "__main__":
    # Demo usage
    print("=== Model Feature Registry Demo ===\n")
    
    # Test different model types
    test_cases = [
        {
            'name': 'Mixed Model (New)',
            'path': '/pscratch/sd/s/sihany/galaxy-sommelier-data/models/mixed',
            'config': {'standard_features': {'use_standard_mapping': True}}
        },
        {
            'name': 'Triple Mixed Model (New)', 
            'path': '/pscratch/sd/s/sihany/galaxy-sommelier-data/models/triple_mixed',
            'config': {'use_standard_mapping': True}
        },
        {
            'name': 'Pretrained Model (Legacy)',
            'path': 'models/pretrained_baseline',
            'config': None
        }
    ]
    
    for case in test_cases:
        print(f"--- {case['name']} ---")
        mapping = get_model_feature_mapping(
            model_path=case['path'],
            config=case['config'],
            survey='ukidss'  # For UKIDSS OOD evaluation
        )
        
        print(f"Model Type: {mapping['model_type']}")
        print(f"Output Dimensions: {mapping['output_dimensions']}")
        print(f"Feature Indices: {mapping['feature_indices']}")
        print(f"First 3 features:")
        for feat in mapping['features'][:3]:
            print(f"  {feat['feature_name']} -> index {feat['model_output_index']} -> {feat['survey_column']}")
        print() 