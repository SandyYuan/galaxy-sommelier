"""
Unified Feature Registry for Galaxy Sommelier
Single source of truth for all feature mappings, model types, and indexing systems.

Replaces:
- standard_26_features.py
- model_feature_registry.py  
- sdss_decals_feature_mapping.py (partial)

Usage:
    from feature_registry import FeatureRegistry
    
    # Get features for any model
    features = FeatureRegistry.get_benchmark_features('mixed_model')
    indices = FeatureRegistry.get_feature_indices('/path/to/model', config)
    
    # Survey mappings
    ukidss_cols = FeatureRegistry.get_survey_columns('ukidss')
    sdss_to_decals = FeatureRegistry.get_survey_mapping('sdss', 'decals')
"""

import pandas as pd
import os
from pathlib import Path
from typing import Optional, Dict, List, Any

class FeatureRegistry:
    """Unified registry for all Galaxy Sommelier feature mappings."""
    
    _features_df: Optional[pd.DataFrame] = None
    _initialized: bool = False
    
    # The 9 key morphological features used for benchmarking across all models
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
    
    # Model type definitions with their indexing characteristics
    MODEL_TYPES = {
        'standard_26': {
            'description': 'New models using standard 26-feature mapping',
            'output_dimensions': 26,
            'feature_indices': list(range(9)),  # [0, 1, 2, 3, 4, 5, 6, 7, 8]
            'feature_order': 'consecutive',
            'detection_patterns': ['mixed', 'triple_mixed', 'standard_26'],
            'config_indicators': ['use_standard_mapping: true']
        },
        'legacy_scattered': {
            'description': 'Old models with scattered feature indices (74 outputs)',
            'output_dimensions': 74,  # 74-output models
            'feature_indices': [11, 17, 23, 29, 35, 41, 47, 53, 59],  # Original scattered indices
            'feature_order': 'scattered',
            'detection_patterns': ['pretrained', 'finetuned', 'headtrained'],
            'config_indicators': ['use_standard_mapping: false']
        },
        'legacy_52_mixed': {
            'description': '52-output mixed model with specific indices',
            'output_dimensions': 52,
            'feature_indices': [11, 17, 23, 29, 35, 41, 47, 48, 49],  # Adjusted for 52-output model
            'feature_order': 'scattered',
            'detection_patterns': ['mixed_firstrun'],
            'config_indicators': ['use_standard_mapping: false']
        },
        'legacy_74_scattered': {
            'description': '74-output models with full scattered indices',
            'output_dimensions': 74,
            'feature_indices': [11, 17, 23, 29, 35, 41, 47, 53, 59],  # Original scattered indices
            'feature_order': 'scattered',
            'detection_patterns': ['pretrained', 'finetuned', 'headtrained'],
            'config_indicators': ['use_standard_mapping: false']
        }
    }
    
    # Registry for specific known models (can be extended)
    KNOWN_MODELS = {
        # New standardized 26-feature models
        '/pscratch/sd/s/sihany/galaxy-sommelier-data/models/mixed': 'standard_26',
        '/pscratch/sd/s/sihany/galaxy-sommelier-data/models/triple_mixed': 'standard_26',
        
        # Legacy scattered models (old _firstrun models with 52/74 features)
        '/pscratch/sd/s/sihany/galaxy-sommelier-data/models/mixed_firstrun': 'legacy_52_mixed',
        '/pscratch/sd/s/sihany/galaxy-sommelier-data/models/max_overlap_firstrun': 'legacy_scattered', 
        '/pscratch/sd/s/sihany/galaxy-sommelier-data/models/mixed_high_quality_firstrun': 'legacy_scattered',
        '/pscratch/sd/s/sihany/galaxy-sommelier-data/models/sdss_only_firstrun': 'legacy_scattered',
        
        # Legacy models from benchmark results
        'models/pretrained_baseline': 'legacy_scattered',
        'models/finetuned_model_6': 'legacy_scattered',
        'models/headtrained_model_6': 'legacy_scattered',
        
        # Alternative path patterns for flexibility
        'mixed_firstrun': 'legacy_52_mixed',
        'max_overlap_firstrun': 'legacy_scattered',
        'mixed_high_quality_firstrun': 'legacy_scattered', 
        'sdss_only_firstrun': 'legacy_scattered'
    }
    
    @classmethod
    def _initialize(cls):
        """Load feature mapping data from CSV."""
        if cls._initialized:
            return
            
        # Find the CSV file
        csv_path = Path(__file__).parent / 'four_survey_feature_mapping.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"Feature mapping CSV not found: {csv_path}")
        
        # Load the feature mapping
        cls._features_df = pd.read_csv(csv_path)
        cls._initialized = True
    
    @classmethod
    def get_features_dataframe(cls):
        """Get the complete features dataframe."""
        cls._initialize()
        return cls._features_df.copy()
    
    @classmethod
    def get_feature_list(cls, survey='sdss'):
        """Get ordered list of all 26 features for a survey."""
        cls._initialize()
        column_name = f'{survey}_column'
        if column_name not in cls._features_df.columns:
            raise ValueError(f"Survey '{survey}' not found. Available: {cls.get_available_surveys()}")
        return cls._features_df[column_name].tolist()
    
    @classmethod
    def get_survey_columns(cls, survey):
        """Get column names for a specific survey in standard order."""
        return cls.get_feature_list(survey)
    
    @classmethod
    def get_available_surveys(cls):
        """Get list of available surveys."""
        cls._initialize()
        return [col.replace('_column', '') for col in cls._features_df.columns if col.endswith('_column')]
    
    @classmethod
    def get_survey_mapping(cls, from_survey, to_survey):
        """Get mapping dictionary from one survey to another."""
        cls._initialize()
        from_col = f'{from_survey}_column'
        to_col = f'{to_survey}_column'
        
        if from_col not in cls._features_df.columns:
            raise ValueError(f"Source survey '{from_survey}' not found")
        if to_col not in cls._features_df.columns:
            raise ValueError(f"Target survey '{to_survey}' not found")
        
        return dict(zip(cls._features_df[from_col], cls._features_df[to_col]))
    
    @classmethod
    def get_sdss_to_decals_mapping(cls):
        """Get SDSS to DECaLS mapping (backwards compatibility)."""
        mapping = cls.get_survey_mapping('sdss', 'decals')
        
        # Add coordinate mappings for full compatibility
        mapping.update({
            'dec': 'dec',
            'dr7objid': 'iauname', 
            'ra': 'ra'
        })
        
        # Add debiased versions (pattern-based)
        debiased_mapping = {}
        for sdss_col, decals_col in list(mapping.items()):
            if '_fraction' in sdss_col:
                sdss_debiased = sdss_col.replace('_fraction', '_debiased')
                decals_debiased = decals_col.replace('_fraction', '_debiased')
                debiased_mapping[sdss_debiased] = decals_debiased
        
        mapping.update(debiased_mapping)
        return mapping
    
    @classmethod
    def get_decals_to_sdss_mapping(cls):
        """Get DECaLS to SDSS mapping (backwards compatibility)."""
        sdss_to_decals = cls.get_sdss_to_decals_mapping()
        return {v: k for k, v in sdss_to_decals.items()}
    
    @classmethod
    def detect_model_type(cls, model_path=None, config=None, model_name=None):
        """
        Detect model type based on various indicators.
        
        Args:
            model_path: Path to model directory
            config: Model configuration dict
            model_name: Model name/identifier
            
        Returns:
            str: Model type key ('standard_26' or 'legacy_scattered')
        """
        
        # Method 1: Check known models registry
        if model_path and model_path in cls.KNOWN_MODELS:
            return cls.KNOWN_MODELS[model_path]
        
        # Method 2: Check config for explicit indicators
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
        
        # Method 3: Check model path/name patterns
        search_text = ""
        if model_path:
            search_text += model_path.lower()
        if model_name:
            search_text += " " + model_name.lower()
        
        # Check for standard_26 patterns first (more specific)
        for pattern in cls.MODEL_TYPES['standard_26']['detection_patterns']:
            if pattern in search_text:
                return 'standard_26'
        
        # Check for legacy patterns
        for pattern in cls.MODEL_TYPES['legacy_scattered']['detection_patterns']:
            if pattern in search_text:
                return 'legacy_scattered'
        
        # Default: if uncertain, assume new standard format
        return 'standard_26'
    
    @classmethod
    def get_feature_indices(cls, model_path=None, config=None, model_name=None):
        """Get feature indices for a model's output."""
        model_type = cls.detect_model_type(model_path, config, model_name)
        return cls.MODEL_TYPES[model_type]['feature_indices']
    
    @classmethod
    def get_benchmark_features(cls, model_path=None, config=None, model_name=None):
        """Get the 9 benchmark feature names (SDSS column names) for any model."""
        return cls.BENCHMARK_9_FEATURES.copy()
    
    @classmethod
    def get_model_feature_mapping(cls, model_path=None, config=None, model_name=None, survey='sdss'):
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
        cls._initialize()
        
        model_type = cls.detect_model_type(model_path, config, model_name)
        type_info = cls.MODEL_TYPES[model_type]
        
        # Get survey-specific column names
        survey_columns = cls.get_survey_columns(survey)
        
        # Map the 9 benchmark features to their indices and survey columns
        feature_mapping = []
        for i, benchmark_feature in enumerate(cls.BENCHMARK_9_FEATURES):
            # Find this feature in the feature dataframe
            feature_row = cls._features_df[cls._features_df['sdss_column'] == benchmark_feature]
            
            if not feature_row.empty:
                row = feature_row.iloc[0]
                survey_col = f'{survey}_column'
                feature_mapping.append({
                    'benchmark_index': i,
                    'model_output_index': type_info['feature_indices'][i],
                    'feature_name': row['feature_name'],
                    'description': row['description'],
                    'sdss_column': row['sdss_column'],
                    'survey_column': row[survey_col] if survey_col in row else row['sdss_column']
                })
        
        return {
            'model_type': model_type,
            'model_info': type_info,
            'features': feature_mapping,
            'output_dimensions': type_info['output_dimensions'],
            'feature_indices': type_info['feature_indices']
        }
    
    @classmethod
    def register_model(cls, model_path, model_type):
        """Register a specific model path with its type."""
        cls.KNOWN_MODELS[model_path] = model_type
    
    @classmethod
    def print_summary(cls):
        """Print a summary of the feature registry."""
        cls._initialize()
        
        print("=== Galaxy Sommelier Feature Registry ===\n")
        print(f"Total features: {len(cls._features_df)}")
        print(f"Available surveys: {', '.join(cls.get_available_surveys())}")
        print(f"Benchmark features: {len(cls.BENCHMARK_9_FEATURES)}")
        print(f"Model types: {', '.join(cls.MODEL_TYPES.keys())}")
        print(f"Known models: {len(cls.KNOWN_MODELS)}")
        
        print("\n--- Benchmark Features ---")
        for i, feat in enumerate(cls.BENCHMARK_9_FEATURES):
            print(f"{i}: {feat}")
        
        print("\n--- Model Types ---")
        for model_type, info in cls.MODEL_TYPES.items():
            print(f"{model_type}: {info['description']}")
            print(f"  Indices: {info['feature_indices']}")
            print(f"  Patterns: {info['detection_patterns']}")

# Convenience functions for backwards compatibility
def get_survey_columns(survey):
    """Get column names for a specific survey."""
    return FeatureRegistry.get_survey_columns(survey)

def get_benchmark_features_for_model(model_path=None, config=None, model_name=None):
    """Get benchmark features for a model."""
    return FeatureRegistry.get_benchmark_features(model_path, config, model_name)

def get_model_feature_indices(model_path=None, config=None, model_name=None):
    """Get feature indices for a model."""
    return FeatureRegistry.get_feature_indices(model_path, config, model_name)

def get_model_feature_mapping(model_path=None, config=None, model_name=None, survey='sdss'):
    """Get complete feature mapping for a model."""
    return FeatureRegistry.get_model_feature_mapping(model_path, config, model_name, survey)

# Legacy compatibility for existing code
SDSS_TO_DECALS = FeatureRegistry.get_sdss_to_decals_mapping()
DECALS_TO_SDSS = FeatureRegistry.get_decals_to_sdss_mapping()

if __name__ == "__main__":
    # Demo the unified system
    FeatureRegistry.print_summary()
    
    print("\n=== Demo: Model Feature Detection ===")
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
        print(f"\n--- {case['name']} ---")
        mapping = FeatureRegistry.get_model_feature_mapping(
            model_path=case['path'],
            config=case['config'],
            survey='ukidss'
        )
        
        print(f"Type: {mapping['model_type']}")
        print(f"Dimensions: {mapping['output_dimensions']}")
        print(f"Indices: {mapping['feature_indices']}")
        print(f"First 3 features:")
        for feat in mapping['features'][:3]:
            print(f"  {feat['feature_name']} -> index {feat['model_output_index']} -> {feat['survey_column']}") 