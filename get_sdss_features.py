import pandas as pd

catalog_path = '/pscratch/sd/s/sihany/galaxy-sommelier-data/catalogs/gz2_master_catalog_corrected.csv'
df = pd.read_csv(catalog_path)

feature_columns = [col for col in df.columns if '_fraction' in col and col.startswith('t')]

print(f"Number of features: {len(feature_columns)}")
for col in feature_columns:
    print(col) 