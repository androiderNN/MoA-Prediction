from pathlib import Path

proj_dir = Path(__file__).parent.parent

export_dir = proj_dir / 'export'

src_dir = proj_dir / 'src'
param_yaml = src_dir / 'params.yaml'

# データセット
data_dir = proj_dir / 'data'
raw_data_dir = data_dir / 'raws'

train_paths = {
    'raw_features': raw_data_dir / 'train_features.csv',
    'raw_drug': raw_data_dir / 'train_features.csv',
    'raw_target_scored': raw_data_dir / 'train_targets_scored.csv',
    'raw_target_nonscored': raw_data_dir / 'train_targets_scored.csv',
    'processed_dir': data_dir / 'train',
    'y': data_dir / 'train' / 'y.csv',
}

test_paths = {
    'raw_features': raw_data_dir / 'test_features.csv',
    'sample_submission': raw_data_dir / 'sample_submission.csv',
    'processed_dir': data_dir / 'test'
}

# 共通のパス設定
for dic in [train_paths, test_paths]:
    dic['tabular_dir'] = dic['processed_dir'] / 'tabular'
    dic['tabular_x'] = dic['tabular_dir'] / 'x.csv'
    dic['sig_id'] = dic['processed_dir'] / 'sig_id.csv'
