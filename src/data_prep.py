import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import sys
sys.path.insert(0, '.')

CATEGORICAL_COLS = [
    'location_sharing', 'password_reuse',
    'cookie_acceptance', 'os_update_frequency'
]
TARGET_COL = 'risk_level'
FEATURE_COLS = [
    'location_sharing', 'location_history_days',
    'apps_with_camera_access', 'apps_with_mic_access',
    'apps_with_contacts_access', 'social_media_accounts',
    'public_profile', 'posts_per_week', 'two_factor_auth',
    'password_reuse', 'vpn_usage', 'third_party_data_sharing',
    'cookie_acceptance', 'data_broker_opted_out', 'os_update_frequency'
]


def full_pipeline(csv_path: str = 'data/raw/dataset.csv'):
    """
    Complete pipeline: load → engineer features → encode → scale → split.
    Saves encoders + scaler to models/encoders.pkl.
    Returns X_train, X_test, y_train, y_test, feature_names, encoders
    """
    from src.feature_engineering import engineer_features

    df = pd.read_csv(csv_path).dropna()
    df = engineer_features(df)

    engineered_cols = [
        'permission_exposure', 'security_score',
        'social_exposure', 'location_risk_index', 'privacy_index'
    ]
    all_features = FEATURE_COLS + engineered_cols

    encoders = {}
    for col in CATEGORICAL_COLS:
        if df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df[TARGET_COL].astype(str))
    encoders['target'] = target_encoder
    encoders['feature_names'] = all_features

    X = df[all_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    encoders['scaler'] = scaler

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs('models', exist_ok=True)
    joblib.dump(encoders, 'models/encoders.pkl')

    print(f"✅ Pipeline complete | Features: {len(all_features)} | "
          f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"   Classes: {list(target_encoder.classes_)}")
    return X_train, X_test, y_train, y_test, all_features, encoders


def prepare_single_record(record: dict, encoders: dict) -> np.ndarray:
    """Prepare one user record for API prediction."""
    from src.feature_engineering import engineer_features
    import pandas as pd

    df = pd.DataFrame([record])
    try:
        df = engineer_features(df)
    except Exception:
        pass

    feature_names = encoders.get('feature_names', FEATURE_COLS)
    row = []
    for col in feature_names:
        val = record.get(col, 0)
        if col in CATEGORICAL_COLS and col in encoders:
            try:
                val = encoders[col].transform([str(val)])[0]
            except ValueError:
                val = 0
        # Try engineered features from df
        if col not in FEATURE_COLS and col in df.columns:
            val = float(df[col].values[0])
        row.append(float(val))

    return encoders['scaler'].transform([row])

if __name__ == "__main__":
    full_pipeline()