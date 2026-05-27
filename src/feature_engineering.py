import pandas as pd
import numpy as np
import os


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 5 engineered features from raw data.
    Each captures a privacy risk concept the raw features miss.
    """
    df = df.copy()

    # Feature 1: Permission Exposure Score
    # Camera/mic = high risk (2x weight), contacts = medium
    df['permission_exposure'] = (
        df['apps_with_camera_access'] * 2 +
        df['apps_with_mic_access'] * 2 +
        df['apps_with_contacts_access'] * 1
    )

    # Feature 2: Security Score (higher = better security = lower risk)
    df['security_score'] = (
        df['two_factor_auth'] * 3 +
        df['vpn_usage'] * 2 +
        df['data_broker_opted_out'] * 1
    )

    # Feature 3: Social Exposure Score
    df['social_exposure'] = (
        df['social_media_accounts'] * 2 +
        df['public_profile'] * 5 +
        df['posts_per_week'] * 0.5
    ).round(1)

    # Feature 4: Location Risk Index (categorical → ordinal 0-2)
    if df['location_sharing'].dtype == object:
        loc_map = {'never': 0, 'while_using': 1, 'always': 2}
        df['location_risk_index'] = df['location_sharing'].map(loc_map).fillna(1)
    else:
        df['location_risk_index'] = df['location_sharing']

    # Feature 5: Privacy Index (composite — higher = more privacy-conscious)
    if df['password_reuse'].dtype == object:
        pw_map = {'never': 0, 'sometimes': 1, 'always': 2}
        ck_map = {'never': 0, 'selective': 1, 'always': 2}
        pw_risk = df['password_reuse'].map(pw_map).fillna(1)
        ck_risk = df['cookie_acceptance'].map(ck_map).fillna(1)
    else:
        pw_risk = df['password_reuse']
        ck_risk = df['cookie_acceptance']

    df['privacy_index'] = (
        df['security_score'] * 2
        - df['permission_exposure'] * 0.5
        - pw_risk * 3
        - ck_risk * 2
        - df['third_party_data_sharing'] * 3
    )

    return df


if __name__ == "__main__":
    df = pd.read_csv('data/raw/dataset.csv')
    df_eng = engineer_features(df)
    os.makedirs('data/processed', exist_ok=True)
    df_eng.to_csv('data/processed/dataset_engineered.csv', index=False)
    new_cols = ['permission_exposure', 'security_score',
                'social_exposure', 'location_risk_index', 'privacy_index']
    print("✅ Feature engineering complete")
    for col in new_cols:
        print(f"   {col}: mean={df_eng[col].mean():.2f}, "
              f"min={df_eng[col].min():.2f}, max={df_eng[col].max():.2f}")