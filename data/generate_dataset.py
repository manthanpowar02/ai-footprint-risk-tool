"""
Generate synthetic digital footprint risk dataset.
500 records with 15 realistic privacy/security features.
Run: python data/generate_dataset.py
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 500

data = {
    'location_sharing': np.random.choice(
        ['always', 'while_using', 'never'], n, p=[0.3, 0.45, 0.25]),
    'location_history_days': np.random.randint(0, 365, n),
    'apps_with_camera_access': np.random.randint(0, 15, n),
    'apps_with_mic_access': np.random.randint(0, 12, n),
    'apps_with_contacts_access': np.random.randint(0, 20, n),
    'social_media_accounts': np.random.randint(0, 10, n),
    'public_profile': np.random.choice([0, 1], n, p=[0.4, 0.6]),
    'posts_per_week': np.random.randint(0, 50, n),
    'two_factor_auth': np.random.choice([0, 1], n, p=[0.55, 0.45]),
    'password_reuse': np.random.choice(
        ['never', 'sometimes', 'always'], n, p=[0.3, 0.45, 0.25]),
    'vpn_usage': np.random.choice([0, 1], n, p=[0.65, 0.35]),
    'third_party_data_sharing': np.random.choice([0, 1], n, p=[0.45, 0.55]),
    'cookie_acceptance': np.random.choice(
        ['always', 'selective', 'never'], n, p=[0.5, 0.35, 0.15]),
    'data_broker_opted_out': np.random.choice([0, 1], n, p=[0.75, 0.25]),
    'os_update_frequency': np.random.choice(
        ['immediate', 'delayed', 'never'], n, p=[0.35, 0.45, 0.20]),
}

df = pd.DataFrame(data)

# --- Realistic risk scoring logic ---
risk_score = np.zeros(n)

# Location risk (0-20 pts)
risk_score += (df['location_sharing'] == 'always').astype(int) * 20
risk_score += (df['location_sharing'] == 'while_using').astype(int) * 10
risk_score += df['location_history_days'] / 365 * 15

# Permission risk
risk_score += df['apps_with_camera_access'] * 1.5
risk_score += df['apps_with_mic_access'] * 1.5
risk_score += df['apps_with_contacts_access'] * 0.8

# Social media risk
risk_score += df['social_media_accounts'] * 2
risk_score += df['public_profile'] * 10
risk_score += df['posts_per_week'] * 0.3

# Security deductions (good habits = lower risk)
risk_score -= df['two_factor_auth'] * 15
risk_score -= (df['password_reuse'] == 'never').astype(int) * 10
risk_score += (df['password_reuse'] == 'always').astype(int) * 15
risk_score -= df['vpn_usage'] * 10

# Data sharing risk
risk_score += df['third_party_data_sharing'] * 12
risk_score += (df['cookie_acceptance'] == 'always').astype(int) * 8
risk_score -= df['data_broker_opted_out'] * 8

# OS update risk
risk_score -= (df['os_update_frequency'] == 'immediate').astype(int) * 5
risk_score += (df['os_update_frequency'] == 'never').astype(int) * 10

# Add noise for realism
risk_score += np.random.normal(0, 5, n)
risk_score = np.clip(risk_score, 0, None)
risk_score = (risk_score / risk_score.max() * 100).round(1)

df['risk_score'] = risk_score
df['risk_level'] = pd.cut(
    risk_score,
    bins=[0, 25, 50, 75, 100],
    labels=['Low', 'Medium', 'High', 'Critical'],
    include_lowest=True
)

os.makedirs('data/raw', exist_ok=True)
df.to_csv('data/raw/dataset.csv', index=False)

print(f"✅ Dataset generated: {len(df)} records")
print(f"\nRisk distribution:")
print(df['risk_level'].value_counts().to_string())
print(f"\nFirst 3 rows:")
print(df.head(3).to_string())