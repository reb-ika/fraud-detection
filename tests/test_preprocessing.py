import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def test_time_since_signup():
    """Test that time_since_signup is always non-negative"""
    df = pd.DataFrame({
        'signup_time': pd.to_datetime(['2015-01-01 10:00:00', '2015-01-01 08:00:00']),
        'purchase_time': pd.to_datetime(['2015-01-01 12:00:00', '2015-01-01 09:00:00'])
    })
    df['time_since_signup'] = (
        df['purchase_time'] - df['signup_time']
    ).dt.total_seconds() / 3600
    assert (df['time_since_signup'] >= 0).all()


def test_hour_of_day_range():
    """Test that hour_of_day is between 0 and 23"""
    df = pd.DataFrame({
        'purchase_time': pd.to_datetime([
            '2015-01-01 00:00:00',
            '2015-01-01 12:00:00',
            '2015-01-01 23:59:59'
        ])
    })
    df['hour_of_day'] = df['purchase_time'].dt.hour
    assert df['hour_of_day'].between(0, 23).all()


def test_no_missing_values_after_scaling():
    """Test StandardScaler produces no NaN values"""
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    assert not np.isnan(scaled).any()


def test_smote_balances_classes():
    """Test that SMOTE produces balanced classes"""
    from imblearn.over_sampling import SMOTE
    X = np.random.randn(100, 5)
    y = np.array([0]*90 + [1]*10)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    counts = pd.Series(y_res).value_counts()
    assert counts[0] == counts[1]


def test_train_test_split_stratified():
    """Test stratified split preserves class ratio"""
    from sklearn.model_selection import train_test_split
    X = np.random.randn(1000, 5)
    y = np.array([0]*900 + [1]*100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    train_fraud_rate = y_train.mean()
    test_fraud_rate = y_test.mean()
    assert abs(train_fraud_rate - test_fraud_rate) < 0.01


def test_random_forest_predicts_binary():
    """Test Random Forest outputs only 0 or 1"""
    X_train = np.random.randn(100, 5)
    y_train = np.array([0]*90 + [1]*10)
    X_test = np.random.randn(20, 5)
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    assert set(preds).issubset({0, 1})