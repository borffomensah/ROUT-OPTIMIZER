import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess(data_path):
  df = pd.read_csv(data_path)
  feature_cols = [
      'start_city',
      'end_city',
      'distance_km',
      'travel_time_min',
      'traffic_level',
      'weather',
      'road_type',
      'accidents',
      'fuel_cost',
      'safety_index',
      'scenic_score',
      'avg_speed_kmph',
  ]

  X = df[feature_cols].copy()
  y = df['recommended']

  encoders = {}
  for col in ['start_city', 'end_city', 'weather', 'road_type']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

  return X, y, encoders