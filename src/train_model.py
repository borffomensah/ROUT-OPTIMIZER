import joblib
from data_preprocessing import load_and_preprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def run_pipeline():
  X, y, encoders = load_and_preprocess('data/rout_ghana_calibrated.csv')
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  model = RandomForestClassifier(
      n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
  )
  model.fit(X_train, y_train)

  joblib.dump(model, 'models/route_recommendation_model.pkl')
  joblib.dump(encoders, 'models/feature_encoders.pkl')
  print('Training successful. Models exported to models/')


if __name__ == '__main__':
  run_pipeline()