import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error

# Деректерді жүктеу
data = np.random.rand(100, 10)  # 100 сызба, 10 мән

# Деректерді бөлу
X = data[:, :-1]
y = data[:, -1]

# Деректерді негізгі және негізгісіз жазу
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM-гі алгоритммен машиналық модельді жасау
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=100,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)

# Модельді тексеру
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
