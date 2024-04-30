import pandas as pd

from autogluon.tabular import TabularDataset, TabularPredictor
import torch

import os

from sklearn.metrics import mean_squared_log_error
from autogluon.core.metrics import make_scorer


num_devices = torch.cuda.device_count()
print(f"Available Device Counts: {num_devices}")

train = pd.read_csv('../dataset/DaeguTrafficAccident/train_1201.csv')
test = pd.read_csv('../dataset/DaeguTrafficAccident/test_1201.csv')

train_data = TabularDataset(train)
test_data = TabularDataset(test)
train_data.head()

drops = list(set(train.columns) - set(test.columns))
drops.remove('ECLO')

train_data.drop(drops, axis=1, inplace=True)
train_data['사고일시'] = pd.to_datetime(train_data['사고일시'])
test_data['사고일시'] = pd.to_datetime(test_data['사고일시'])

# cat_cols = [x for x in train.columns if x.dtype]

label = "ECLO"

rmsle = make_scorer(name='RMSLE',
                    score_func=mean_squared_log_error,
                    optimum=1,
                    greater_is_better=False)

predictor = TabularPredictor(label=label, eval_metric=rmsle, problem_type='regression').fit(train_data,
                                                                                            num_gpus=num_devices,
                                                                                            num_cpus=os.cpu_count()
                                                                                            )
print(predictor.leaderboard(train_data, extra_metrics=[rmsle], silent=True))

pred = predictor.predict(test_data)

submission = pd.read_csv('../dataset/DaeguTrafficAccident/sample_submission.csv')
submission[label] = pred

submission.to_csv('../dataset/DaeguTrafficAccident/submission_autogluon_1201.csv', index=False)