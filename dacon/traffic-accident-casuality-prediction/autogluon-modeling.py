import pandas as pd

from autogluon.tabular import TabularDataset, TabularPredictor
import torch

import os

from sklearn.metrics import mean_squared_log_error
from autogluon.core.metrics import make_scorer


class AutoGluonRegression(TabularDataset):
    def __init__(self,
                 train_path='../dataset/DaeguTrafficAccident/train_1201.csv',
                 test_path='../dataset/DaeguTrafficAccident/test_1201.csv',
                 ):
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)

    def get_dataset(self, data):
        df = TabularDataset(data)
        df['사고일시'] = pd.to_datetime(df['사고일시'])
        return df

    def make_rmsle(self):
        scorer = make_scorer(name='RMSLE',
                             score_func=mean_squared_log_error,
                             optimum=1,
                             greater_is_better=True)
        return scorer

    def preprocess(self, train, test):
        drops = list(set(train.columns) - set(test.columns))
        drops.remove('ECLO')

        train_data = self.get_dataset(train)
        test_data = self.get_dataset(test)

        train_data.drop(drops, axis=1)

        return train_data, test_data



    def predict(self):

        train_data, test_data = self.preprocess(self.train, self.test)

        rmsle = self.make_rmsle()

        predictor = TabularPredictor(label='ECLO', eval_metric=rmsle, problem_type='regression').fit(train_data,
                                                                                            num_gpus=num_devices,
                                                                                            num_cpus=os.cpu_count()
                                                                                                    )
        print(predictor.leaderboard(train_data, extra_metrics=[rmsle], silent=True))

        preds = predictor.predict(test_data)
        return preds


def main():
    autogluon_reg = AutoGluonRegression()
    preds = autogluon_reg.predict()

    submission = pd.read_csv('../dataset/DaeguTrafficAccident/sample_submission.csv')
    submission['ECLO'] = preds

if __name__ == '__main__':

    num_devices = torch.cuda.device_count()
    print(f"Available Device Counts: {num_devices}")

    main()