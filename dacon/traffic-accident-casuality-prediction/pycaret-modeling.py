import pandas as pd

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from pycaret.regression import *

class PycaretRegression:
    def __init__(self,
                 train_path='../dataset/DaeguTrafficAccident/train.csv',
                 test_path='../dataset/DaeguTrafficAccident/test.csv'
                 ):
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)

    def split_datetime(self, df, col_name='datetime'): # datetime을 year, month, day, hour로 쪼개기
        df['year'] = pd.DatetimeIndex(df[col_name]).year
        df['month'] = pd.DatetimeIndex(df[col_name]).month
        df['day'] = pd.DatetimeIndex(df[col_name]).day
        df['hour'] = pd.DatetimeIndex(df[col_name]).hour

        return df

    def preprocess(self):

        if len(self.train.columns) == 23: # 한글로 된 컬럼명을 영어로 변환 (부르기/외우기 편하게)
            self.train.columns = ['ID', 'datetime', 'dayWeek', 'weather', 'location',
                                  'roadType', 'roadState',
                                  'accidentType', 'accidentType_detail', 'violation',
                                  'modelF', 'genderF', 'ageF', 'injuryF',  # drive at Fault의 F
                                  'modelV', 'genderV', 'ageV', 'injuryV',  # victim의 V
                                  'death', 'injuredS', 'injuredL', 'injured', # injuredS: Severly injured, injuredL: Lightly injured
                                  'ECLO']
            self.test.columns = ['ID', 'datetime', 'dayWeek', 'weather', 'location',
                                 'roadType', 'roadState',
                                 'accidentType']
        
        # 하지만 이미 컬럼 명이 변경된, 1차 전처리가 끝난 파일일 경우 컬럼명 변경은 생략하고 진행

        # datetime의 데이터 형식을 object에서 datetime으로 변경
        self.train['datetime'] = pd.to_datetime(self.train['datetime'])
        self.test['datetime'] = pd.to_datetime(self.test['datetime'])

        self.train = self.split_datetime(self.train)
        self.test = self.split_datetime(self.test)

        # 날짜를 쪼개놨으니 datetime 버림
        self.train.drop('datetime', axis=1, inplace=True)
        self.test.drop('datetime', axis=1, inplace=True)

        # test에 없는 컬럼은 빼고 학습
        exclude_cols = list(set(self.train.columns) - set(self.test.columns))
        exclude_cols.remove('ECLO')

        train_df = self.train.drop(exclude_cols, axis=1)
        train_df = train_df.drop('ID', axis=1)

        cat_cols = [col for col in train_df.columns if train_df[col].dtype == 'object']
        numeric_cols = [col for col in train_df.columns if train_df[col].dtype == 'float64']

        # categorical data 전처리
        for col in cat_cols:
            enc = LabelEncoder()
            train_df[col] = enc.fit_transform(train_df[col])

        X_test = self.test.drop(['ID'], axis=1)
        for col in cat_cols:
            enc = LabelEncoder()
            X_test[col] = enc.fit_transform(X_test[col])

        # numerical data 전처리
        for col in numeric_cols:
            scaler = StandardScaler()
            scaler.fit(train_df[col].values.reshape(-1, 1))
            train_df[col] = scaler.transform(train_df[col].values.reshape(-1, 1))
            X_test[col] = scaler.transform(X_test[col].values.reshape(-1, 1))

        return train_df, X_test

    def predict(self):
        train_df, X_test = self.preprocess()

        # setup
        setup(train_df,
              target='ECLO',
              session_id=42,
              use_gpu=True,
              train_size=.8,
              normalize=True)

        # 상위 5개의 모델 뽑기
        top5_models = compare_models(n_select=5,
                                     sort="RMSLE",
                                     exclude=['catboost', 'lightgbm', 'rf']) # catboost는 OOM 때문에 제외, lightgbm은 학습 시간이 너무 오래 걸려 제외

        models = []

        for m in top5_models: # 모델 튜닝
            models.append(tune_model(m,
                                     optimize="RMSLE",
                                     choose_better=True,
                                     n_iter=50,
                                     early_stopping='asha'
                                     )
                          )

        # final_model = blend_models(models,
        #                            choose_better=True,
        #                            optimize="RMSLE")
        # 모델 stacking
        final_model = stack_models(models,
                                   choose_better=True,
                                   optimize="RMSLE")

        # 최종 모델 tuning
        final_model = tune_model(final_model,
                                 optimize="RMSLE",
                                 choose_better=True,
                                 n_iter=50,
                                 early_stopping='asha')

        predictions = predict_model(final_model, data=X_test)


        return predictions # DataFrame 형식

    @staticmethod
    def submit(submission_path, prediction):
        submission = pd.read_csv(submission_path)
        submission['ECLO'] = prediction['prediction_label'].apply(lambda x: round(x))

        submission.to_csv(submission_path.replace("sample_submission", "submission"), index=False)



def main():
    submission_path = '../dataset/DaeguTrafficAccident/sample_submission.csv'

    pycaret_reg = PycaretRegression(
        train_path='../dataset/DaeguTrafficAccident/train_with_countrywide.csv',
        test_path='../dataset/DaeguTrafficAccident/test.csv'
    )
    pycaret_preds = pycaret_reg.predict()
    PycaretRegression.submit(submission_path, pycaret_preds)


if __name__ == "__main__":
    main()
