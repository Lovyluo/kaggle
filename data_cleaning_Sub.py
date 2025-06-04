# data_cleaning.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

def preprocess_data(df, label_encoders=None, scaler=None, is_train=True):
    # ———— 0. 保留一份原始特征（方便 Error Rate 分析） ————
    df_orig = df.copy()
    
    # ———— 1. Cabin Flag ————
    df['HasCabin'] = df['Cabin'].notnull().astype(int)
    df = df.drop(columns=['Cabin'])
    
    # ———— 2. 基本缺失值填充 ————
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    if is_train:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # ———— 3. Age 回归 + 中位数回退 ————
    missing_age = df['Age'].isnull()
    if missing_age.any():
        tmp = df[['Age','Pclass','SibSp','Parch','Fare','Embarked','Sex']].copy()
        tmp = pd.get_dummies(tmp, columns=['Embarked','Sex'], dummy_na=True)
        known = tmp[tmp['Age'].notnull()]
        unknown = tmp[tmp['Age'].isnull()].drop(columns=['Age'])
        rfr = RandomForestRegressor(n_estimators=100, random_state=42)
        rfr.fit(known.drop(columns=['Age']), known['Age'])
        df.loc[missing_age, 'Age'] = rfr.predict(unknown)
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # ———— 4. 家庭特征 ————
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone']    = (df['FamilySize'] == 1).astype(int)
    # 4.1 家庭规模细分
    
    df['FamilyBucket'] = df['FamilySize'].map(
        lambda s: 'Alone' if s == 1
                  else ('Two' if s == 2
                  else ('ThreeToFour' if 3 <= s <= 4 else 'Large'))
    )
    
    # ———— 5. 针对高误差群体 ————
    # Plass 1 & Male、Plass 3 & Female
    df['IsPclass1Male']   = ((df['Pclass']==1) & (df['Sex']=='male')  ).astype(int)
    df['IsPclass3Female'] = ((df['Pclass']==3) & (df['Sex']=='female')).astype(int)

    


    # 票号前缀（提取票号字母/数字前缀）
    df['TicketPrefix'] = df['Ticket'].str.split().str[0].str.replace(r'[^A-Za-z]', '', regex=True).replace('', 'NUM')

    # ———— 6. 头衔细化 ————
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    # 主要头衔
    keep = ['Mr','Mrs','Miss','Master','Dr','Rev','Officer']
    df['Title'] = df['Title'].apply(lambda t: t if t in keep else 'Rare')
    
    # ———— 7. 年龄分箱 ————
    df['AgeBucket'] = pd.cut(
        df['Age'],
        bins=[0,5,12,20,35,60, df['Age'].max()+1],
        labels=['Baby','Child','Teen','YA','Adult','Senior']
    )
    
    # ———— 8. 票价对数 ————
    df['LogFare'] = np.log1p(df['Fare'])
    
    # ———— 9. 类别编码 ————
    cat_cols = ['Sex','Embarked','Title','AgeBucket','FamilyBucket']
    if label_encoders is None:
        label_encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    else:
        for col in cat_cols:
            le = label_encoders[col]
            known = set(le.classes_)
            default = le.classes_[0]
            df[col] = df[col].astype(str).apply(lambda x: x if x in known else default)
            df[col] = le.transform(df[col])
    
    # Plass 3 & Female & Alone
    df['P3_Female_Alone'] = ((df.Pclass==3) & (df.Sex=='female') & (df.IsAlone==1)).astype(int)

    # Plass 1 & Male & Fare（Fare > middle）
    fare_med = df.Fare.median()
    df['P1_Male_HighFare'] = ((df.Pclass==1) & (df.Sex=='male') & (df.Fare>fare_med)).astype(int)

    # Mrs & Parch > 0
    df['MotherWithChild'] = ((df.Title=='Mrs') & (df.Parch>0)).astype(int)

    df['Mrs_Adult']  = ((df.Title=='Mrs') & (df.Age>=20) & (df.Age<60)).astype(int)
    df['Mrs_Senior'] = ((df.Title=='Mrs') & (df.Age>=60)).astype(int)

    # ———— 10. 构建特征矩阵 & 标准化 ————
    features = [
        'Pclass','Sex','Age','LogFare','Embarked',
        'FamilySize','IsAlone','FamilyBucket','Title',
        'AgeBucket','HasCabin','IsPclass1Male','IsPclass3Female'
    ]
    X = df[features].copy()
    
    num_cols = ['Age','LogFare']
    if scaler is None:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
    else:
        X[num_cols] = scaler.transform(X[num_cols])
    
    return X, label_encoders, scaler