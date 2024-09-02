import numpy as np
import pandas as pd
from deslib.static.epfh import EnsemblePruneFH
from deslib.static.des_fh import DESFH
from deslib.static.static_selection import StaticSelection
from deslib.static.stacked import StackedClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
import time


def clean_data(df):
    df = df.fillna(method='ffill')
    df = df.drop_duplicates()
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df


def load_and_clean_data(urls):
    dataframes = []
    for url, name in urls:
        df = pd.read_csv(url)
        df = clean_data(df)
        df.attrs['source'] = name  # Store the dataset name in DataFrame attributes
        dataframes.append(df)
    return dataframes


urls = [
    ('https://github.com/ikedss/EnsemblePruneFH/blob/master/Bases/pima.csv?raw=true', 'pima'),
    ('https://github.com/ikedss/EnsemblePruneFH/blob/master/Bases/breast%2Bcancer%2Bwisconsin%2Bdiagnostic/wdbc.csv?raw=true', 'wdbc'),
    ('https://github.com/ikedss/EnsemblePruneFH/blob/master/Bases/connectionist%2Bbench%2Bsonar%2Bmines%2Bvs%2Brocks/sonar%20data.csv?raw=true', 'sonar'),
    ('https://github.com/ikedss/EnsemblePruneFH/blob/master/Bases/blood%2Btransfusion%2Bservice%2Bcenter/transfusion.data?raw=true', 'transfusion'),
    ('https://github.com/ikedss/EnsemblePruneFH/blob/master/Bases/monk%2Bs%2Bproblems/monk.csv?raw=true', 'monk')
]

dataframes = load_and_clean_data(urls)


def select_x_y(df):
    target_column = None
    for col in df.columns:
        if col.lower() in ['outcome', 'diagnosis', '60', 'donated_blood_in_march_2007', 'class']:
            target_column = col
            break
    if target_column is None:
        raise ValueError('No target column found')

    X = df.drop(columns=[target_column]).values
    y = df[target_column].astype('category').cat.codes.values
    return X, y


def process_datasets(dataframes, random_seed):
    results = []
    rng = np.random.RandomState(random_seed)
    for data in dataframes:
        X, y = select_x_y(data)

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=rng, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=rng, stratify=y_temp)

        classifiers = BaggingClassifier(n_estimators=50, random_state=rng)
        startBaggingClassifier = time.time()
        classifiers.fit(X_train, y_train)
        endBaggingClassifier = time.time()

        ss = StaticSelection(pool_classifiers=classifiers, random_state=rng)
        startStaticSelection = time.time()
        ss.fit(X_val, y_val)
        endStaticSelection = time.time()

        des = DESFH(pool_classifiers=classifiers, random_state=rng)
        startDESFH = time.time()
        des.fit(X_val, y_val)
        endDESFH = time.time()

        fh = EnsemblePruneFH(pool_classifiers=classifiers, random_state=rng, overlap_threshold=0, threshold_remove=0.1)
        startPrune = time.time()
        fh.fit(X_val, y_val)
        endPrune = time.time()

        results.append({
            'dataset': data.attrs['source'],
            'BaggingClassifier_accuracy': classifiers.score(X_test, y_test),
            'BaggingClassifier_time': endBaggingClassifier - startBaggingClassifier,
            'StaticSelection_accuracy': ss.score(X_test, y_test),
            'StaticSelection_time': endStaticSelection - startStaticSelection,
            'DESFH_accuracy': des.score(X_test, y_test),
            'DESFH_time': endDESFH - startDESFH,
            'EnsemblePruneFH_accuracy': fh.score(X_test, y_test),
            'EnsemblePruneFH_time': endPrune - startPrune
        })
    return results


seed_list = [42, 78, 43, 81, 13, 73, 23, 64, 9, 54]
all_results = []
for seed in seed_list:
    results = process_datasets(dataframes, random_seed=seed)
    all_results.append(results)

for run_idx, results in enumerate(all_results):
    print(f"Run {run_idx + 1}")
    for result in results:
        print(f"Results for dataset: {result['dataset']}")
        print(f"BaggingClassifier accuracy: {result['BaggingClassifier_accuracy']}")
        print(f"BaggingClassifier time: {result['BaggingClassifier_time']}")
        print(f"StaticSelection accuracy: {result['StaticSelection_accuracy']}")
        print(f"StaticSelection time: {result['StaticSelection_time']}")
        print(f"DESFH accuracy: {result['DESFH_accuracy']}")
        print(f"DESFH time: {result['DESFH_time']}")
        print(f"EnsemblePruneFH accuracy: {result['EnsemblePruneFH_accuracy']}")
        print(f"EnsemblePruneFH time: {result['EnsemblePruneFH_time']}")
