import numpy as np
import pandas as pd
from deslib.static.epfh import EnsemblePruneFH
from deslib.static.des_fh import DESFH
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
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
        df.attrs['source'] = name
        dataframes.append(df)
    return dataframes


urls = [
    ('https://github.com/ikedss/EnsemblePruneFH/blob/master/Bases/pima.csv?raw=true', 'pima'),
    ('https://github.com/ikedss/EnsemblePruneFH/blob/master/Bases/breast%2Bcancer%2Bwisconsin%2Bdiagnostic/wdbc.csv?raw=true', 'wdbc'),
    ('https://github.com/ikedss/EnsemblePruneFH/blob/master/Bases/connectionist%2Bbench%2Bsonar%2Bmines%2Bvs%2Brocks/sonar%20data.csv?raw=true', 'sonar'),
    ('https://github.com/ikedss/EnsemblePruneFH/blob/master/Bases/blood%2Btransfusion%2Bservice%2Bcenter/transfusion.data?raw=true', 'transfusion'),
    ('https://github.com/ikedss/EnsemblePruneFH/blob/master/Bases/monk%2Bs%2Bproblems/monk.csv?raw=true', 'monk'),
    ('https://github.com/ikedss/EnsemblePruneFH/blob/master/Bases/adult/adult.csv?raw=true', 'adult'),
    ('https://github.com/ikedss/EnsemblePruneFH/blob/master/Bases/cardiotocography/CTG.csv?raw=true', 'cardiotocography'),
    ('https://github.com/ikedss/EnsemblePruneFH/blob/master/Bases/dermatology/derm.csv?raw=true', 'dermatology'),
    ('https://github.com/ikedss/EnsemblePruneFH/blob/master/Bases/glass%2Bidentification/glass.csv?raw=true', 'glass'),
    ('https://github.com/ikedss/EnsemblePruneFH/blob/master/Bases/haberman%2Bs%2Bsurvival/haberman.csv?raw=true', 'haberman'),
    ('https://github.com/ikedss/EnsemblePruneFH/blob/master/Bases/ilpd%2Bindian%2Bliver%2Bpatient%2Bdataset/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv?raw=true', 'ilpd'),
    ('https://github.com/ikedss/EnsemblePruneFH/blob/master/Bases/ionosphere/ionosphere.csv?raw=true', 'ionosphere'),
    ('https://github.com/ikedss/EnsemblePruneFH/blob/master/Bases/magic%2Bgamma%2Btelescope/telescope_data.csv?raw=true', 'telescope'),
    ('https://github.com/ikedss/EnsemblePruneFH/blob/master/Bases/mammographic%2Bmass/mammographic_mass.csv?raw=true', 'mammographic_mass'),
    ('https://github.com/ikedss/EnsemblePruneFH/blob/master/Bases/statlog%2Bgerman%2Bcredit%2Bdata/german_credit_data.csv?raw=true', 'german_credit'),
    ('https://github.com/ikedss/EnsemblePruneFH/blob/master/Bases/statlog%2Bheart/statlog_heart.csv?raw=true', 'heart'),
    ('https://github.com/ikedss/EnsemblePruneFH/blob/master/Bases/statlog%2Blandsat%2Bsatellite/Sat.csv?raw=true', 'satellite'),
    ('https://github.com/ikedss/EnsemblePruneFH/blob/master/Bases/statlog%2Bvehicle%2Bsilhouettes/vehicle.csv?raw=true', 'vehicle'),
    ('https://github.com/ikedss/EnsemblePruneFH/blob/master/Bases/vertebral%2Bcolumn/column_3C.csv?raw=true', 'vertebral_column'),
    ('https://github.com/ikedss/EnsemblePruneFH/blob/master/Bases/wine/wine.csv?raw=true', 'wine')
]

dataframes = load_and_clean_data(urls)


def select_x_y(df):
    target_column = None
    for col in df.columns:
        if col.lower() in ['outcome', 'diagnosis', '60', 'donated_blood_in_march_2007', 'class', 'income', 'site', 'type', 'status', 'is_patient', 'column_ai', 'severity', 'risk', 'presence', 'label', 'target', 'wine']:
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

        classifiers = {
            'BaggingClassifier': BaggingClassifier(n_estimators=10, random_state=rng),
            'RandomForestClassifier': RandomForestClassifier(n_estimators=10, random_state=rng),
            'AdaBoostClassifier': AdaBoostClassifier(n_estimators=10, random_state=rng)
        }

        for name, clf in classifiers.items():
            startClassifier = time.time()
            clf.fit(X_train, y_train)
            endClassifier = time.time()

            des = DESFH(pool_classifiers=clf, random_state=rng)
            startDESFH = time.time()
            des.fit(X_val, y_val)
            endDESFH = time.time()

            fh = EnsemblePruneFH(pool_classifiers=clf, random_state=rng, overlap_threshold=0.01, threshold_remove=0.01)
            startPrune = time.time()
            fh.fit(X_val, y_val)
            endPrune = time.time()

            results.append({
                'Run': seed_list.index(random_seed) + 1,
                'Dataset': data.attrs['source'],
                'Classifier': name,
                'Classifier_accuracy': round(clf.score(X_test, y_test), 5),
                'Classifier_time': round(endClassifier - startClassifier, 5),
                'DESFH_accuracy': round(des.score(X_test, y_test), 5),
                'DESFH_time': round(endDESFH - startDESFH, 5),
                'EnsemblePruneFH_accuracy': round(fh.score(X_test, y_test), 5),
                'EnsemblePruneFH_time': round(endPrune - startPrune, 5)
            })
    return results


seed_list = [42, 78, 43, 81, 13, 73, 23, 64, 9, 54]
all_results = []
for seed in seed_list:
    results = process_datasets(dataframes, random_seed=seed)
    all_results.extend(results)

df_results = pd.DataFrame(all_results)
df_results.to_csv('results.csv', index=False)
