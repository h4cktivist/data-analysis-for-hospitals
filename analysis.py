import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt


def create_patients_df(hospital, prenatal, sports):
    prenatal.columns, sports.columns = hospital.columns, hospital.columns
    patiens = pd.concat([hospital, prenatal, sports], ignore_index=True)

    patiens.drop(columns=['Unnamed: 0'], inplace=True)
    patiens.dropna(axis=0, how='all', inplace=True)

    patiens['gender'].replace(
        ['female', 'male', 'man', 'woman'],
        ['f', 'm', 'm', 'f'],
        inplace=True
    )

    patiens['gender'].replace(np.nan, 'f', inplace=True)

    columns = ['bmi', 'diagnosis', 'blood_test', 'ecg', 'ultrasound', 'mri', 'xray', 'children', 'months']
    for c in columns:
        patiens[c] = patiens[c].replace(np.nan, 0)

    return patiens


def show_df_sample(patiens):
    print(patiens.sample(n=len(patiens)))


def visualize_data(patiens):
    patiens.plot(title='Distribution of patients by age.', y='age', kind='hist', bins=range(0, 80))
    plt.show()

    patiens.diagnosis.value_counts().plot(title='Distribution of patients by diagnosis.', kind='pie')
    plt.show()

    ax = plt.subplots()[1]
    ax.set_title('Distribution of patients by height.')
    plt.violinplot(patiens['height'])
    plt.show()


if __name__ == '__main__':
    matplotlib.use('TkAgg')

    pd.set_option('display.max_columns', 8)

    hospital = pd.read_csv('test/general.csv')
    prenatal = pd.read_csv('test/prenatal.csv')
    sports = pd.read_csv('test/sports.csv')

    patiens = create_patients_df(hospital, prenatal, sports)
    show_df_sample(patiens)
    visualize_data(patiens)
