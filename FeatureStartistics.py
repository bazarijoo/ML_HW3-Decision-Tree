import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def show_statistics(data):
    # fig, axes = plt.subplots(ncols=3, nrows=5)

    plt.xlabel('age')
    plt.ylabel('probability')
    sns.distplot(data['age'], hist_kws={'edgecolor': 'black'})
    plt.show()

    plt.xlabel('sex')
    plt.ylabel('counts')
    sns.countplot(x='sex', data=data, palette="bwr")
    plt.show()

    plt.xlabel('cp')
    plt.ylabel('counts')
    sns.countplot(x='cp', data=data, palette="bwr")
    plt.show()

    plt.xlabel('trestbps')
    plt.ylabel('probability')
    sns.distplot(data['trestbps'], hist_kws={'edgecolor': 'black'})
    plt.show()

    plt.xlabel('chol')
    plt.ylabel('probability')
    sns.distplot(data['chol'], hist_kws={'edgecolor': 'black'})
    plt.show()

    plt.xlabel('fbs')
    plt.ylabel('counts')
    sns.countplot(x='fbs', data=data, palette="bwr")
    plt.show()

    plt.xlabel('restecg')
    plt.ylabel('counts')
    sns.countplot(x='restecg', data=data, palette=sns.color_palette("coolwarm",5))
    plt.show()

    plt.xlabel('thalach')
    plt.ylabel('probability')
    sns.distplot(data['thalach'], hist_kws={'edgecolor': 'black'})
    plt.show()

    plt.xlabel('exang')
    plt.ylabel('counts')
    sns.countplot(x='exang', data=data, palette="bwr")
    plt.show()

    plt.xlabel('oldpeak')
    plt.ylabel('probability')
    sns.distplot(data['oldpeak'], hist_kws={'edgecolor': 'black'})
    plt.show()

    plt.xlabel('slope')
    plt.ylabel('counts')
    sns.countplot(x='slope', data=data, palette=sns.color_palette("coolwarm",3))
    plt.show()

    plt.xlabel('ca')
    plt.ylabel('counts')
    sns.countplot(x='ca', data=data, palette="bwr")
    plt.show()

    plt.xlabel('thal')
    plt.ylabel('counts')
    sns.countplot(x='thal', data=data, palette="bwr")
    plt.show()

    plt.xlabel('num')
    plt.ylabel('counts')
    sns.countplot(x='num', data=data, palette=sns.color_palette("coolwarm",5))
    plt.show()

    # fig.tight_layout()
    # plt.show()


data = pd.read_csv("processed.cleveland.data", header=None,
                    names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs','restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal','num'], na_values='?') #(num is the predicted attribute)

# handling missing values
data['thal'] = data['thal'].fillna(value=data['thal'].median())
data['ca'] = data['ca'].fillna(value=data['ca'].median())

show_statistics(data)