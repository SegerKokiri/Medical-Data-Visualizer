import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#On Problem 5
# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 3
df['cholesterol'] = np.where(df['cholesterol'] > 1, 1, 0)
df['gluc'] = np.where(df['gluc'] > 1, 1, 0)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'], var_name='variable', value_name='value')
    # 6
    df_cat = df_cat.groupby(['cardio', 'variable'])['value'].value_counts().reset_index(name='total')

    # 7
    # sns.catplot(x='variable', y='value', hue='cardio', data=df_cat, kind='count')


    # 8
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar').fig

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))

    ]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # 14
    fig, ax = plt.subplots(figsize=(8,6))

    # 15
    ax = sns.heatmap(corr, annot=True, mask=mask, ax=ax, fmt='.1f', linewidths=.5, annot_kws={'size': 8})
    # plt.show()

    # 16
    fig.savefig('heatmap.png')
    return fig

