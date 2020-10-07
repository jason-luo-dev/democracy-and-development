# 1. scatterplot of democracy vs indicators from 1980 to 2017
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm
import scipy.stats as stats
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from matplotlib import colors as mcolors
import seaborn as sns
from statsmodels.tsa.ar_model import AutoReg
import re
# sns.set(style="darkgrid")
plt.style.use('seaborn-darkgrid')
# plt.style.use('fivethirtyeight')
pd.set_option('display.max_rows', None)

connection = psycopg2.connect(
    host="econvalues.cv6icouvimoi.us-east-1.rds.amazonaws.com",
    database="postgres",
    user="postgres",
    password="rocduyPzTUfLeXEHwUhj")

# interpolate indicator data
df = pd.read_sql_query("""select * from indicators_a where year >= 1980 and year <= 2017""", con=connection)

final = pd.DataFrame()
pct_indicators = ['employ_pct','energy_pct','internet_pct','uni_pct','urban_pct']

country = df.groupby('country')
for c in country.groups:
    group_country = country.get_group(c)

    indicator = group_country.groupby('indicator_name')

    for i in indicator.groups:
        group_indicator = indicator.get_group(i)
        group_indicator = group_indicator.set_index('year')

        filled = group_indicator['value'].interpolate(method='spline', order=1, limit_direction='both')

        data = pd.DataFrame(filled)
        data[data < 0] = 0

        if i in pct_indicators:
            data[data > 100] = 100

        data['indicator'] = i
        data['country'] = c
        data.reset_index(level=0, inplace=True)
        final = final.append(data, ignore_index=True)

final['percentile'] = final.groupby('indicator')['value'].rank(pct=True)
final['percentile'] = final['percentile'] * 100

# pivot indicator data for analysis

df_pivoted = pd.pivot_table(final, values=['value','percentile'], index=['country','year'], columns = 'indicator').reset_index()

df_pivoted.columns = df_pivoted.columns.droplevel()

df_pivoted.columns = ['country', 'year',
                      'p_employ_pct', 'p_energy_pct', 'p_gdp_usd', 'p_gni_usd',
                      'p_internet_pct', 'p_life_yrs', 'p_uni_pct', 'p_urban_pct',
                      'employ_pct', 'energy_pct', 'gdp_usd', 'gni_usd',
                      'internet_pct', 'life_yrs', 'uni_pct', 'urban_pct']

combined = df_pivoted[['p_energy_pct','p_uni_pct','p_urban_pct','p_employ_pct','p_gdp_usd','p_life_yrs']]

df_pivoted['Economy Score'] = combined.mean(axis=1)

# get democracy data

df_democracy = pd.read_sql_query("""select * from democracy_a where year >= 1980 and year <= 2017""", con=connection)

dem_final = pd.DataFrame()

df_country = df_democracy.groupby('country')

for d in df_country.groups:
    group_country = df_country.get_group(d)
    group_country = group_country.set_index('year')

    filled = group_country['value'].interpolate(method='spline', order=1, limit_direction='both')

    data = pd.DataFrame(filled)

    data[data > 1] = 1

    data['country'] = d
    data.reset_index(level=0, inplace=True)
    dem_final = dem_final.append(data, ignore_index=True)

dem_final['Democracy Score'] = dem_final['value'].rank(pct=True)
dem_final['Democracy Score'] = dem_final['Democracy Score'] * 100

# join the two dataframes
result = pd.concat([df_pivoted, dem_final], axis=1, join='inner')
result = result.loc[:,~result.columns.duplicated()]

# 2. autoregression to 2030

result = result[['country','year','Democracy Score','Economy Score']]

ar_final = pd.DataFrame()

ar_country = result.groupby('country')

for d in ar_country.groups:
    group_country = ar_country.get_group(d)

    ar_dem = group_country[['Democracy Score']].values
    model = AutoReg(ar_dem, lags=1)
    model_fit = model.fit()
    # make prediction
    yhat = model_fit.predict(len(ar_dem), len(ar_dem) + 12)

    projected = pd.DataFrame(yhat, columns=['Democracy Score'])

    projected['country'] = d
    projected.reset_index(level=0, inplace=True)
    projected['year'] = projected['index'] + 2018

    ar_econ = group_country[['Economy Score']].values
    model2 = AutoReg(ar_econ, lags=1)
    model_fit2 = model2.fit()
    # make prediction
    yhat2 = model_fit2.predict(len(ar_econ), len(ar_econ) + 12)

    projected2 = pd.DataFrame(yhat2, columns=['Economy Score'])
    projected2.reset_index(level=0, inplace=True)

    projected = pd.concat([projected, projected2], axis=1)
    projected = projected[['country','year','Democracy Score','Economy Score']]

    projected.loc[(projected['Democracy Score'] < 0) | (projected['Democracy Score'] > 100), 'Democracy Score'] = 0

    ar_final = ar_final.append(projected, ignore_index=True)

result = result.append(ar_final)

result = result.loc[result['year'] == 2017]

# result[['country','Democracy Score','Economy Score']].to_csv('countries_2030.csv')


#----------elbow test----------
# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, n_init=100, init='k-means++')
#     kmeans.fit(X[:,[1,2]])
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()


#----------kmeans clustering----------
X = result[['country','Democracy Score','Economy Score']].values

kmeans = KMeans(n_clusters=4, n_init=100, init='k-means++')
kmeans.fit(X[:,[1,2]])

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

colors = 10*['go','co','bo','ro','ko','yo','mo','wo']

for i in range(len(X)):
    plt.plot(X[i][1], X[i][2],
             colors[labels[i]],
             markersize=10,
             alpha=0.4,
             markeredgecolor='black',
             linewidth=2)

# plt.scatter(centroids[:,0], centroids[:,1], marker="x", color='crimson', s=150, linewidths=5, zorder=-1)

for i, txt in enumerate(X[:,[0]]):
    plt.annotate(' '.join(map(str, txt)), (X[i][1], X[i][2]), size=11, xytext=(7, 7),  textcoords="offset points")

plt.xlabel('Democracy Score', fontsize=12)
plt.ylabel('Economy Score', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.axis([0, 100, 0, 100])

plt.show()


# ----------scatterplots with distributions----------
# g = sns.jointplot("Democracy Score", "Economy Score", data=result,
#                   kind="reg", truncate=False,
#                   xlim=(0, 100), ylim=(0, 100),
#                   color="darkolivegreen", scatter_kws={"s": 110},
#                   height=7, stat_func=pearsonr)
#
# plt.xlabel("Democracy Score", fontsize=12)
# plt.ylabel("Economy Score", fontsize=12)
# plt.tick_params(axis="both", labelsize=12)
# plt.show()