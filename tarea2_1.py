import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('housing.csv')
df.dropna(inplace=True)

##Asignamos los limites
df = df[df['median_house_value'] < 500000]
df = df[df['housing_median_age'] < 52]
df = df[df['median_income'] < 15]

##Dummies de ocean_proximity
df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)
df['coastal_premium'] = df.get('ocean_proximity_NEAR OCEAN',0)*1.2 + df.get('ocean_proximity_<1H OCEAN',0)*0.6 + df.get('ocean_proximity_NEAR BAY',0)*0.9

##Ratios, polinomios para nuevas caracterisiticas y transformaciones para aumentar la precision
df['log_median_income'] = np.log1p(df['median_income'])
df['log_population'] = np.log1p(df['population'])
df['rooms_per_household'] = df['total_rooms'] / df['households']
df['population_per_household'] = df['population'] / df['households']
df['bedroom_ratio'] = df['total_bedrooms'] / df['total_rooms']
df['lat_sin'] = np.sin(np.radians(df['latitude']))
df['lon_cos'] = np.cos(np.radians(df['longitude']))
df['income_sq'] = df['median_income'] ** 2
df['income_cub'] = df['median_income'] ** 3
df['age_sq'] = df['housing_median_age'] ** 2
df['rooms_sq'] = df['rooms_per_household'] ** 2
df['pop_sq'] = df['population_per_household'] ** 2

##Combinacion de caracteristicas claves
df['income_rooms'] = df['median_income'] * df['rooms_per_household']
df['income_pop'] = df['median_income'] * df['population_per_household']
df['income_age'] = df['median_income'] * df['housing_median_age']
df['rooms_pop'] = df['rooms_per_household'] * df['population_per_household']
df['rooms_age'] = df['rooms_per_household'] * df['housing_median_age']
df['pop_age'] = df['population_per_household'] * df['housing_median_age']
df['income_coastal'] = df['median_income'] * df['coastal_premium']
df['income_lat'] = df['median_income'] * df['lat_sin']
df['income_lon'] = df['median_income'] * df['lon_cos']

##Caracterisitcas economicas para mejorar la prediccion
df['wealth_density'] = df['median_income'] / (df['population_per_household'] + 1)
df['space_quality'] = df['rooms_per_household'] / (df['housing_median_age'] + 1)
df['investment_score'] = df['median_income'] * df['rooms_per_household'] / (df['housing_median_age'] + 1)
df['income_rank'] = df['median_income'].rank(pct=True)
df['rooms_rank'] = df['rooms_per_household'].rank(pct=True)
df['age_rank'] = df['housing_median_age'].rank(pct=True)
df['income_rooms_rank'] = df['income_rank'] * df['rooms_rank']
df['quality_location'] = df['income_rank'] * (1 - df['age_rank'])

##Dummies
df['high_income'] = (df['median_income'] > df['median_income'].quantile(0.8)).astype(int)
df['spacious_home'] = (df['rooms_per_household'] > df['rooms_per_household'].quantile(0.75)).astype(int)
df['new_home'] = (df['housing_median_age'] <= 10).astype(int)
df['luxury_spacious'] = df['high_income'] * df['spacious_home']
df['new_luxury'] = df['high_income'] * df['new_home']

##Características por ubicacion geográfica
df['distance_wealth'] = df['median_income'] / (np.abs(df['longitude'] + 118) + 1)
df['lat_wealth'] = df['median_income'] * np.abs(df['latitude'] - 34)
df['future_value'] = df['median_income'] * df['rooms_per_household'] / (df['housing_median_age'] + 1)
df['investment_momentum'] = df['median_income'] * (1 / (df['population_per_household'] + 1)) * (1 / (df['housing_median_age'] + 1))

##Características de percentiles 
df['top_quartile_income'] = (df['median_income'] > df['median_income'].quantile(0.75)).astype(int)
df['top_quartile_rooms'] = (df['rooms_per_household'] > df['rooms_per_household'].quantile(0.75)).astype(int)
df['luxury_space_combo'] = df['top_quartile_income'] * df['top_quartile_rooms']

##Características de valor relativo
df['income_relative'] = df['median_income'] / df['median_income'].median()
df['rooms_relative'] = df['rooms_per_household'] / df['rooms_per_household'].median()
df['master_desirability'] = (df['income_rank'] * 0.4 + df['rooms_rank'] * 0.3 + (1-df['age_rank']) * 0.3) * (1 + df['coastal_premium'])

##Dummies
df['income_near_ocean'] = df['median_income'] * df.get('ocean_proximity_NEAR OCEAN', 0)
df['income_near_bay'] = df['median_income'] * df.get('ocean_proximity_NEAR BAY', 0)

df['income_tier'] = pd.cut(df['median_income'], bins=5, labels=False)
df['premium_income_tier'] = (df['income_tier'] >= 3).astype(int)
df['bedrooms_per_person'] = df['total_bedrooms'] / df['population']
df['sqrt_median_income'] = np.sqrt(df['median_income'])
df['income_power_15'] = df['median_income'] ** 1.5

##Cambios adicionales para mejorar la precisión
df['log_income_rooms'] = df['log_median_income'] * df['rooms_per_household']
df['income_sq_coastal'] = df['income_sq'] * df['coastal_premium']
df['wealth_space_interaction'] = df['wealth_density'] * df['space_quality']
df['income_investment_interaction'] = df['median_income'] * df['investment_score']
df['ultra_luxury'] = (df['median_income'] > df['median_income'].quantile(0.9)) * (df['rooms_per_household'] > df['rooms_per_household'].quantile(0.9))
df['prime_location'] = (df['coastal_premium'] > 0) * (df['median_income'] > df['median_income'].median())
df['economic_efficiency'] = df['median_income'] / (df['population_per_household'] + df['housing_median_age'] + 1)

##Características geográficas 
df['distance_to_sf'] = np.sqrt((df['latitude'] - 37.7749)**2 + (df['longitude'] + 122.4194)**2)
df['distance_to_la'] = np.sqrt((df['latitude'] - 34.0522)**2 + (df['longitude'] + 118.2437)**2)
df['min_city_distance'] = np.minimum(df['distance_to_sf'], df['distance_to_la'])
df['income_distance_ratio'] = df['median_income'] / (df['min_city_distance'] + 1)
df['luxury_index'] = df['median_income'] * df['rooms_per_household'] * df['coastal_premium']
df['value_score'] = df['median_income'] * df['rooms_per_household'] / (df['housing_median_age'] + 1)
df['location_value'] = df['median_income'] * (1 + df['coastal_premium']) / (df['population_per_household'] + 1)

# Características adicionales simples para llegar a 70%
df['income_rooms_age'] = df['median_income'] * df['rooms_per_household'] * (1 / (df['housing_median_age'] + 1))
df['coastal_wealth'] = df['median_income'] * df['coastal_premium'] * df['rooms_per_household']
df['efficiency_score'] = df['median_income'] / (df['population_per_household'] * df['housing_median_age'] + 1)
df['super_high_income'] = (df['median_income'] > df['median_income'].quantile(0.9)).astype(int)
df['premium_rooms'] = (df['rooms_per_household'] > df['rooms_per_household'].quantile(0.85)).astype(int)
df['elite_combo'] = df['super_high_income'] * df['premium_rooms'] * (df['coastal_premium'] > 0).astype(int)
df['log_total_rooms'] = np.log1p(df['total_rooms'])
df['log_households'] = np.log1p(df['households'])
df['log_wealth_density'] = np.log1p(df['wealth_density'])

##logaritmos
df['log_income_log_rooms'] = df['log_median_income'] * df['log_total_rooms']
df['log_income_age'] = df['log_median_income'] * df['housing_median_age']
df['income_cubed'] = df['median_income'] ** 3
df['rooms_cubed'] = df['rooms_per_household'] ** 3
df['age_inverse'] = 1 / (df['housing_median_age'] + 1)
df['market_score'] = df['median_income'] ** 2 * df['rooms_per_household'] * df['coastal_premium']
df['desirability_premium'] = df['income_rank'] * df['rooms_rank'] * (1 + df['coastal_premium'])
df['investment_grade'] = df['median_income'] * df['rooms_per_household'] * df['age_inverse']
df['bay_area_premium'] = (df['latitude'] > 37.2) * (df['longitude'] > -122.8) * df['median_income']
df['socal_premium'] = (df['latitude'] < 35) * df['coastal_premium'] * df['median_income']
df['income_per_room'] = df['median_income'] / (df['rooms_per_household'] + 1)
df['wealth_per_person'] = df['median_income'] / (df['population_per_household'] + 1)
df['space_premium'] = df['rooms_per_household'] / (df['population_per_household'] + 1)

##Asignar las etiquetas 
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

##Separar datos
df_train, df_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

##Escalado estándar
scaler = StandardScaler()
X_train = scaler.fit_transform(df_train)
X_test = scaler.transform(df_test)

##Modelo lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train)

##Evaluación
print(f"Score entrenamiento: {modelo.score(X_train, y_train):.4f}")
print(f"Score prueba: {modelo.score(X_test, y_test):.4f}")
