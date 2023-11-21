# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')


# %%
df = pd.read_csv('data_3years.csv')
df = df.set_index('Fecha')
df.index = pd.to_datetime(df.index)

# %%
df.plot(style='-',
        figsize=(20, 5),
        color=color_pal[0],
        title='Ventas ')

plt.show()

# %% [markdown]
# # Train / Test Split

# %%
train = df.loc[df.index < '01-01-2014']
test = df.loc[df.index >= '01-01-2014']

fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label='Training Set', title='Data Train/Test Split')
test.plot(ax=ax, label='Test Set')
ax.axvline('01-01-2014', color='black', ls='--')
ax.legend(['Training Set', 'Test Set'])
plt.show()

# %%
df.loc[(df.index > '01-01-2016') & (df.index < '01-08-2016')] \
    .plot(figsize=(15, 5), title='Week Of Data')
plt.show()

# %% [markdown]
# # Feature Creation

# %%
def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

df = create_features(df)

# %% [markdown]
# # Visualize our Feature / Target Relationship

# %%
fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='dayofweek', y='Venta')
ax.set_title('Venta por semana')
plt.show()

# %%
fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='month', y='Venta', palette='Blues')
ax.set_title('Ventas por mes')
plt.show()

# %% [markdown]
# # Create our Model

df.columns

# %%
train = create_features(train)
test = create_features(test)

FEATURES = ['dayofyear', 'dayofweek', 'quarter', 'month', 'year']
TARGET = 'Venta'

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

# %%
reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                       n_estimators=500,
                       early_stopping_rounds=50,
                       objective='reg:linear',
                       max_depth=3,
                       learning_rate=0.01)
reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100)

# %% [markdown]
# # Feature Importance

# %%
fi = pd.DataFrame(data=reg.feature_importances_,
             index=reg.feature_names_in_,
             columns=['importance'])
fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
plt.show()

# %% [markdown]
# # Forecast on Test

# %%
test['prediction'] = reg.predict(X_test)
df = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)
ax = df[['Venta']].plot(figsize=(15, 5))
df['prediction'].plot(ax=ax, style='-')
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Raw Dat and Prediction')
plt.show()

# %%
ax = df.loc[(df.index > '02-01-2014') & (df.index < '03-01-2014')]['Venta'] \
    .plot(figsize=(15, 5), title='Month Of Data')
df.loc[(df.index > '02-01-2014') & (df.index < '03-01-2014')]['prediction'] \
    .plot(style='-')
plt.legend(['Truth Data','Prediction'])
plt.show()

# %% [markdown]
# # Score (RMSE)

# %%
score = np.sqrt(mean_squared_error(test['Venta'], test['prediction']))
mape = mean_absolute_percentage_error(test['Venta'], test['prediction'])

print("Error: "f"{mape:.0%}")
print(f'RMSE Score on Test set: {score:0.2f}')

# %% [markdown]
# # Calculate Error
# - Look at the worst and best predicted days

# %%
test['error'] = np.abs(test[TARGET] - test['prediction'])
test['date'] = test.index.date
test.groupby(['date'])['error'].mean().sort_values(ascending=False).head(10)

# %%
