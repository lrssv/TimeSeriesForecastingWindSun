import pmdarima as pm
from pmdarima.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#obtem os dados
url1 = 'https://raw.githubusercontent.com/lrssv/TimeSeriesForecastingWindSun/master/series_ventovel_pampulha_2018_2019'
url2 = 'https://raw.githubusercontent.com/lrssv/TimeSeriesForecastingWindSun/master/series_radiacao_pampulha_2018_2019'

df_ventovel = pd.read_csv(url1, header=0, parse_dates=[0], index_col=0, squeeze=True)
df_radiacao = pd.read_csv(url2, header=0, parse_dates=[0], index_col=0, squeeze=True)

series_ventovel = pd.Series(df_ventovel)
series_radiacao = pd.Series(df_radiacao)

# Load/split your data
train = series_radiacao.loc['2018-01-01 01:00:00':'2019-12-29 01:00:00']
test = series_radiacao.loc['2019-12-29 01:00:00':] 

# Fit your model
model = pm.auto_arima(train, seasonal=False, m=1)

# make your forecasts
forecasts = model.predict(test.shape[0])  # predict N steps into the future

# Visualize the forecasts (blue=train, green=forecasts)

print(forecasts)
#x = np.arange(y.shape[0])
#plt.plot(x[:150], train, c='blue')
#plt.plot(x[150:], forecasts, c='green')
#plt.show()