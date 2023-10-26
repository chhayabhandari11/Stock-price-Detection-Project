import requests
import pandas as pd
import datetime
import numpy as np
import pickle as pk
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

url = "https://api.tiingo.com/tiingo/daily/appl/prices?startDate=2015-01-02&token=424c723fd790ecb18d9ba7cd2e8834b6b5a7eaa7"
session = requests.Session()
retry = Retry(connect=3,backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

print(session.get(url))
# df = pd.DataFrame(response.json())

# class APPL_STOCK:

#     def string_to_date(date):
#         date = date[0:10]
#         d = date.split('-')
#         year , month , date = int(d[0]),int(d[1]),int(d[2])
#         return datetime.datetime(year,month,date)
    
#     def __init__(self):
#         headers = {
#             'Content-Type': 'application/json'
#         }
#         api = "https://api.tiingo.com/tiingo/daily/appl/prices?startDate=2015-01-02&token=424c723fd790ecb18d9ba7cd2e8834b6b5a7eaa7"
#         requestResponse = requests.get(api, headers=headers)
#         df = pd.DataFrame(requestResponse.json())
#         data = df[['date','close']]
#         data['date'] = data['date'].apply(string_to_date)
#         data.drop(0,axis=0,inplace=True)
#         t = data_to_window_data(data)
#         t = pd.DataFrame(t,columns=['date','day1','day2','day3','day4','day5','Target'])
#         filename = 'apple_stock_predictor.pkl'


#         data_np = t.to_numpy()
#         time,x,y = input_data(data_np)
#         split1 = int(len(x) * 0.8)
#         split2 = int(len(x) * 0.9)

#         time_train , x_train , y_train = time[:split1],x[:split1],y[:split1]
#         time_val , x_val , y_val = time[split1:split2],x[split1:split2],y[split1:split2]
#         time_test , x_test , y_test = time[split2:],x[split2:],y[split2:]

#         # model = define_model()
#         # model.fit(x_train,y_train,validation_data = (x_val,y_val) , epochs = 50)
#         # with open(filename, 'wb') as file:
#         #     pk.dump(model, file)
        
#     def forecast_prices(model,x_test,time_test):
#         forecast = x_test[-1:][0]
#         forecast = forecast.reshape(5,)
#         dates = []
#         dates.append(time_test[len(time_test)-1])
#         for i in range(0,5):
#             dates.append(dates[i] + datetime.timedelta(days=1))
#             print(forecast[i:i+5])
#             prev_data = forecast[i:i+5]
#             prev_data = prev_data.reshape(1,5,1)
#             next_data = model.predict(prev_data)
#             forecast = np.append(forecast,next_data)
            
#         forecasted_dates = np.array(dates[1:])
#         forecasted_prices = forecast[5:]
#         return forecasted_dates,forecasted_prices

#     def define_model():
#         model = Sequential()
#         model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
#         model.add(LSTM(64, return_sequences=False))
#         model.add(Dense(25))
#         model.add(Dense(1))
#         model.compile(loss='mean_squared_error',optimizer='adam')
#         return model

#     def plot_training_peformance(y_train,time_train,output_train):
#         plt.plot(time_train,y_train)
#         plt.plot(time_train,output_train)
#         plt.legend(['Actual data','Prediction'])

#     def plot_testing_peformance(y_test,time_test,output):
#         plt.plot(time_test,y_test)
#         plt.plot(time_test,output)
#         plt.legend(['Actual data','Prediction'])



#     def model_summary(model):
#         return model.summary()

#     def input_data(data):
#         time = data[:,0:1]
#         x = data[:,1:-1]
#         x = x.reshape(len(time),len(x[0]),1)
#         y = data[:,-1:]
#         return time,x.astype(np.float32),y.astype(np.float32);
        
#     def data_to_window_data(data):
#         l = []
#         temp = data
#         temp = temp.to_numpy()
#         for i in range(6,len(temp)):
#             t = []
#             t.append(temp[i][0])
#             for j in range(i-6,i):
#                 t.append(temp[j][1])
#             l.append(t)
#         return l

    # def string_to_date(date):
    #     date = date[0:10]
    #     d = date.split('-')
    #     year , month , date = int(d[0]),int(d[1]),int(d[2])
    #     return datetime.datetime(year,month,date)

    # headers = {
    #     'Content-Type': 'application/json'
    # }
    # api = "https://api.tiingo.com/tiingo/daily/appl/prices?startDate=2015-01-02&token=424c723fd790ecb18d9ba7cd2e8834b6b5a7eaa7"
    # requestResponse = requests.get(api, headers=headers)
    # df = pd.DataFrame(requestResponse.json())
    # data = df[['date','close']]
    # data['date'] = data['date'].apply(string_to_date)
    # data.drop(0,axis=0,inplace=True)


    # t = data_to_window_data(data)
    # t = pd.DataFrame(t,columns=['date','day1','day2','day3','day4','day5','Target'])

    # data_np = t.to_numpy()
    # time,x,y = input_data(data_np)
    # split1 = int(len(x) * 0.8)
    # split2 = int(len(x) * 0.9)

    # time_train , x_train , y_train = time[:split1],x[:split1],y[:split1]
    # time_val , x_val , y_val = time[split1:split2],x[split1:split2],y[split1:split2]
    # time_test , x_test , y_test = time[split2:],x[split2:],y[split2:]

    # plt.plot(time,y)
    # plt.plot(time_val,y_val)
    # plt.plot(time_test,y_test)

    # model = Sequential()
    # model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    # model.add(LSTM(64, return_sequences=False))
    # model.add(Dense(25))
    # model.add(Dense(1))
    # model.compile(loss='mean_squared_error',optimizer='adam')

    # model_summary(model)
    # model.fit(x_train,y_train,validation_data = (x_val,y_val) , epochs = 50)

    # Save the model to a file
    # filename = 'apple_stock_predictor.pkl'
    # model = 0
    # if os.path.isfile(filename) != True:
    #     model = define_model()
    #     model.fit(x_train,y_train,validation_data = (x_val,y_val) , epochs = 50)
    #     with open(filename, 'wb') as file:
    #         pk.dump(model, file)
    # else:
        # with open(filename, 'rb') as file:
        #     model = pk.load(file)


    # output = model.predict(x_test)
    # plt.plot(time_test,y_test)
    # plt.plot(time_test,output)
    # plt.legend(['Actual data','Prediction'])


    # output_train = model.predict(x_train)
    # plt.plot(time_train,y_train)
    # plt.plot(time_train,output_train)
    # plt.legend(['Actual data','Prediction'])

    # output_val = model.predict(x_val)
    # plt.plot(time_val,y_val)
    # plt.plot(time_val,output_val)
    # plt.legend(['Actual data','Prediction'])

    # a = x_test[-1:][0]
    # a = a.reshape(5,)

    # forecast = x_test[-1:][0]
    # forecast = forecast.reshape(5,)
    # dates = []
    # dates.append(time_test[len(time_test)-1])
    # for i in range(0,5):
    #     dates.append(dates[i] + datetime.timedelta(days=1))
    #     print(forecast[i:i+5])
    #     prev_data = forecast[i:i+5]
    #     prev_data = prev_data.reshape(1,5,1)
    #     next_data = model.predict(prev_data)
    #     forecast = np.append(forecast,next_data)
        
    # forecasted_dates = np.array(dates[1:])
    # forecasted_prices = forecast[5:]

    # forecasted_dates,forecasted_prices = forecast_prices(model,x_test,time_test)
    # print(forecasted_dates)
    # print(forecasted_prices)

    # plt.plot(time_train,y_train)
    # plt.plot(time_val,y_val)
    # plt.plot(time_test,y_test)
    # plt.plot(forecasted_dates,forecasted_prices)
    # plt.legend(['Train','Validation','Test','forecasted'])

