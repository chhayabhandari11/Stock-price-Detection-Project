import requests
import pandas as pd
import datetime
import numpy as np
import pickle as pk
import os
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

class STOCK:
    
    df = None
    window_data = None
    time = None
    x = None
    y = None
    time_train = None
    x_train = None
    y_train = None
    time_val = None
    x_val = None
    y_val = None
    time_test = None
    x_test = None
    y_test = None
    model = None
    forecasted_dates = None
    forecasted_data = None

    def __init__(self,type):    
        headers = {
            'Content-Type': 'application/json'
        }
        url = "https://api.tiingo.com/tiingo/daily/" + type + "/prices?startDate=2015-01-02&token=424c723fd790ecb18d9ba7cd2e8834b6b5a7eaa7"
        requestResponse = requests.get(url, headers=headers)
        self.df = pd.DataFrame(requestResponse.json())
        self.df = self.df[['date','close']]
        self.df['date'] = self.df['date'].apply(self.string_to_date)
        self.df.drop(0,axis=0,inplace=True)
        self.window_data = self.data_to_window_data()
        self.window_data = pd.DataFrame(self.window_data,columns=['date','day1','day2','day3','day4','day5','Target'])
        self.time , self.x , self.y = self.reshape_data()
        self.test_train_split()
        self.plotted_data(type)
        self.define_model()
        self.train_model()
        self.plot_prediction(type)
        self.forecast(type)
        

    def string_to_date(self,date):
        date = date[0:10]
        d = date.split('-')
        y , m , d = int(d[0]),int(d[1]),int(d[2])
        return datetime.datetime(y,m,d)
    
    def data_to_window_data(self):
        l = []
        temp = self.df
        temp = temp.to_numpy()
        for i in range(6,len(temp)):
            t = []
            t.append(temp[i][0])
            for j in range(i-6,i):
                t.append(temp[j][1])
            l.append(t)
        return l

    def reshape_data(self):
        self.window_data_np = self.window_data.to_numpy()
        time = self.window_data_np[:,0:1]
        x = self.window_data_np[:,1:-1]
        x = x.reshape(len(time),len(x[0]),1)
        y = self.window_data_np[:,-1:]
        return time,x.astype(np.float32),y.astype(np.float32)

    def test_train_split(self):
        split1 = int(len(self.x) * 0.8)
        split2 = int(len(self.x) * 0.9)

        self.time_train , self.x_train , self.y_train = self.time[:split1],self.x[:split1],self.y[:split1]
        self.time_val , self.x_val , self.y_val = self.time[split1:split2],self.x[split1:split2],self.y[split1:split2]
        self.time_test , self.x_test , self.y_test = self.time[split2:],self.x[split2:],self.y[split2:]
    
    def plotted_data(self,type):
        fig, ax = plt.subplots()
        ax.plot(self.time_train, self.y_train)
        ax.plot(self.time_val, self.y_val)
        ax.plot(self.time_test, self.y_test)
        ax.legend(['Training', 'Validation', 'Testing'])
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.set_title('Stock Data')
        path = 'C:\\Users\\amanr\\Desktop\\FLASK\\static\\' + type + '_PLOTTED_DATA.png'
        fig.savefig(path)
        plt.close(fig)

    def define_model(self):
        self.model = Sequential()
        self.model.add(LSTM(128, return_sequences=True, input_shape= (self.x_train.shape[1], 1)))
        self.model.add(LSTM(64, return_sequences=False))
        self.model.add(Dense(25))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error',optimizer='adam')

    def show_model_summary(self):
        self.model_summary = self.model.summary()
        return self.model_summary

    def train_model(self):
        self.model.fit(self.x_train,self.y_train,validation_data = (self.x_val,self.y_val) , epochs = 50)

    
    def plot_prediction(self,type):
        fig, ax = plt.subplots()
        ax.plot(self.time_test, self.y_test)
        ax.plot(self.time_test, self.model.predict(self.x_test))
        ax.legend(['Actual data', 'Prediction'])
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.set_title('Stock Prediction')
        path = 'C:\\Users\\amanr\\Desktop\\FLASK\\static\\' + type + '_PREDICTED_DATA.png'
        fig.savefig(path)
        plt.close(fig)

    
    def forecast(self,type):     
        self.forecasted_data = self.x_test[-1:][0]
        self.forecasted_data = self.forecasted_data.reshape(5,)
        dates = []
        dates.append(self.time_test[len(self.time_test)-1])
        for i in range(0,5):
            dates.append(dates[i] + datetime.timedelta(days=1))
            prev_data = self.forecasted_data[i:i+5]
            prev_data = prev_data.reshape(1,5,1)
            next_data = self.model.predict(prev_data)
            self.forecasted_data = np.append(self.forecasted_data,next_data)

        self.forecasted_dates = np.array(dates[1:])
        self.forecasted_data = self.forecasted_data[5:]
        fig, ax = plt.subplots()
        ax.plot(self.forecasted_dates, self.forecasted_data)
        ax.legend(['Forecasted_data'])
        ax.set_xlabel('DATE')
        ax.set_ylabel('Price')
        ax.set_title('Stock Forecasting')
        path = 'C:\\Users\\amanr\\Desktop\\FLASK\\static\\'+ type + '_FORECASTING_DATA.png'
        fig.savefig(path)
        plt.close(fig)


# APPL = STOCK('AAPL')
# print(APPL.forecasted_data)