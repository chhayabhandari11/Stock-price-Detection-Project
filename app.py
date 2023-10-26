import requests
import pandas as pd
import datetime
import numpy as np
import pickle as pk
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from flask import Flask , render_template , current_app, url_for
from STOCK_CLASS import STOCK


app = Flask(__name__)


def get_object(stock_symbol):
    if stock_symbol == 'AAPL':
        if not hasattr(current_app, 'APPL'):
            current_app.APPL = STOCK('AAPL')
        return current_app.APPL
    elif stock_symbol == 'GOOGL':
        if not hasattr(current_app, 'GOOGL'):
            current_app.GOOGL = STOCK('GOOGL')
        return current_app.GOOGL
    elif stock_symbol == 'MSFT':
        if not hasattr(current_app, 'MSFT'):
            current_app.MSFT = STOCK('MSFT')
        return current_app.MSFT


@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/apple')
def apple():
    APPL = get_object('AAPL')
    combine = list(zip(APPL.forecasted_dates,APPL.forecasted_data))
    Forecasting_DATA = url_for('static', filename='APPL_FORECASTING_DATA.png')
    DATASET = url_for('static', filename='FETCHED_AAPL_DATA.jpeg')
    WINDOWED_DATA = url_for('static', filename='WINDOWED_AAPL_DATA.jpeg')
    TEST_TRAIN_SPLIT = url_for('static', filename='APPL_PLOTTED_DATA.png')
    PREDICTED_DATA = url_for('static', filename='APPL_PREDICTED_DATA.png')
    return render_template('second_page.html',title='APPLE',combine=combine,Forecasting_DATA=Forecasting_DATA,DATASET=DATASET,WINDOWED_DATA=WINDOWED_DATA,TEST_TRAIN_DATA = TEST_TRAIN_SPLIT,PREDICTED_DATA=PREDICTED_DATA)


@app.route('/google')
def google():
    GOOGL = get_object('GOOGL')
    combine = list(zip(GOOGL.forecasted_dates,GOOGL.forecasted_data))
    Forecasting_DATA = url_for('static', filename='GOOGL_FORECASTING_DATA.png')
    DATASET = url_for('static', filename='FETCHED_GOOGL_DATA.jpeg')
    WINDOWED_DATA = url_for('static', filename='WINDOWED_GOOGL_data.jpeg')
    TEST_TRAIN_SPLIT = url_for('static', filename='GOOGL_PLOTTED_DATA.png')
    PREDICTED_DATA = url_for('static', filename='GOOGL_PREDICTED_DATA.png')
    return render_template('second_page.html',title='Google',combine=combine,Forecasting_DATA=Forecasting_DATA,DATASET=DATASET,WINDOWED_DATA=WINDOWED_DATA,TEST_TRAIN_DATA = TEST_TRAIN_SPLIT,PREDICTED_DATA=PREDICTED_DATA)


@app.route('/microsoft')
def microsoft():
    MSFT = get_object('MSFT')
    combine = list(zip(MSFT.forecasted_dates,MSFT.forecasted_data))
    Forecasting_DATA = url_for('static', filename='MSFT_FORECASTING_DATA.png')
    DATASET = url_for('static', filename='FETCHED_MSFT_DATA.jpeg')
    WINDOWED_DATA = url_for('static', filename='WINDOWED_MSFT_data.jpeg')
    TEST_TRAIN_SPLIT = url_for('static', filename='MSFT_PLOTTED_DATA.png')
    PREDICTED_DATA = url_for('static', filename='MSFT_PREDICTED_DATA.png')
    return render_template('second_page.html',title='Microsoft',combine=combine,Forecasting_DATA=Forecasting_DATA,DATASET=DATASET,WINDOWED_DATA=WINDOWED_DATA,TEST_TRAIN_DATA = TEST_TRAIN_SPLIT,PREDICTED_DATA=PREDICTED_DATA)

if __name__ == "__main__":
    app.run(debug=True)