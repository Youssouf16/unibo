import pandas as pd
import numpy as np
import os
import requests
import base64
import time
import json
import ast
from itertools import product


def get_access_token():
    """
    get access token to get geolocalization data from Twitter API
    """
    #keys
    key = ''
    secret_key = ''

    #reformatting keys - base64key
    secret_key_ = '{}:{}'.format(key, secret_key).encode('ascii')
    b64_encoded_key = base64.b64encode(secret_key_)
    b64_encoded_key = b64_encoded_key.decode('ascii')

    #sending our key to Twitter API
    auth_url = 'https://api.twitter.com/oauth2/token'
    auth_headers = {
        'Authorization': 'Basic {}'.format(b64_encoded_key),
        'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'
    }
    auth_data = {
        'grant_type': 'client_credentials'
    }
    auth_resp = requests.post(auth_url, headers=auth_headers, data=auth_data)
    print(auth_resp.status_code)
    access_token = auth_resp.json()['access_token']
    
    return access_token


def get_geo_data(place_id, access_token):
    """
    get geolocalization data from Twitter API
    """

    geo_headers = {
    'Authorization': 'Bearer {}'.format(access_token)    
    }

    geo_url = 'https://api.twitter.com/1.1/geo/id/{}.json'.format(place_id)  
    geo_resp = requests.get(geo_url, headers=geo_headers)

    return geo_resp.json()


def rain_request(x):
    """
    getting rainfall data from worldweather API
    """
    #create enddate
    tmp = x[0]
    tmp = tmp.replace(tmp[5:7], '{:02d}'.format(int(tmp[5:7])+1))
    tmp = tmp.replace(tmp[8:10], '07')
    
    #request data month by month
    base_url = 'https://api.worldweatheronline.com/premium/v1/past-weather.ashx'
    key = ''
    q = '{},nigeria'.format(x[1])
    extra = 'utcDateTime'
    date = x[0]
    enddate = tmp
    format = 'json'
    tp = '24'

    request_ = '{}?key={}&q={}&extra={}&date={}&enddate={}&tp={}&format={}'\
        .format(base_url, key, q, extra, date, enddate, tp, format)

    resp = requests.get(request_)
    out = resp.json()

    return out


def dict_selection(x, date):
    """
    select the right datum from the dictionary
    """
    try:
        for i in range(len(x['data']['weather'])):
            if x['data']['weather'][i]['date'] == str(date)[:10]:
                select = x['data']['weather'][i]['hourly'][0]['precipMM']
    except KeyError:
        select = np.NaN
    return select