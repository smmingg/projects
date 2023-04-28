#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, jsonify, request
from flask_marshmallow import Marshmallow
from marshmallow import Schema, fields
from flask_restful import Resource, Api
import json
import pandas as pd
import boto3
import io
import os
import time
import datetime
import requests


app = Flask(__name__)
api = Api(app)
ma = Marshmallow(app)


 
@app.route('/')
def home():
    return '''<h1>Testing</h1>
    <p>A prototype API.</p>'''


class BotSchema(ma.Schema):
    id = fields.Str()
    model = fields.Str()
    version = fields.Str()
    hostname = fields.Str()
    ip_address = fields.Str()
    
    
class TargetSchema(ma.Schema):
    host = fields.Str()
    host_status = fields.Str()
    username = fields.Str()
    password = fields.Str()
    login_status = fields.Str()
    provider_id = fields.Str()
    login_id = fields.Str()
    
    
class PayDataSchema(ma.Schema):
    ln = fields.Str(allow_none=True)
    mt = fields.Str(allow_none=True)
    ed = fields.Str(allow_none=True)
    et = fields.Str(allow_none=True)
    rs = fields.Str(allow_none=True)
    rt = fields.Str(allow_none=True)
    th = fields.Str(allow_none=True)
    ta = fields.Str(allow_none=True)
    ot = fields.Int(allow_none=True)
    fhdpt = fields.Str(allow_none=True)
    fhdp = fields.Str(allow_none=True)
    fhdph = fields.Str(allow_none=True)
    fhdpa = fields.Str(allow_none=True)
    fou = fields.Str(allow_none=True)
    fover = fields.Str(allow_none=True)
    funder = fields.Str(allow_none=True)
    f1 = fields.Str(allow_none=True)
    fx = fields.Str(allow_none=True)
    f2 = fields.Str(allow_none=True)
    hhdpt = fields.Str(allow_none=True)
    hhdp = fields.Str(allow_none=True)
    hhdph = fields.Str(allow_none=True)
    hhdpa = fields.Str(allow_none=True)
    hou = fields.Str(allow_none=True)
    hover = fields.Str(allow_none=True)
    hunder = fields.Str(allow_none=True)
    h1 = fields.Str(allow_none=True)
    hx = fields.Str(allow_none=True)
    h2 = fields.Str(allow_none=True)
    arc = fields.Str(allow_none=True)
    hrc = fields.Str(allow_none=True)
    hodd = fields.Str(allow_none=True)
    fodd = fields.Str(allow_none=True)
    feven = fields.Str(allow_none=True)
    fmlh = fields.Str(allow_none=True)
    fmla = fields.Str(allow_none=True)
    heven = fields.Str(allow_none=True)
    hmlh = fields.Str(allow_none=True)
    hmla = fields.Str(allow_none=True)
    lnt = fields.Str(allow_none=True)
    refid = fields.Int(allow_none=True)

    
class PayloadSchema(ma.Schema):
    data = fields.List(fields.Nested(PayDataSchema))
    market_type = fields.Str()
    sport_type = fields.Str()
    bet_type = fields.Str()
    count = fields.Int()
    
    
class OddsSchema(ma.Schema):
    date_time = fields.Str()
    bot = fields.Nested(BotSchema)
    target = fields.Nested(TargetSchema)
    channel = fields.Str()
    queue = fields.Str()
    epoch_time = fields.Int()
    login_id = fields.Str()
    payload = fields.Nested(PayloadSchema)
        

class Odds():
    def __init__(self, date_time, bot, target, channel, queue, epoch_time, login_id, payload):
        self.date_time = date_time
        self.bot = bot
        self.target = target
        self.channel = channel
        self.queue = queue
        self.epoch_time = epoch_time
        self.login_id = login_id
        self.payload = payload


data = []

@app.route("/api/push", methods=['POST'])
def post():
    jsondata = request.get_json()
    odds_schema = OddsSchema()
    item_data = odds_schema.load(jsondata) 
    data.append(item_data)
    json_full = jsonify({'Data':data})
    return json_full

    
@app.route("/api/get", methods=['GET'])
def get():
    try:
        temp_data = data
        return jsonify({'Data':temp_data})
    
    finally:
        data.clear()

# Get
# @app.route('/api/get', methods=['GET'])
# def get():
#     return jsonify({'Data':data}) 



if __name__ == '__main__':
    app.run(debug=True)

    
