from flask import Flask, render_template, request, Response, url_for
import requests
import json

import pandas as pd
import numpy as np
import os
from os.path import join, dirname, realpath

import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

app = Flask(
  __name__,
  template_folder='templates',
  static_folder='static'
)

base_dir = dirname(realpath(__file__))
df_pop = pd.read_pickle(join(base_dir, 'static/data/dfpo1/df_pop.pkl'))
df_pop = df_pop.reset_index()

response_API = requests.get('https://servicodados.ibge.gov.br/api/v1/localidades/municipios')
df_municipios = pd.read_json(response_API.text)

df_proc_mun = pd.read_pickle(join(base_dir, 'static/data/dfpo1/df_proc_mun.pkl'))
df_procedimentos = df_proc_mun[['codigo_municipio', 'codigo_procedimento', 'procedimento']].drop_duplicates()

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/mun', defaults={'search': ''})
@app.route('/mun/<search>')
def get_mun(search=''):
    merged = pd.merge(df_pop.astype({'codigo': int}), df_municipios, left_on='codigo', right_on='id')
    
    search = request.args.get('search')
    if search != '' and search != None:
        merged = merged[merged['nome'].str.upper()
                    .str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
                    .str.contains(search.upper())]
    return Response(merged[['codigo', 'nome']].to_json(orient="records"), mimetype='application/json')

@app.route('/proc')
def get_proc(mun=0, search=''):
    mun = int(request.args.get('mun')[0:6])
    search = request.args.get('search')
    merged = pd.DataFrame(df_procedimentos[df_procedimentos['codigo_municipio'] == mun])
    if search != '' and search != None:
        merged = df_procedimentos[df_procedimentos['procedimento'].str.upper()
                    .str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
                    .str.contains(search.upper())]
    return Response(merged.to_json(orient="records"), mimetype='application/json')

@app.route('/pop/<codigo>')
def pop(codigo):
    
    df = df_pop[df_pop["codigo"] == codigo].drop('codigo', axis=1).transpose()
    if df.shape[1] == 0:
        return ""
        
    df.index = df.index.set_names('year')
    df.columns = ['count']
    df = df.reset_index()
    
    histograma_fig = px.histogram(df,
                              title="Histograma",
                              x="year",
                              y="count",
                              width=700,
                              height=400)
    
    return histograma_fig.to_html()

@app.route('/procmun')
def proc_mun(mun=0, proc=0):
    mun = int(request.args.get('mun')[0:6])
    proc = int(request.args.get('proc'))
    
    df_proc_mun = pd.read_pickle(join(base_dir, 'static/data/dfpo1/df_proc_mun.pkl'))
    df = df_proc_mun[(df_proc_mun['codigo_municipio'] == mun) & (df_proc_mun['codigo_procedimento'] == proc)].drop(['codigo_municipio', 'municipio', 'uf', 'codigo_procedimento', 'procedimento'], axis=1).transpose()
    if df.shape[1] == 0:
        return ""
        
    df.index = df.index.set_names('year')
    df.columns = ['count']
    df = df.reset_index()
    
    histograma_fig = px.histogram(df,
                              title="Histograma",
                              x="year",
                              y="count",
                              width=700,
                              height=400)
    
    return histograma_fig.to_html()
    
@app.route('/calculate',methods = ['POST', 'GET'])
def calculate():
    if request.method == 'POST':
        value1 = request.form['value1']
        value2 = request.form['value2']
    else:
        value1 = request.args.get('value1')
        value2 = request.args.get('value2')
        
    # processa os dados ML
    return "calculate " + str(value1) + str(value2)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port, threaded=True)