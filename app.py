
import streamlit as st
import panda as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

@st.cache
def get_data():
  return pd.read_csv("c:/ATVPP2/data.csv")

  def train_model():
    data = get_data()
    x = data.drop("MEDV",axis=1)
    y = data["MEDV"]
    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(x, y)
    return rf_regressor

data = get_data()

model = train_model()

# título
st.title("Data App - Prevendo Valores de Imóveis")

# subtítulo
st.markdown("Este é um Data App utilizado para exibir a solução de Machine Learning para o problema de predição de valores de imóveis de Boston.")

# verificando o dataset
st.subheader("Selecionando apenas um pequeno conjunto de atributos")

# atributos para serem exibidos por padrão
defaultcols = ["RM","PTRATIO","LSTAT","MEDV"]

# defindo atributos a partir do multiselect
cols = st.multiselect("Atributos", data.columns.tolist(), default=defaultcols)

# exibindo os top 10 registro do dataframe
st.dataframe(data[cols].head(10))

st.subheader("Distribuição de imóveis por preço")

# definindo a faixa de valores
faixa_valores = st.slider("Faixa de Preço", float(data.MEDV.min()), 150., (10.0, 100.0))

# filtrando os dados
dados = data[data["MEDV"].between(left=faixa_valores[0], right=faixa_valores[1])]

# plot da distribuição dos dados
f = px.histogram(dados, x="MEDV", nbins=100, title="Distribuição de Preços")
f.update_xaxes(title="MEDV")
f.update_yaxes(title="Total Imóveis")
st.plotly_chart(fig)


