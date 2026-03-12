import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

st.set_page_config(page_title="Dashboard Titanic", layout="wide")
st.title("🚢 Relatório Consolidado Titanic")

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(diretorio_atual, 'modelo_titanic.pkl')
DATA_PATH = os.path.join(diretorio_atual, 'train_titanic.csv')

@st.cache_data
def carregar_dados_e_gerar():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        st.error("❌ Arquivos .pkl ou .csv não encontrados na raiz!")
        st.stop()
        
    model = joblib.load(MODEL_PATH)
    df_original = pd.read_csv(DATA_PATH)
    
    df_pred = df_original.copy()
    df_pred['Sex'] = df_pred['Sex'].map({'male': 0, 'female': 1})
    df_pred['Age'] = df_pred['Age'].fillna(df_pred['Age'].median())
    df_pred['Fare'] = df_pred['Fare'].fillna(df_pred['Fare'].median())
    
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']
    
    df_original['Probabilidade'] = model.predict_proba(df_pred[features])[:, 1]
    
    return df_original

try:
    df_final = carregar_dados_e_gerar()

    st.subheader("📋 Histórico e Chances de Sobrevivência")
    st.dataframe(
        df_final[['PassengerId', 'Name', 'Sex', 'Age', 'Probabilidade']].sort_values(by='Probabilidade', ascending=False),
        column_config={"Probabilidade": st.column_config.ProgressColumn("Chance", format="%.2f", min_value=0, max_value=1)},
        use_container_width=True, hide_index=True
    )

    st.divider()

    st.subheader("📊 Relatório de Padrões")
    c1, c2 = st.columns(2)
    with c1:
        fig_sex = px.histogram(df_final, x='Sex', y='Probabilidade', histfunc='avg', title="Chance Média por Sexo", color='Sex')
        st.plotly_chart(fig_sex, use_container_width=True)
    with c2:
        fig_pclass = px.box(df_final, x='Pclass', y='Probabilidade', title="Distribuição por Classe", color='Pclass')
        st.plotly_chart(fig_pclass, use_container_width=True)

    st.success("Dashboard carregado com sucesso!")

except Exception as e:
    st.error(f"Erro de compatibilidade: {e}")
    st.warning("Dica: Se o erro persistir, rode o 'train.py' de novo para atualizar o arquivo .pkl")