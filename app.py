import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load('modelo_titanic.pkl')

try:
    model = load_model()
except:
    st.error("❌ Arquivo 'modelo_titanic.pkl' não encontrado. Rode o script de treino primeiro!")
    st.stop()

st.title("🚢 Simulador de Sobrevivência: Titanic")
st.info("Este app utiliza Inteligência Artificial (Random Forest) para prever se um passageiro sobreviveria ao desastre de 1912.")

with st.expander("📝 Dados do Passageiro", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        sexo = st.selectbox("Sexo", ["Masculino", "Feminino"])
        idade = st.slider("Idade", 0, 90, 25)
        pclasse = st.radio("Classe do Navio", [1, 2, 3], horizontal=True, help="1ª Classe é a mais luxuosa")

    with col2:
        tarifa = st.number_input("Preço da Passagem (Fare)", min_value=0.0, max_value=512.0, value=30.0)
        parentes = st.number_input("Nº de Irmãos/Cônjuges a bordo", 0, 10, 0)

if st.button("Analisar Chance de Sobrevivência"):
    sexo_bin = 0 if sexo == "Masculino" else 1
    
    dados = pd.DataFrame([[pclasse, sexo_bin, idade, parentes, tarifa]], 
                         columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Fare'])
    
    predicao = model.predict(dados)[0]
    probabilidade = model.predict_proba(dados)[0][1]

    st.markdown("---")
    
    if predicao == 1:
        st.balloons()
        st.success(f"### Resultado: SOBREVIVERIA")
        st.metric(label="Probabilidade de Sobrevivência", value=f"{probabilidade:.1%}")
        st.write("Baseado nas características, o modelo indica que este passageiro teria prioridade no acesso aos botes.")
    else:
        st.error(f"### Resultado: NÃO SOBREVIVERIA")
        st.metric(label="Probabilidade de Sobrevivência", value=f"{probabilidade:.1%}")
        st.write("Infelizmente, as estatísticas para este perfil de passageiro são desfavoráveis conforme os dados históricos.")

st.markdown("---")
st.caption("Projeto de Machine Learning utilizando o dataset clássico do Titanic.")