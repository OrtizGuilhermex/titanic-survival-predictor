# 🚢 Titanic Survival Predictor & Analytics Dashboard

Este projeto utiliza Machine Learning para prever a probabilidade de sobrevivência dos passageiros do Titanic e apresenta os resultados através de um dashboard interativo. 

O objetivo é ir além de uma classificação binária (viveu/morreu), oferecendo uma visão estatística detalhada e análise de padrões baseada nas características dos passageiros.

---

# 🚀 Funcionalidades

- **Predição Automática:** Processamento em massa do dataset para cálculo de probabilidades.
- **Dashboard Interativo:** Visualização de dados utilizando Streamlit.
- **Histórico Detalhado:** Tabela de passageiros com barra de progresso para níveis de probabilidade.
- **Relatório de Insights:** Gráficos que demonstram o impacto do Sexo, Classe Social e Idade na sobrevivência.
- **Exportação:** Geração de arquivo de submissão no formato padrão do Kaggle.

---

# 🛠️ Tecnologias Utilizadas

- **Python 3.10+**
- **Scikit-Learn:** Para o modelo de Random Forest.
- **Pandas:** Manipulação e limpeza de dados.
- **Streamlit:** Interface web do dashboard.
- **Plotly:** Gráficos dinâmicos e interativos.
- **Joblib:** Persistência do modelo treinado.

---

# 📁 Estrutura do Projeto

```text
├── modelos/
│   └── modelo_titanic.pkl    # Modelo treinado salvo
├── train.py                  # Script para treinamento do modelo
├── app.py                    # Script principal do Dashboard (Streamlit)
├── train_titanic.csv         # Dataset de treino/base de dados
└── requirements.txt          # Dependências do projeto
```

---

# 🔧 Como Executar

## 1️⃣ Clonar o repositório

```bash
git clone https://github.com/ortizGuilhermex/titanic-dashboard.git
cd titanic-dashboard
```

## 2️⃣ Instalar dependências

```bash
pip install -r requirements.txt
```

## 3️⃣ Treinar o modelo

Antes de rodar o app, gere o arquivo `.pkl`:

```bash
python train.py
```

## 4️⃣ Rodar o Dashboard

```bash
streamlit run app.py
```

---

# 📊 O que o modelo analisa?

O algoritmo **Random Forest** foi treinado considerando:

- **Pclass (Classe):** Impacto socioeconômico na prioridade de resgate.
- **Sex (Sexo):** Reflete a política de "mulheres e crianças primeiro".
- **Age (Idade):** Identifica a taxa de sobrevivência por faixa etária.
- **SibSp:** Número de irmãos ou cônjuges (grupos familiares).
- **Fare (Tarifa):** Correlação entre poder aquisitivo e sobrevivência.

---

# 📈 Resultados e Insights

- **Viés de Gênero:** Passageiros do sexo feminino possuem probabilidade média significativamente superior.
- **Hierarquia de Classes:** A **1ª Classe** detém a maior taxa de sucesso no resgate.
- **Fator Financeiro:** Existe uma correlação positiva direta entre o valor da **Tarifa (Fare)** e a chance de vida.

---

# 📦 requirements.txt

Crie um arquivo chamado **requirements.txt** na raiz do projeto e adicione:

```text
pandas
scikit-learn
streamlit
joblib
plotly
statsmodels
```

---

# 👨‍💻 Autor

Desenvolvido por **Luiz Guilherme**  
Desafio **Titanic ML - Kaggle**
