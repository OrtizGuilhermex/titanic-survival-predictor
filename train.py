import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('train_titanic.csv')

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']
target = 'Survived'

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

x = df[features]
y = df[target]

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(x, y)

joblib.dump(model, 'modelo_titanic.pkl')
print("✅ Modelo treinado com sucesso!")