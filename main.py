import re
import html2text
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Crie uma instância do LabelEncoder
label_encoder = LabelEncoder()



df = pd.read_csv('teste.csv')
df = df.drop('codigo', axis=1)
df = df.drop('vigência', axis=1)
df = df.drop('data_assinatura', axis=1)
df = df.drop('links', axis=1)
df = df.drop('a', axis=1)


#==============# filtrando os dados #==============#
def remover_html_links(texto_html):
    # Converte HTML para texto
    texto_sem_html = html2text.html2text(texto_html)
    # Remove links usando expressão regular
    texto_sem_links = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', texto_sem_html)
    return texto_sem_links

# Aplica a função para remover HTML e links a uma coluna específica (por exemplo, 'Conteudo')
df['Conteudo'] = df['Conteudo'].apply(remover_html_links)

# Lista personalizada de stop words em português
stop_words_pt = get_stop_words('portuguese')

# Vetorização usando TF-IDF com a lista personalizada de stop words
tfidf_vectorizer = TfidfVectorizer(max_features=50, stop_words=stop_words_pt)
X_tfidf = tfidf_vectorizer.fit_transform(df['Conteudo'])


# Criando um DataFrame para visualização
dfX_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out(), index=range(1, len(df['Conteudo'])+1))
print(dfX_tfidf)

'''
# Visualização
print("Matriz TF-IDF:")
print(df_tfidf)
'''
df['Classificação_encoded'] = label_encoder.fit_transform(df['Classificação'])
print(df)


xgb_clf = XGBClassifier()

x, y = dfX_tfidf, df['Classificação_encoded']

xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.2, random_state=123)

xgb_clf.fit(xtrain, ytrain)

def print_score(clf, xtrain, xtest, ytrain, ytest, train=True):
    if train:
        pred = clf.predict(xtrain)
        x = ytrain
    elif train==False:
        pred = clf.predict(xtest)
        x = ytest
    clf_report = pd.DataFrame(classification_report(x, pred, output_dict=True))
    print("Train Result:\n================================================")
    print(f"Accuracy Score: {accuracy_score(x, pred) * 100:.2f}%")
    print("_______________________________________________")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")
    print("_______________________________________________")
    print(f"Confusion Matrix: \n {confusion_matrix(x, pred)}\n")

print_score(xgb_clf, xtrain, xtest, ytrain, ytest, train=0)

novo_extrato = 'Espécie: Proc. 23072.257253/2021-48 - Contrato de Licenciamento para exploração dos Programas de Computador registrados sob os números 20210012, 20210013, 20210014 e 20210015, que entre si celebram a Universidade Federal de Minas Gerais - UFMG - CNPJ nº 17.217.985/0001-04, por meio de sua Coordenadoria de Transferência e Inovação Tecnológica - CTIT, a Companhia de Desenvolvimento de Minas Gerais - CODEMGE - CNPJ nº 29.768.219/0001-17, o Serviço Nacional de Aprendizagem Industrial - SENAI CIMATEC - CNPJ nº 03.795.071/0013/50, conjuntamente denominadas LICENCIANTES, e a FABNS Fábrica de Nanosoluções e Participações Ltda. - CNPJ nº 36.615.002/0001-32, doravante denominada LICENCIADA, com interveniência da Fundação de Desenvolvimento da Pesquisa - FUNDEP - CNPJ nº 18.720.938/0001-41. Objeto: Constitui objeto do presente Contrato de Licenciamento, a título oneroso, sem exclusividade, pelas Licenciantes à Licenciada, dos direitos para uso, desenvolvimento, produção, exploração comercial, prestação de serviços ou obtenção de qualquer vantagem econômica relacionada aos Programas de Computador intitulados "Software para controle de equipamento para espectroscopia óptica de campo próximo" registrado sob o nº 20210012, conferido pela CTIT em 07/05/2021, " Software para análise flexível de dados gerados por equipamentos de caracterização espectral", registrado sob o nº 20210013, conferido pela CTIT em 07/05/2021, "Software embarcado para comunicação de dados, interface e controle de dispositivos periféricos para espectroscopia óptica de campo próximo", registrado sob o nº 20210014, conferido pela CTIT em 07/05/2021 e "Design em Verilog/SystemVerilog para gerenciamento das malhas de controle e comunicação com os periféricos utilizados no equipamento para realização de scan AFM (Atomic Force Microscope), STM ( Scanning Tunneling Microscope) e TERS ( Tip-Enhanced Raman Scattering)", registrado sob o nº 20210015, conferido pela CTIT em 07/05/2021 . Início da vigência: o presente instrumento terá vigência de 10 (dez) anos, a contar da data de sua assinatura, em 10 de março de 2022 . Nomes e cargos dos signatários: o Prof. Gilberto Medeiros Ribeiro - Diretor da CTIT/UFMG, os Srs. Mateus Ayer Quintela e Eduardo Zimmer Sampaio - representantes legais da CODEMGE, o Sr. Leone Peter Correia da Silva Andrade - Diretor da Tecnologia e Inovação e Reitor do Centro Universitário SENAI CIMATEC, os Srs. Cassiano Rabelo e Silva e Hudson Luiz Silva de Miranda - Diretores da FABNS, o Prof. Jaime Arturo Ramírez - Presidente da FUNDEP.'
# Pré-processamento
novo_extrato_sem_html_links = remover_html_links(novo_extrato)

# Vetorização 
novo_extrato_tfidf = tfidf_vectorizer.transform([novo_extrato_sem_html_links])

# Classificação
predicao = xgb_clf.predict(novo_extrato_tfidf)

# Traduzir a classe de volta para a categoria original (usando o label_encoder)
categoria_predita = label_encoder.inverse_transform(predicao)

print("O novo extrato foi classificado como:", categoria_predita[0])