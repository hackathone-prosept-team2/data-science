import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances

# исходим из того,что таблицы для работы к этому этапу имеются целиком

def preprocessing_data(marketing_dealerprice, marketing_product):
    marketing_dealerprice.drop_duplicates(subset=['product_key', 'product_url', 'product_name'], inplace=True)
    marketing_dealerprice.reset_index(drop=True, inplace=True)
    marketing_product = marketing_product.dropna(subset='name')
    return marketing_dealerprice, marketing_product

# обработка текста
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    # отделение английских слов
    pattern = re.compile(r'(?<=[а-яА-Я])(?=[A-Z])|(?<=[a-zA-Z])(?=[а-яА-Я])')
    text = re.sub(pattern, ' ', text)
    # приведение к нижнему регистру 
    text = text.lower()
    # удаление символов
    text = re.sub(r'\W', ' ', str(text))
    # удаление одиноко стоящих слов
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    # соотношения объемов 
    pattern2 = re.compile(r'\b\d+:\d+\s*-\s*\d+:\d+\b|\s*\d+:\d+\s*')
    text = re.sub(pattern2, ' ', text)
    return "".join(lemmatizer.lemmatize(text)) 


# векторизация названий
def vectoriz(marketing_dealerprice, marketing_product):
    df_1 = marketing_dealerprice[['product_name_lem']]
    df_1 = df_1.rename(columns={'product_name': 'name'})
    df_2 = marketing_product[['name_lem']]
    df_2 = df_2.rename(columns={'name_lem': 'name'})
    df = pd.concat([df_1, df_2])
    count_tf_idf = TfidfVectorizer()
    df = count_tf_idf.fit_transform(df['name'])
    df_1 = count_tf_idf.transform(df_1['name'])
    df_2 = count_tf_idf.transform(df_2['name'])
    df_1 = df_1.toarray()
    df_2 = df_2.toarray()
    return df_1, df_2

# получение матрицы с расстояниями
def matching_names(marketing_product, marketing_dealerprice, df_1, df_2):
    df = pd.DataFrame(index = marketing_product['id'], 
                    columns = marketing_dealerprice['product_key']+ '_' + pd.Series(range(marketing_dealerprice.shape[0])).astype(str), 
                    data = pairwise_distances(df_2, df_1, metric = 'cosine'))
    return df

# вывод n-го количества семантически похожих названий
def top_k_names(df, name, top_k):
    product_key = marketing_dealerprice.loc[marketing_dealerprice['product_name'] == name, 'product_key']
    product_key = product_key.to_list()[0] + '_' + str(product_key.index[0])
    z = df[product_key].sort_values()[:top_k].index.to_list()
    final = marketing_product.loc[marketing_product['id'].isin(z) , 'name']
    return final


name = input()

marketing_dealerprice, marketing_product = preprocessing_data(marketing_dealerprice, marketing_product)
marketing_dealerprice['product_name_lem'] = marketing_dealerprice['product_name'].apply(lemmatize_text)
marketing_product['name_lem'] = marketing_product['name'].apply(lemmatize_text)
df_1, df_2 = vectoriz(marketing_dealerprice, marketing_product)
df = matching_names(marketing_product, marketing_dealerprice, df_1, df_2)
final_names = top_k_names(df, name, top_k)
print(final_names)




