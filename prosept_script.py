import pandas as pd
import numpy as np
import os
import re
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
nlp = spacy.load("ru_core_news_sm")


# исходим из того,что таблицы для работы к этому этапу имеются целиком
TOP_K = 5
DEALER_PRICE = pd.read_csv("data/marketing_dealerprice.csv", sep=";")
PRODUCT = pd.read_csv("data/marketing_product.csv", sep=";")


def preprocessing_data(marketing_dealerprice, marketing_product):
    marketing_dealerprice.drop_duplicates(
        subset=["product_key", "product_url", "product_name"], inplace=True
    )
    marketing_dealerprice.reset_index(drop=True, inplace=True)
    marketing_product = marketing_product.dropna(subset="name")
    marketing_product.reset_index(drop=True, inplace=True)
    return marketing_dealerprice, marketing_product


# обработка текста
def lemmatize_text(text):
    # отделение английских слов
    pattern = re.compile(r"(?<=[а-яА-Я])(?=[A-Z])|(?<=[a-zA-Z])(?=[а-яА-Я])")
    text = re.sub(pattern, " ", text)
    # приведение к нижнему регистру
    text = text.lower()
    # удаление символов
    text = re.sub(r"\W", " ", str(text))
    # удаление одиноко стоящих слов
    text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)
    # соотношения объемов
    pattern2 = re.compile(r"\b\d+:\d+\s*-\s*\d+:\d+\b|\s*\d+:\d+\s*")
    text = re.sub(pattern2, " ", text)
    # лемматизация
    spacy_results = nlp(text)
    text = " ".join([token.lemma_ for token in spacy_results])
    return text


# векторизация названий
def vectoriz(marketing_dealerprice, marketing_product):
    df_1 = marketing_dealerprice[["product_name_lem"]]
    df_1 = df_1.rename(columns={"product_name_lem": "name"})
    df_2 = marketing_product[["name_lem"]]
    df_2 = df_2.rename(columns={"name_lem": "name"})
    df = pd.concat([df_1, df_2])
    count_tf_idf = TfidfVectorizer()
    df = count_tf_idf.fit_transform(df["name"])
    df_1 = count_tf_idf.transform(df_1["name"])
    df_2 = count_tf_idf.transform(df_2["name"])
    df_1 = df_1.toarray()
    df_2 = df_2.toarray()
    return df_1, df_2


# получение матрицы с расстояниями
def matching_names(marketing_product, marketing_dealerprice, df_1, df_2):
    df = pd.DataFrame(
        index=marketing_product["id"],
        columns=marketing_dealerprice["product_key"]
        + "_"
        + pd.Series(range(marketing_dealerprice.shape[0])).astype(str),
        data=pairwise_distances(df_2, df_1, metric="cosine"),
    )
    return df


# вывод n-го количества семантически похожих названий
def top_k_names(df, name, top_k):
    # получаем ключи по названию
    product_key = DEALER_PRICE.loc[
        DEALER_PRICE["product_name"] == name, "product_key"
    ]
    product_key = product_key.to_list()[0] + "_" + str(product_key.index[0])
    # получаем id и расстояния
    z = df[product_key].sort_values()[:top_k]
    # формиркем список списков на выход
    z = pd.DataFrame(z)
    z["id"] = z.index.values
    z = z.values.tolist()
    return z


def main(dealer_key_name):
    # name = input()

    marketing_dealerprice, marketing_product = preprocessing_data(
        DEALER_PRICE, PRODUCT
    )
    marketing_dealerprice["product_name_lem"] = marketing_dealerprice[
        "product_name"
    ].apply(lemmatize_text)
    marketing_product["name_lem"] = marketing_product["name"].apply(
        lemmatize_text
    )
    df_1, df_2 = vectoriz(marketing_dealerprice, marketing_product)
    df = matching_names(marketing_product, marketing_dealerprice, df_1, df_2)
    final_names = top_k_names(df, dealer_key_name, TOP_K, DEALER_PRICE)
    print(final_names)


if __name__ == "__main__":
    main(
        dealer_key_name="Огнебиозащита PROSEPT prof 2 группа для наружных и внутренних работ с индикатором 10л"
    )
