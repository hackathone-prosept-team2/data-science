import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances


class RecommendationService:
    def __init__(
        self,
        products_file: str = "data/marketing_product.csv",
        prices_file: str = "data/marketing_dealerprice.csv",
    ) -> None:
        self.prices = self._prepare_prices(prices_file)
        self.products = self._prepare_products(products_file)
        self.dataframe = self._make_dataframe()

    def _prepare_products(self, products_file):
        products = self._read_file(products_file)
        products = products.dropna(subset="name")
        products["name_lem"] = products["name"].apply(self._lemmatize_text)
        return products

    def _prepare_prices(self, prices_file):
        prices = self._read_file(prices_file)
        prices.drop_duplicates(
            subset=["product_key", "product_url", "product_name"], inplace=True
        )
        prices.reset_index(drop=True, inplace=True)
        prices["product_name_lem"] = prices["product_name"].apply(
            self._lemmatize_text
        )
        return prices

    def _make_dataframe(self):
        df_1, df_2 = self._vectoriz()
        dataframe = self._matching_names(df_1, df_2)
        return dataframe

    @staticmethod
    def _read_file(path):
        return pd.read_csv(path, sep=";")

    def _preprocessing_data(self, prices, products):
        prices.drop_duplicates(
            subset=["product_key", "product_url", "product_name"], inplace=True
        )
        prices.reset_index(drop=True, inplace=True)
        products = products.dropna(subset="name")
        return prices, products

    @staticmethod
    def _lemmatize_text(text):
        lemmatizer = WordNetLemmatizer()
        # отделение английских слов
        pattern = re.compile(
            r"(?<=[а-яА-Я])(?=[A-Z])|(?<=[a-zA-Z])(?=[а-яА-Я])"
        )
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
        return "".join(lemmatizer.lemmatize(text))

    def _vectoriz(self):
        df_1 = self.prices[["product_name_lem"]]
        df_1 = df_1.rename(columns={"product_name_lem": "name"})
        df_2 = self.products[["name_lem"]]
        df_2 = df_2.rename(columns={"name_lem": "name"})
        df = pd.concat([df_1, df_2])
        count_tf_idf = TfidfVectorizer()
        df = count_tf_idf.fit_transform(df["name"])
        df_1 = count_tf_idf.transform(df_1["name"])
        df_2 = count_tf_idf.transform(df_2["name"])
        df_1 = df_1.toarray()
        df_2 = df_2.toarray()
        return df_1, df_2

    def _matching_names(self, df_1, df_2):
        df = pd.DataFrame(
            index=self.products["id"],
            columns=self.prices["product_key"]
            + "_"
            + pd.Series(range(self.prices.shape[0])).astype(str),
            data=pairwise_distances(df_2, df_1, metric="cosine"),
        )
        return df

    def get_recommendations(
        self, dealer_name: str, recommendations_number: int = 10
    ):
        # получаем ключи по названию
        product_key = self.prices.loc[
            self.prices["product_name"] == dealer_name, "product_key"
        ]
        product_key = (
            product_key.to_list()[0] + "_" + str(product_key.index[0])
        )
        # получаем id и расстояния
        z = self.dataframe[product_key].sort_values()[:recommendations_number]
        # формиркем список списков на выход
        z = pd.DataFrame(z)
        z["id"] = z.index.values
        z = z.values.tolist()
        return z


import time

a = time.time()
service = RecommendationService()
print("подготовка", time.time() - a)
a = time.time()
name1 = "Огнебиозащита PROSEPT prof 2 группа для наружных и внутренних работ с индикатором 10л"
name2 = "Просепт Professional Bath Acid Plus усиленное средство для ухода за сантехникой (750 мл)"
name3 = "ОгнебиозащитадлядревесиныProseptОГНЕБИОPROFI,2л,007-5"
res = service.get_recommendations(dealer_name=name1)
print(res)
print("слово1", time.time() - a)
a = time.time()

res = service.get_recommendations(dealer_name=name2)
print(res)
print("слово2", time.time() - a)
a = time.time()

res = service.get_recommendations(dealer_name=name3)
print(res)
print("слово3", time.time() - a)
a = time.time()
