import pandas as pd
import os
import re
import pickle
from logzero import logger
import json

# concat_df["uppercase"] = concat_df.apply(
#     lambda row: " ".join(re.findall(r"[A-Z][a-z]*", row["body"])), axis=1
# )
# potential_cities = concat_df["uppercase"].str.cat(sep=" ")
# cities_in_review = set([])
# all_cities = get_cities()
# for i, city in enumerate(all_cities):
#     if i % 1000 == 0:
#         logger.debug(f"Parsing city {i}, found {len(cities_in_review)} cities")
#     if city in potential_cities:
#         cities_in_review.add(city)
# with open("cities.pkl", "wb") as f:
#     pickle.dump(cities_in_review, f)


class Preprocessing:
    def __init__(self):
        self.i = 0
        self.concat_df = self.concat_raw()

    def concat_raw(self):

        if not os.path.isfile(
            os.path.join("..", "data", "preprocessed", "concat_raw_reviews.csv")
        ):
            df1 = pd.read_csv(
                os.path.join("..", "data", "inputs", "skytrax_scraping_2.csv"),
                index_col=0,
            )
            df2 = pd.read_csv(
                os.path.join("..", "data", "inputs", "seatguru_python_scraping.csv"),
                index_col=0,
            )
            concat_df = pd.concat([df1, df2])
            concat_df.to_csv(
                os.path.join("..", "data", "preprocessed", "concat_raw_reviews.csv")
            )
        concat_df = pd.read_csv(
            os.path.join("..", "data", "preprocessed", "concat_raw_reviews.csv"),
            index_col=0,
        )
        return concat_df

    def preprocess_row(self, row, cities_in_reviews):
        if self.i % 10_000 == 0:
            logger.debug(f"Parsing review {self.i} / {len(self.concat_df)}")
        self.i += 1
        body = row["body"]
        if "Verified | " in body:
            body = body.split("Verified | ")[-1]
        elif "Review | " in body:
            body = body.split("Review | ")[-1]
        row["body"] = body
        upper_case_body = " ".join(re.findall(r"[A-Z][a-z]*", body))
        cities = []
        countries = []
        for city, country in cities_in_reviews.items():
            if city in upper_case_body:
                cities.append(city)
                countries.append(country)
        row["cities"] = cities
        row["countries"] = list(set(countries))
        return row

    def preprocess(self):
        with open(
            os.path.join("..", "data", "resources", "cities_and_countries.json"), "rb"
        ) as f:
            cities_in_reviews = json.load(f)

        self.preprocessed_reviews = self.concat_df.apply(
            lambda row: self.preprocess_row(row, cities_in_reviews), axis=1
        )
        self.preprocessed_reviews.to_pickle("preprocessed_reviews.pkl")


if __name__ == "__main__":
    obj = Preprocessing()
    obj.preprocess()
