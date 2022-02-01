import pandas as pd
import os
import re
import pickle
from logzero import logger
import json
import numpy as np

from collections import OrderedDict

AIRPLANE_NAMES = OrderedDict()
AIRPLANE_NAMES[r".*embraer.*"] = ["short_haul", "small"]
AIRPLANE_NAMES[r".*erj.*"] = ["short_haul", "small"]
AIRPLANE_NAMES[r"e1\d\d"] = ["short_haul", "small"]
AIRPLANE_NAMES[r".*crj.*"] = ["short_haul", "small"]
AIRPLANE_NAMES[r".*bombardier.*"] = ["short_haul", "small"]
AIRPLANE_NAMES[r".*atr.*"] = ["short_haul", "small"]
AIRPLANE_NAMES[r".*a(irbus|\B)31\d.*"] = ["short_haul", "small"]
AIRPLANE_NAMES[r".*dhc.*"] = ["short_haul", "small"]
AIRPLANE_NAMES[r".*7[3, 2, 1, 0]7.*"] = ["short_haul", "medium"]
AIRPLANE_NAMES[r".*a(irbus|\B)32\d.*"] = ["short_haul", "medium"]
AIRPLANE_NAMES[r".*md.*"] = ["short_haul", "medium"]
AIRPLANE_NAMES[r".*a(irbus|\B)3[5, 3]\d.*"] = ["long_haul", "large"]
AIRPLANE_NAMES[r".*7[8, 6, 5]7.*"] = ["long_haul", "large"]
AIRPLANE_NAMES[r".*a(irbus|\B)38\d.*"] = ["long_haul", "jumbo"]
AIRPLANE_NAMES[r".*7[7, 4]7.*"] = ["long_haul", "jumbo"]

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


class MetaDataPreprocessor:
    def __init__(self):
        self.i = 0

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

    @staticmethod
    def bin(rating):
        if not rating is np.nan:
            if rating <= 3:
                return 0
            elif rating > 3 and rating < 8:
                return 1
            else:
                return 2
        return rating

    @staticmethod
    def clean_body(body):
        if "Verified | " in body:
            body = body.split("Verified | ")[-1]
        elif "Review | " in body:
            body = body.split("Review | ")[-1]
        return body

    def preprocess_row(self, row, cities_in_reviews):
        if self.i % 10_000 == 0:
            logger.debug(f"Parsing review {self.i}")
        self.i += 1
        row["body"] = self.clean_body(row["body"])
        row["bin"] = self.bin(row["rating"])
        upper_case_body = " ".join(re.findall(r"[A-Z][a-z]*", row["body"]))
        cities = []
        countries = []
        for city, country in cities_in_reviews.items():
            if city in upper_case_body:
                cities.append(city)
                countries.append(country)
        row["cities"] = cities
        row["countries"] = list(set(countries))
        return row

    def preprocess(self, df):
        with open(
            os.path.join("..", "data", "resources", "cities_and_countries.json"), "rb"
        ) as f:
            cities_in_reviews = json.load(f)

        logger.debug("Extracting airplane data")
        df = self.extract_airplane_data(df, AIRPLANE_NAMES)
        logger.debug("Extracting cities and countries")
        preprocessed_reviews = df.apply(
            lambda row: self.preprocess_row(row, cities_in_reviews), axis=1
        )
        preprocessed_reviews.to_pickle("preprocessed_reviews.pkl")
        return preprocessed_reviews

    def extract_airplane_data(
        self, data: pd.DataFrame, airplane_categories: OrderedDict
    ) -> pd.DataFrame:
        """Extracts the type of trip (long or short haul) and the size of the
        aircraft from the aircraft name information.

        Args:
            data (pd.DataFrame): original data. Must contain a column named
                'aircraft' containing the names of the aircraft associated to a
                review.
            airplane_categories (OrderedDict): keys are the regex pattern that
                identifies one aircraft type and the values are a list containing
                the associated trip type and aircraft size.

        Returns:
            pd.DataFrame: original data frame with two added columns (trip_type and
                size).
        """
        # standardize text
        data["aircraft_clean"] = (
            data.loc[data["aircraft"].notnull(), "aircraft"]
            .str.strip(" ")
            .str.replace(" ", "")
            .str.replace("-", "")
            .str.lower()
        )
        for pattern, (trip_type, size) in airplane_categories.items():
            mask = data["aircraft_clean"].apply(
                lambda x: bool(re.match(pattern, x)) if x is not np.nan else False
            )
            data.loc[mask, "trip_type"] = trip_type
            data.loc[mask, "size"] = size

        data.drop(columns="aircraft_clean", inplace=True)

        return data


if __name__ == "__main__":
    DATA_FOLDER = "../data/inputs"
    DATA1 = "seatguru_python_scraping.csv"
    DATA2 = "skytrax_scraping_2.csv"
    df1 = pd.read_csv(os.path.join(DATA_FOLDER, DATA1), index_col=0)
    df2 = pd.read_csv(os.path.join(DATA_FOLDER, DATA2), index_col=0)
    concat_df = pd.concat([df1, df2])
    concat_df.reset_index(inplace=True)
    logger.debug(f"\n{concat_df.head()}")
    meta_data_preprocessor = MetaDataPreprocessor()
    df_with_metadata = meta_data_preprocessor.preprocess(concat_df)
