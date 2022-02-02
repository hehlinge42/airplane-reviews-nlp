# airplane-reviews-nlp


# EDA
## Notebooks
We used exploration notebooks to extract quick insights from the data to then inform our next steps.


# Data Pipeline
## 1) Metadata preprocessor
We first extract relevant information about the flight such as the airline, the aircraft model, arrival, and departure from the reviews.

```
src/meta_data_preprocessor.py
```

## 2) Text preprocessor
We apply NLP preprocessing steps such as removing stop words, lemmatization, change accents, decontract words, change to lower case, and create n grams.

```
src/text_preprocessor.py
```

## 3) Embeddor
We apply a pre-trained Word2Vec model to embed the preprocessed n grams in vectors. We also add the option to apply a dimesion reduction technique, PCA,to condense information. 

```
src/embeddor.py
```

# Next steps
## 1) Streamlit application
Our next step would add a streamlit app that aggregates the most insightful visualizations.

## 2) Automated scrapper
We would also like to implement an automated scrapper to extract fresh data from the review website so that our client can stay up to date with topic trends and flyer satisfaction.