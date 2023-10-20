"""
Helper file containing functions for getting the most similar prompt.
"""
from typing import Dict, Tuple
import string
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords


ENGLISH_EMBEDDING_FILE = "../docs/english_embeddings.json"
URDU_EMBEDDING_FILE = "../docs/urdu_embeddings.json"


def read_data(file_path: str) -> Dict[str, str]:
    """This function will read the data and return it.

    Args:
        file_path (str): The file path where the prompt data is.

    Returns:
        Dict[str, str]: The resultant dict.
    """
    with open(file_path, "r", encoding="utf-8") as _f:
        farm_data = json.load(_f)

    farm_dict = {}

    for entry in farm_data:
        key, value = entry.split(":", 1)
        farm_dict[key.strip()] = value.strip()

    return farm_dict


def preprocess_text(text):
    """
    Function for preprocessing text before finding TF-IDF
    """
    tokens = text.split()
    tokens = [
        word.lower().translate(str.maketrans("", "", string.punctuation))
        for word in tokens
    ]

    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)


def get_most_similar_en(query: str) -> Tuple[str, float]:
    """This function will return the most similar prompt from embedding file.

    Args:
        query (str): The string to find similarity with.

    Returns:
        Tuple[str, float]: The most similar prompt and its similarity.
    """

    farm_dict = read_data(ENGLISH_EMBEDDING_FILE)

    crop_info = list(farm_dict.values())

    tfidf_vectorizer = TfidfVectorizer(preprocessor=preprocess_text)

    tfidf_matrix = tfidf_vectorizer.fit_transform(crop_info)

    query_tfidf = tfidf_vectorizer.transform([preprocess_text(query)])

    similarities = cosine_similarity(query_tfidf, tfidf_matrix)

    most_similar_idx = similarities.argmax()
    most_similar_crop = list(farm_dict.keys())[most_similar_idx]
    similarity_score = similarities[0, most_similar_idx]

    return farm_dict[most_similar_crop], similarity_score


def get_most_similar_ur(query: str) -> Tuple[str, float]:
    """This function will return the most similar prompt from embedding file.

    Args:
        query (str): The string to find similarity with.

    Returns:
        Tuple[str, float]: The most similar prompt and its similarity.
    """

    farm_dict = read_data(URDU_EMBEDDING_FILE)

    crop_info = list(farm_dict.values())

    tfidf_vectorizer = TfidfVectorizer(preprocessor=preprocess_text)

    tfidf_matrix = tfidf_vectorizer.fit_transform(crop_info)

    query_tfidf = tfidf_vectorizer.transform([preprocess_text(query)])

    similarities = cosine_similarity(query_tfidf, tfidf_matrix)

    most_similar_idx = similarities.argmax()
    most_similar_crop = list(farm_dict.keys())[most_similar_idx]
    similarity_score = similarities[0, most_similar_idx]

    return farm_dict[most_similar_crop], similarity_score
