import string
import json
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import numpy as np
import gensim.downloader as api


def preprocess_text(text):
    tokens = text.split()
    tokens = [word.lower().translate(str.maketrans('', '', string.punctuation)) for word in tokens]
    
    
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def retrieve_response(query):
    glove_model = api.load("glove-wiki-gigaword-100")

    with open('Embeddings/english_embeddings.json') as f:
        farm_data = json.load(f)
    farm_dict = {}

    for entry in farm_data:
        key, value = entry.split(':', 1)
        farm_dict[key.strip()] = value.strip()
    # Sample crop descriptions
    crop_info = list(farm_dict.values())

    

    # Sample crop descriptions
    crop_info = list(farm_dict.values())

    # Preprocess and calculate document embeddings
    document_embeddings = []

    for document in crop_info:
        tokens = preprocess_text(document).split()
        word_vectors = [glove_model[word] for word in tokens if word in glove_model]
        
        if word_vectors:
            doc_embedding = np.mean(word_vectors, axis=0)
            document_embeddings.append(doc_embedding)
        else:
            # Handle cases where no valid word vectors are found
            document_embeddings.append(np.zeros(100))  # Assuming 100-dimensional vectors

    # Prepare the query and calculate its embedding
    query = "Can you tell me the varieties of onions?"
    query_embedding = np.mean([glove_model[word] for word in preprocess_text(query).split() if word in glove_model], axis=0)

    # Calculate cosine similarities
    similarities = cosine_similarity([query_embedding], document_embeddings)

    # Find the most similar crop
    most_similar_idx = np.argmax(similarities)
    most_similar_crop = list(farm_dict.keys())[most_similar_idx]
    similarity_score = similarities[0, most_similar_idx]



    return crop_info[most_similar_idx]