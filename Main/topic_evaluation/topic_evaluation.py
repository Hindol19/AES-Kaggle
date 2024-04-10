# Topic evaluation step is done using Latent Dirichilt Allocation [LDA]
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.utils import simple_preprocess
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# Step 0: Read CSV
df = pd.read_csv("Main\\input_data\\train.csv")
df = df.drop(columns=['essay_id'], axis=1)

essays = []

for i in range(0, len(df)-1):
    print(df['full_text'].iloc[i])
    # print(df['full_text'].tail(5))

'''
# Step 1: Preprocessing
# Assume essays is a list of strings where each string represents an essay
processed_essays = [simple_preprocess(essay) for essay in essays]

# Step 2: Create dictionary and corpus
dictionary = corpora.Dictionary(processed_essays)
corpus = [dictionary.doc2bow(essay) for essay in processed_essays]

# Step 3: TF-IDF Vectorization
tfidf = TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

# Step 4: LDA Model Training
num_topics = 5  # Number of topics to identify
lda_model = LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary)

# Step 5: Topic Relevance Calculation
# Assume essay_scores is a list of scores corresponding to each essay
essay_scores = [0.8, 0.6, 0.9, 0.7, 0.5]  # Example scores

topic_relevance_scores = []

for i, essay in enumerate(processed_essays):
    # Get the topic distribution for the essay
    topic_distribution = lda_model.get_document_topics(
        corpus_tfidf[i], minimum_probability=0.0)
    topic_distribution = np.array([prob for _, prob in topic_distribution])

    # Compute the cosine similarity between the essay's topic distribution and the scores
    similarity_score = cosine_similarity(
        [topic_distribution], [essay_scores])[0][0]
    topic_relevance_scores.append(similarity_score)

# Print or use the topic relevance scores
print("Topic Relevance Scores:", topic_relevance_scores)
'''
