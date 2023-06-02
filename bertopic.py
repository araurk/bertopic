import pandas as pd
from bertopic import BERTopic

country = 'pl'

corpus = pd.read_csv(rf'C:\Users\agni\Desktop\linking\doc2vec\data\corpus_{country}.csv')
corpus.dropna(subset = ['text'], inplace = True)
corpus.reset_index(inplace = True, drop = True)

topic_model = BERTopic(language= 'multilingual', calculate_probabilities=True, verbose=True, nr_topics=15, min_topic_size = 50)

topics, probs = topic_model.fit_transform(corpus['text'])

# save model
topic_model.save(rf'C:\Users\agni\Desktop\linking\doc2vec\data\multilingual\models\{country}_corpus_topic_model')

# load model
topic_model = BERTopic.load(rf'C:\Users\agni\Desktop\linking\doc2vec\data\\multilingual\models\{country}_corpus_topic_model')

corpus = pd.read_csv(rf'C:\Users\agni\Desktop\linking\doc2vec\data\corpus_{country}.csv')
corpus.dropna(subset = ['text'], inplace = True)
corpus.reset_index(inplace = True, drop = True)

topics, probs = topic_model.fit_transform(corpus['text'])

# get topics
topics = topic_model.get_topics()

# save topics and probabilities
topics_and_probs = pd.DataFrame({'topics': topics})
for i in range(14):
    corpus[f'prob_{i}'] = [prob[i] for prob in probs]

corpus['prob_minus_1'] = [prob[-1] for prob in probs]

corpus.to_csv(rf'C:\Users\agni\Desktop\linking\doc2vec\data\multilingual\{country}_corpus_topics_and_probs.csv')
topics_and_probs.to_csv(rf'C:\Users\agni\Desktop\linking\doc2vec\data\multilingual\{country}_topicsdesc.csv')
