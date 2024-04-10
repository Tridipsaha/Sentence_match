#install sentence_transformers-- pip install -q sentence_transformers


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

sentences = [
    ''' The sun sets in the west. ''',
    ''' Birds chirp merrily in the morning. ''',
    ''' Books open doors to new worlds. ''',
    ''' Reading books enriches the mind with wisdom. ''',
    ''' The sun rises in the east. '''
    
]

sentence_embeddings = model.encode(sentences)

pprint('first sentence= {}, second sentence= {}, score = {}'.format(sentences[0],sentences[4],
                                            cosine_similarity(sentence_embeddings[0].reshape(1,-1),
                                                              sentence_embeddings[4].reshape(1,-1))[0][0]))