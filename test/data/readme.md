The word2vec text and binary files are GenSim data generated with python:

```python
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_cou\
nt=1, workers=4)
model.wv.save_word2vec_format("word2vec.bin", binary=True)
model.wv.save_word2vec_format("word2vec.txt", binary=False)
```