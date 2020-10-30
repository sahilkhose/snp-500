Things with loopholes:
- The bert embeddings of articles are clipped.
    - Could be loosing important information.
- Normal average instead of a weighted average.
- We zero out the node emb for stocks which are not present in the article on a particular day, loosing the dependency learning of different stocks 
- News articles contains unecessary noise, stock emb used only titles.

To work on:
1. Weighted Average + Classifier Sharing
    - Market vector will substitute the node embedding directly
    - Create stock embedding. 
    - Use same bert embedding for n_value and n_key 
    - 
2. Title
3. Dual representation

Stock Embedding paper:
- Calculate attention score for every article i published on day t
- For every article i:
    - score(i, j) = n_key(i) . s(j)
    - i: article
    - j: stock
- alpha(i, j) = softmax(score(i, j)) over i (articles)
- market_vector(j) = sum(alpha(i, j) . n_value(i))  for a day t
- n_key = word2vec on news corpus (d=64)
- n_value = bert google research (o/p = 1024dim) -> pca 256

Doubts:
- How did stock emb have minibatch 64 even though using GRU and shared classifier