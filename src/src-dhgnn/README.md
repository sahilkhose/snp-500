Things with loopholes in master:
- The bert embeddings of articles are clipped.
    - Could be loosing important information.
- Normal average instead of a weighted average.
- We zero out the node emb for stocks which are not present in the article on a particular day, loosing the dependency learning of different stocks 
- News articles contains unecessary noise, stock emb used only titles.

DHGNN idea:
- Use price information as node emb.
- Use news article as hyperedge emb.

TO-DO:
- Read DHGNN paper.
- Understand imoonlab code.
- Make changes in our code and train.