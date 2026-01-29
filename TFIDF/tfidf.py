import numpy as np

##to match our implementation from that on sklearn, we need regex to take just all the letters (no punctutaions), remove the stop words

class tfidf_vecotrizer():
    def __init__(self,smoothed=True,norm='l2',lower=True) -> None:
        self.smoothed = smoothed
        self.norm = norm
        self.lower = lower
        self.tf = None
        self.df = None
        self.tfidf = None
        self.eps = 1e-9

    def _get_unique_words(self,x:list[str]) -> tuple[list[str],dict]:
        unique = []
        seen = set()
        for docs in x:
            words = docs.split() if not self.lower else docs.lower().split()
            for w in words:
                if w not in seen:
                    seen.add(w)
                    unique.append(w.lower())
        word2idx = {word:idx for idx,word in enumerate(unique)}
        return unique,word2idx
    
    def fit(self,x:list[str]):
        unique_words,word2idx = self._get_unique_words(x)

        self.tf = np.zeros((len(x),len(unique_words)))

        for doc_idx,doc in enumerate(x):
            for word in doc.split():
                if self.lower:
                    self.tf[doc_idx][word2idx[word.lower()]] += 1
                else:
                    self.tf[doc_idx][word2idx[word]] += 1

        self.df = np.sum(self.tf != 0,axis=0)

        idf = np.log(len(x)/self.df) + 1 if not self.smoothed else np.log((len(x)+1)/(self.df+1)) + 1

        self.tfidf = self.tf * idf
        
        if self.norm.lower() == 'l2':
            norm = np.linalg.norm(self.tfidf,axis=1,keepdims=True)
            self.tfidf = self.tfidf/(norm+self.eps)

        return self