# Latent Semantic Analysis Preprocessing (Tf-Idf, PCA)
A preprocessing pipeline for latent semantic analysis is defined which performs term frequency-inverse document frequency (Tf-Idf) weighting and then dimensionality reduction via principal component analysis (PCA). This pipeline is then demonstrated using two sample categories from the native Scikit learn dataset, '20newsgroups'.
## Overview
For this example we'll create a complete pipeline using the python machine learning library scikit learn. The pipeline will consist of 3 major steps. First we'll import our sample text dataset and convert it into a matrix of token counts using the CountVectorizer implementation. Then we'll convert the count matrix into normalized term frequency-inverse document frequency representation using the Tfidftransformer function.
## Term Frequency-Inverse Document Frequency Weighting (Tf-Idf)
The first step in the latent semantic analysis pipeline involves weighting terms in order to determine the importance of each term, in each document, of a collection. We accomplish this using the term frequency-inverse document frequency method. Recall that this method produces vectors which are often large and need to be reduced in order to implement further operations like clustering and classification.
## Dimensionality Reduction via Principal Component Analysis (PCA)
Dimensionality reduction is accomplished via PCA on the Tf-Idf vector space. We apply principal component analysis to reduce the tf-idf representation into a 2 dimensional approximation. The resulting 2D representation allows us to visualize our dataset much more easily and we demonstrate that by creating a couple plots.
## Full Code with 2-D Feature Space Representation
![alt text](https://github.com/edwardsta/latent-semantic-analysis/blob/master/TF-IDF_PCA.PNG)
