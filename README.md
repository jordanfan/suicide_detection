
# Suicide Comments Detection 

### Contributors 
Erin Smith, Jordan Fan, Shaki Pothini

## About / Synopsis 

Model detecting whether a comment contains themes of suicide using sentiment analysis, TF-IDF, and Word/Document Embeddings. The models were trained on the Suicide and Depression Detection kaggle dataset in which comments classified as suicide originate from the “SuicideWatch” subreddit while all other comments are classified as non-suicide. 

## Data Sources
* Kaggle dataset - https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch
* Internet slang translation - https://floatcode.wordpress.com/2015/11/28/internet-slang-dataset/

## Notebooks

* 1_slang_spellcheck.ipynb - converts slang word translation and performs spellcheck on text
* 2_data_processing.ipynb - translate contractions, remove stop words & punctuations, lemmatization, POS tagging, chunking/chinking
* 3_eda.ipynb - explore text lengths, word frequencies, and bigrams
* 4a_sentiment.ipynb - sentiment analysis with VADER and logistic regression model implementation
* 4b_tf_idf.ipynb - TF-IDF implementation and random forest model
* 4c_embeddings.ipynb - Average Pooling Word Embedding and Doc2Vec Embedding implementation
* 4c_embeddings_1D_cnn.py - 1D CNN Word Embedding implementation
* 5_embeddings_model.ipynb - Logistic Regression, Neural Network, and Gradient Boosted Trees on embeddings implementation 
* 6_model_blending.ipynb - blend all model outputs with Gradient Boosted Trees
