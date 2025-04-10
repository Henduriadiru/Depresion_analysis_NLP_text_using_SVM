# Depresion_analysis_NLP_text_using_SVM
This project uses Natural Language Processing (NLP) techniques to analyze tweets and predict signs of depression. The model utilizes Word2Vec embeddings and a classification algorithm (SVM) to detect depressive sentiment from text. And finding the best parameters of SVM using gridsearch.

🧠 Depression Text Analysis with NLP

A machine learning project that classifies text related to depression using Natural Language Processing (NLP). The app includes data visualization (PCA and t-SNE) and sentiment prediction using a custom-trained Word2Vec + SVM pipeline.

---

## 🚀 Features

- 🔍 **Data Preprocessing**: Cleaning and tokenization of tweets.
- 🧠 **Word2Vec Embedding**: Custom-trained Word2Vec model for text representation.
- 📊 **Dimensionality Reduction**: PCA and t-SNE visualizations of text clusters.
- 🧪 **Sentiment Classification**: Simple sentiment classifier with placeholder for future model improvement.
- 🌐 **Interactive UI** *(optional)*: Ready to integrate with Streamlit for real-time user input and visualization.

---
## 🧪 Sample Dataset

The dataset should contain two columns:
- `tweet` — Raw tweet text in data repo. You can use sample data. Its cleaned data and i added tokens column
- `label` — `1` for depressed sentiment, `0` for non-depressed

---

---
Example data 

	tweet	label	cleaned_tweet
0	the real reason why you be sad you be attach t...	1	[real, reason, sad, attach, people, distant, p...
1	my biggest problem be overthinking everything	1	[biggest, problem, overthinking]
2	the worst sadness be the sadness you have teac...	1	[worst, sadness, sadness, teach, hide]
3	i cannot make you understand i cannot make any...	1	[make, understand, make, understand, happen, i...
4	i do not think anyone really understand how ti...	1	[think, really, understand, tire, act, okay, s...
---
