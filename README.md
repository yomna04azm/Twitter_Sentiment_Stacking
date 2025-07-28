# Sentiment Analysis Classifier (BERT + Stacking)

This project provides a trained sentiment analysis model that classifies text data (e.g., tweets or reviews) into positive and negative. It leverages BERT for deep contextual embeddings and a stacking ensemble classifier (SVM + XGBoost + Logistic Regression) for robust prediction.

---

##  Features

-  Pre-trained model on 20,000 labeled tweets
-  Uses `bert-base-uncased` for text embeddings
-  StackingClassifier with SVM and XGBoost as base models
-  Easy to load and use for inference
-  Supports classification of noisy social media content (hashtags, mentions, URLs)

---

## Download the Trained Model

You can download the trained model (`sentiment_stack_model.pkl`) here:

[Click to download sentiment_stack_model.pkl](https://drive.google.com/file/d/1y29grIZ6HI-v9OaEEbLhgGEswoAbCYz-/view?usp=sharing)
