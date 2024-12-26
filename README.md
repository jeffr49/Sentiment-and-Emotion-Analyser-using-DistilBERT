# Sentiment-and-Emotion-Analyser-using-DistilBERT

# Overview

This project is a Sentiment Analysis Web Application designed to analyze the sentiment of input text and detect emotions. It uses a trained sentiment analysis model to classify text into positive, neutral, or negative categories with associated probabilities. Additionally, the application includes emotion detection capabilities, identifying the dominant emotion in the input text. This web app provides an easy-to-use interface for users to input text and view the sentiment analysis and emotion detection results in real-time.

# Dataset
[Dataset for training the sentiment model](https://drive.google.com/file/d/1AUcoTg2S8nUC3kDLOgNReAGXP1cXy2gR/view?usp=sharing)

# Trained Model
[Sentiment Model Files](https://drive.google.com/drive/folders/1YEmm0SniMUmwC--PedwjBorrZZTkhVnJ?usp=sharing)

# Live Working Explanation
[Presentation of the Working Model](https://github.com/jeffr49/Sentiment-and-Emotion-Analyser-using-DistilBERT/blob/main/web_workingmodel.pptx)

# Features
Sentiment Analysis  
Classifies text into three categories: Positive, Neutral, and Negative.  
Displays associated probabilities for each category.  
Emotion Detection  
Identifies the dominant emotion in the input text using a pre-trained emotion detection model.  
Supports multiple emotion categories, enhancing the depth of text analysis.  
Real-time Results  
Users can see analysis results immediately after entering text.  
Interactive UI  
Clean and user-friendly interface with a responsive layout.  
Navigation and Results Display  
Back button for easy navigation.  
Results display the input text in bold, the sentiment classification, probabilities, and the detected emotion.  

# Why Use This?

Sentiment Analysis Tool  
Customer Feedback Analysis: Analyze reviews, comments, or feedback to determine customer sentiment.  
Social Media Sentiment Tracking: Monitor the sentiment of posts or tweets on specific topics or brands.  
Text Analytics: Gain insights from large amounts of text data by automatically classifying sentiments.  
Emotion Detection Tool  
Enhanced Insights: Go beyond sentiment and understand the emotional undertone of textual data.  
Use in Marketing: Tailor strategies based on customers' emotional responses.  
With real-time results and clear presentation, this tool helps users make data-driven decisions with ease.  

# Why We Built This?

The need for sentiment and emotion analysis tools is growing rapidly as companies and individuals strive to understand public opinion and feedback in an efficient way. With the massive amounts of text data produced daily, it's important to have a tool that quickly identifies both sentiment and emotions without manually reading each piece of content.

This project was built to provide an accessible web interface for sentiment and emotion analysis, making it usable by a wide range of users without requiring expertise in machine learning. By combining NLP (Natural Language Processing) with a sleek web interface, the app helps users gain deeper insights from textual data.

# Deployment Instructions

# Prerequisites

Before running the application, make sure you have the following installed:

Python 3.x (preferably the latest version)

Flask: Python web framework to run the server

PyTorch (for the sentiment and emotion analysis models)

Transformers library (for loading pre-trained models)

scikit-learn (if you're using additional tools for model preprocessing)

Numpy and Pandas (for data manipulation)

# Final Training Metrics (Sentiment Analysis)
Training Loss: 0.178  
Validation Loss: 0.470  
Accuracy: 86.5%  

You can fork this repository and use it for your own projects or further customizations. This tool provides both sentiment and emotional insights, making it an invaluable resource for understanding textual data.



