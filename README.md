# Cs439-FOMC
Predictive model, to predict market movement after FOMC meeting, based on FOMC meeting. 

Here's a summary of what each file does:

frequency_analysis:
Performs text preprocessing and word frequency analysis on FOMC speeches
Creates visualizations:
Top 20 most frequent words
Word frequency distribution
Word frequencies by market reaction (Up/Down)
Generates two CSV files:
word_frequencies.csv: Basic word frequency counts
word_frequencies_with_market_reaction.csv: Extended analysis with market reaction correlations

predictive_model:
Implements a machine learning model using:
TF-IDF text features
Speech length
Word frequency features from the frequency analysis
Uses Random Forest Classifier
Includes functions for training and prediction
Saves the trained model as fomc_predictor.joblib

tester_predictive_model:
Tests the model on historical FOMC speeches
Generates a comprehensive report showing:
Overall accuracy (88.89%)
Prediction results by market direction
Detailed results saved in prediction_results.csv

current_fomc_predictor:
Simple interface to predict market reaction for new FOMC speeches
Takes speech text as input
Returns predicted market direction and confidence score
The model shows good performance with an accuracy of 88.89% on the test data. The confusion matrix shows that it performs well for both "Up" and "Down" predictions.

To use the current FOMC predictor for new speeches:
