# News Category Text Classification

One popular text classification application is used in digital advertising for contextual targeting. <br>
For contextual targeting, ads are placed based on the context of the news article, assuming that ads placed in a related context have a higher probability of being clicked on (higher click through rate). <br>

In this project we train several deep learning models to predict the news topic of an article from the raw text of the article. <br>

## Required Python Packages
- numpy
- pandas
- sklearn
- matplotlib
- seaborn
- tensorflow
- keras

## Data Used
This dataset used contains around 200k news headlines from the year 2012 to 2018 obtained from HuffPost. <br>
The raw dataset was downloaded from Kaggle. <br>

Refer to below link for more details:
<https://www.kaggle.com/rmisra/news-category-dataset> <br>

For model training, we will use the combined text from the headline + short description, to try to predict the news category out of 41 different categories.

For reference, below are a few samples from the dataset.

| category      | headline                                          | authors         | link                                              | short\_description                                | date      |
| ------------- | ------------------------------------------------- | --------------- | ------------------------------------------------- | ------------------------------------------------- | --------- |
| CRIME         | There Were 2 Mass Shootings In Texas Last Week... | Melissa Jeltsen | https://www.huffingtonpost.com/entry/texas-ama... | She left her husband. He killed their children... | 5/26/2018 |
| ENTERTAINMENT | Will Smith Joins Diplo And Nicky Jam For The 2... | Andy McDonald   | https://www.huffingtonpost.com/entry/will-smit... | Of course it has a song.                          | 5/26/2018 |
| ENTERTAINMENT | Hugh Grant Marries For The First Time At Age 57   | Ron Dicker      | https://www.huffingtonpost.com/entry/hugh-gran... | The actor and his longtime girlfriend Anna Ebe... | 5/26/2018 |
| ENTERTAINMENT | Jim Carrey Blasts 'Castrato' Adam Schiff And D... | Ron Dicker      | https://www.huffingtonpost.com/entry/jim-carre... | The actor gives Dems an ass-kicking for not fi... | 5/26/2018 |
| ENTERTAINMENT | Julianna Margulies Uses Donald Trump Poop Bags... | Ron Dicker      | https://www.huffingtonpost.com/entry/julianna-... | The "Dietland" actress said using the bags is ... | 5/26/2018 |

## Training

After combining the headline and short description columns and vectorizing the text, the 

After splitting the dataset into training set (90%) and test set (10%) as well as vectorizing the text, 6 different deep learning models were trained in Tenserflow Keras. <br>
During each training epoch, a random 10% of the training set was held out for validation. <br>
For comparison, the accuracy on the last training epoch validation set was considered. <br>


## Evaluation and model selection with cross-validation

Below performance was achieved by the different candidate models on training set validation portion on the last training epoch.

| Model                                                                              | Validation\_Loss | Validation\_Accuracy |
| ---------------------------------------------------------------------------------- | ---------------- | -------------------- |
| Model 1: Pre-trained sentence embeddings (nnlm-en-dim50) + Fully Connected Layers  | 3.058574         | 0.292803             |
| Model 2: Pre-trained sentence embeddings (nnlm-en-dim128) + Fully Connected Layers | 3.031723         | 0.304641             |
| Model 3: Pre-trained sentence embeddings (BERT) + Fully Connected Layers           | 3.213881         | 0.234193             |
| Model 4: Pre-trained word embeddings (BERT) + RNN (GRU) layers                     | 1.425814         | 0.657631             |
| Model 5: Embedding Layer (not pre-trained) + RNN (GRU) layers                      | 1.810337         | 0.590308             |
| Model 6: Embedding Layer (not pre-trained) + Bidirectional RNN (GRU) layers        | 1.794118         | 0.596504             |

The model selected with the best Validation performance was the Model 4 (Pre-trained word embeddings (BERT) + RNN (GRU) layers), with a validation accuracy of 65.7%. <br>

| Model                                                                              | Validation\_Loss | Validation\_Accuracy |
| ---------------------------------------------------------------------------------- | ---------------- | -------------------- |
| Model 4: Pre-trained word embeddings (BERT) + RNN (GRU) layers                     | 1.425814         | 0.657631             |


## Best model architecture

![Best Model (Model 4) Architecture](model_04.png)

## Results on out of sample test-set

The model selected with the best Validation performance was the Model 4 (Pre-trained word embeddings (BERT) + RNN (GRU) layers), with a validation accuracy of 65.7%. <br>

Below out of sample performance was achieved by the best model on the test set.

| Model                                                           | Test\_Accuracy       |
| --------------------------------------------------------------- | -------------------- |
| Model 4: Pre-trained word embeddings (BERT) + RNN (GRU) layers  | 0.667                |

The model selected with the best Validation performance was the Model 4 (Pre-trained word embeddings (BERT) + RNN (GRU) layers), with a test set accuracy of 66.7%. <br>


## Appendix

![](confusion_matrix_test_set.png)

