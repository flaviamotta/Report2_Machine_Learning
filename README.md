# Report2_Machine_Learning
Movie Rating Prediction using Text Analysis and Machine Learning

This project focuses on predicting movie ratings based on text analysis of movie synopses. We explore various machine learning algorithms, including KNN, random forests, boosting, lasso regression, and kernel ridge regression, to build predictive models. The goal is to leverage the content of movie synopses to estimate the ratings they will receive.

Key Findings:
- The lasso regression model outperformed other models in terms of estimated risk and standard error, demonstrating its effectiveness in rating prediction.
- The importance of specific terms in the synopses was most accurately captured by the lasso model, providing insights into which genres of movies were associated with higher or lower ratings.
- The addition of drop-out improved the performance of neural networks in rating prediction, indicating its potential for enhancing model accuracy.
- Visual analysis of predicted versus observed values revealed that the KNN and random forests models achieved better alignment with the identity line, although further improvements are desirable for all models.

Implementation Details:
- Data set avaiable at (https://www.dropbox.com/s/6ltw600uoiynd3t/TMDb_updated.CSV.zip?dl=0).
- For the construction of the term-document matrix, terms that appeared in less than 50 evaluations were disregarded.
- The dataset was split into training (60%), validation (20%), and test (20%) sets, considering the size of the dataset after removing observations without evaluations.
- Different machine learning algorithms were employed, including KNN, random forests, boosting, lasso regression, and kernel ridge regression.
- Evaluation metrics such as estimated risk, standard error, and importance of covariates were used to assess model performance.

Overall, this project provides valuable insights into the prediction of movie ratings using text analysis and machine learning techniques. The findings contribute to understanding the influence of specific terms in movie synopses on ratings.
