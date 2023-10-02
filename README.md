# Heart Disease Detection using Machine Learning

This project aims to predict potential heart diseases in people using Machine Learning algorithms. The algorithms include K-Neighbors Classifier, Support Vector Classifier and Decision Tree Classifier. The dataset has been taken from [UCI Machine Learning Repository](^1^).

## Data preprocessing

The dataset contains 14 features and 303 instances. The features are:

- age: age in years⁴[4]
- sex: (1 = male; 0 = female)
- cp: chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic)⁵[5]
- trestbps: resting blood pressure (in mm Hg on admission to the hospital)
- chol: serum cholestoral in mg/dl⁶[6]
- fbs: (fasting blood sugar > 120 mg/dl) (1: true; 0: false)⁷[7]
- restecg: resting electrocardiographic results (0: normal, 1: having abnormality, 2 = showing ventricular hypertrophy)⁸[8]
- thalach: maximum heart rate achieved⁹[9]
- exang: exercise induced angina (1: yes, 0: no)[^10^][10]
- oldpeak: ST depression induced by exercise relative to rest¹¹[11]
- slope: the slope of the peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping)¹²[12]
- ca: number of major vessels (0-3) colored by flourosopy
- thal: thalassemia(a blood disorder) (3 = normal; 6 = fixed defect; 7 = reversable defect)¹³[13]
- target: heart disease (0: no, 1: yes)

The data preprocessing steps are:

- There are no missing values in the dataset, so no imputation or deletion is needed¹⁴[14].
- Some features are categorical but represented by decimal values, such as cp, thal and slope¹⁵[15]. These features are converted to one-hot encoded variables using pandas.get_dummies() function.
- The non-categorical features are normalized by mapping them to [0, 1] range using sklearn.preprocessing.MinMaxScaler() function.

## Data inspection and visualization

The correlation matrix of features and the target is plotted using seaborn.heatmap() function. The histogram plots of some features and the target are plotted using matplotlib.pyplot.hist() function. The plots show the distribution of data and the relationship between features and the target.

## Models and their performances

Three models are used to classify the data: Support Vector Classifier (SVC), K-Neighbors Classifier (KNN) and Decision Tree Classifier (DTC). The models are imported from sklearn.svm, sklearn.neighbors and sklearn.tree modules respectively. The models are trained on 80% of the data and tested on 20% of the data. The performance metrics are accuracy and recall, which are calculated using sklearn.metrics.accuracy_score() and sklearn.metrics.recall_score() functions. The confusion matrices of the models are plotted using sklearn.metrics.plot_confusion_matrix() function.

The results of the models are:

| Model | Accuracy | Recall |
| ----- | -------- | ------ |
| SVC   | 0.885    | 0.885  |
| KNN   | 0.885    | 0.885  |
| DTC   | 0.787    | 0.787  |

The SVC and KNN models have similar and better performance than the DTC model. The SVC model uses the radial basis function (rbf) kernel with gamma set to 0.3, which is selected by cross-validation. The KNN model uses 7 neighbors, which is selected by trial and error. The DTC model uses the default parameters and does not perform pruning, which may cause overfitting.

## References

- [UCI Machine Learning Repository: Heart Disease Dataset](^1^)
- [SVM Tutorial](^2^)
- [Decision Tree Tutorial](^3^)
- [KNN Tutorial](^4^)
