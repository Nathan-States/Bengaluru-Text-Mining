### Import Libraries ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

### Read CSV ###
df = pd.read_csv("bengaluru-grievances.csv")

pd.DataFrame(df.category.unique()).values

# Reduce Character Limit  
character_limit = (df["description"].str.len() > 12)
df = df.loc[character_limit]

# Select Random 15,000 Rows 
df = df.sample(15000, random_state = 1).copy()

df['category_id'] = df["category"].factorize()[0]
category_id_df = df[["category", "category_id"]].drop_duplicates()

category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[["category_id", "category"]].values)

### Text Preprocessing ### 
tfidf = TfidfVectorizer(
    sublinear_tf = True, 
    min_df = 5,
    ngram_range = (1, 2), 
    stop_words = 'english'
)

# Default Variables 
features = tfidf.fit_transform(df.description).toarray()
labels = df.category_id
N = 3

# For Loop for Most Common Phrases 
for category, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names_out())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("\n==> %s:" %(category))
  print("  * Most Correlated Unigrams are: %s" %(', '.join(unigrams[-N:])))
  print("  * Most Correlated Bigrams are: %s" %(', '.join(bigrams[-N:])))

### Create Base Models ###

# Variables   
X = df["description"]
y = df["category"]

# Test and Train 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Store Models in List 
models = [
    RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state = 0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state = 0)    
]

# Cross Validation Defaults 
CV = 5
cv_df = pd.DataFrame(index = range(CV * len(models)))

# Create Models 
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
    
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

# Get Accuracy and Standard Deviation 
mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
std_accuracy = cv_df.groupby('model_name').accuracy.std()

acc = pd.concat([mean_accuracy, std_accuracy], axis= 1, ignore_index=True)
acc.columns = ['Mean Accuracy', 'Standard deviation']
acc

### Improving LinearSVC Model ###

# First Confusion Matrix 
model = LinearSVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize = (8,8))
sns.heatmap(conf_mat, annot=True, cmap="Greens", fmt='d',
            xticklabels=category_id_df.category.values, 
            yticklabels=category_id_df.category.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("Confusion Matrix for LinearSVC\n", size=16)

# Reread in Data 
df = pd.read_csv("bengaluru-grievances.csv")
pd.DataFrame(df.category.unique()).values 
character_limit = (df["description"].str.len() > 12)
df = df.loc[character_limit]
df = df.sample(15000, random_state = 2).copy()

# Convert Columns to 'Others'
df["category"] = df["category"].replace({'Markets': 'Others'})
df["category"] = df["category"].replace({'Estate': 'Others'})
df["category"] = df["category"].replace({'Welfare Schemes': 'Others'})
df["category"] = df["category"].replace({'Education': 'Others'})
df["category"] = df["category"].replace({'Town Planning': 'Others'})

# Convert Columns to 'Road Maintenance(Engg)'
df["category"] = df["category"].replace({'Optical Fiber Cables (OFC)': 'Road Maintenance(Engg)'})
df["category"] = df["category"].replace({'Storm  Water Drain(SWD)': 'Road Maintenance(Engg)'})

# Add Stop Words 
from sklearn.feature_extraction import text

stop_words = text.ENGLISH_STOP_WORDS.union(["please", "plz", "look", "help", "causing", "walk", "pedestrians", "kindly", "refused", "senior", "help", "one", "two", "three", "77", "1", "2", "3"])

tfidf = TfidfVectorizer(
    sublinear_tf = True, 
    min_df = 5,
    ngram_range = (1, 2), 
    stop_words = stop_words
)

# Rebuild Model
df['category_id'] = df["category"].factorize()[0]
category_id_df = df[["category", "category_id"]].drop_duplicates()
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[["category_id", "category"]].values)

features = tfidf.fit_transform(df.description).toarray()
labels = df.category_id
X = df["description"]
y = df["category"]

X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(features, 
                                                               labels, 
                                                               df.index, test_size=0.25, 
                                                               random_state=2)
model = LinearSVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Results 
acc_svc = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
cv_mean_accuracy_svc = acc_svc.mean()
cv_mean_std_svc = acc_svc.std()
print(cv_mean_accuracy_svc * 100)
print(cv_mean_std_svc * 100)

# Second Confusion Matrix 
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize = (8,8))
sns.heatmap(conf_mat, annot=True, cmap="BuPu", fmt='d',
            xticklabels=category_id_df.category.values, 
            yticklabels=category_id_df.category.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("Confusion Matrix for LinearSVC\n", size=16)
