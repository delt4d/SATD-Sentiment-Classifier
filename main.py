import warnings
import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB as MultiNB, ComplementNB as ComplNB, CategoricalNB as CatNB, GaussianNB as GausNB, BernoulliNB as BerNB
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import make_pipeline, FunctionTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

BALANCED_DATA_CSV = 'data/balanced.csv'
ORIGINAL_DATA_CSV = 'data/original.csv'
TEST_CSV = 'data/test.csv'

warnings.filterwarnings('ignore')

class BaseNB:
    def __init__(self):
        self.gs = None

    def fit(self, x_train, y_train):
        self.gs.fit(x_train, y_train)

    def predict(self, x_test):
        return self.gs.predict(x_test)
    
    def score(self, x_test, y_test):
        return self.gs.score(x_test, y_test)

class MultinomialNB(BaseNB):
    def __init__(self):
        super().__init__()
        self.gs = GridSearchCV(
            make_pipeline(TfidfVectorizer(), MultiNB()), 
            param_grid={
                'multinomialnb__fit_prior': [True, False],  
                'multinomialnb__force_alpha': [True, False],
                'multinomialnb__alpha': [i/10 for i in range(1, 30)]
            }, return_train_score=True
        )

class ComplementNB(BaseNB):
    def __init__(self):
        super().__init__()
        self.gs = GridSearchCV(
            make_pipeline(TfidfVectorizer(), ComplNB()), 
            param_grid={
                'complementnb__fit_prior': [True, False],  
                'complementnb__force_alpha': [True, False],
                'complementnb__alpha': [i/10 for i in range(1, 30)],
                'complementnb__norm': [True, False],
            }, return_train_score=True
        )

class GaussianNB(BaseNB):
    def __init__(self):
        super().__init__()
        self.gs = GridSearchCV(
            make_pipeline(TfidfVectorizer(), FunctionTransformer(lambda x: x.toarray(), accept_sparse=True), StandardScaler(with_mean=False), GausNB()),  
            param_grid={}, return_train_score=True
        )


class CategoricalNB(BaseNB):
    def __init__(self):
        super().__init__()
        self.gs = GridSearchCV(
            make_pipeline(CountVectorizer(), FunctionTransformer(lambda x: x.toarray(), accept_sparse=True), KBinsDiscretizer(encode='ordinal', strategy='uniform'), CatNB()), 
            param_grid={
                'categoricalnb__fit_prior': [True, False],
                'categoricalnb__alpha': [i/10 for i in range(1, 30)]
            }, return_train_score=True
        )

class BernoulliNB(BaseNB):
    def __init__(self):
        super().__init__()
        self.gs = GridSearchCV(
            make_pipeline(TfidfVectorizer(), BerNB()), 
            param_grid={
                'bernoullinb__fit_prior': [True, False],  
                'bernoullinb__force_alpha': [True, False],
                'bernoullinb__alpha': [i/10 for i in range(1, 30)]
            }, return_train_score=True
        )

class TextClassifier:
    def __init__(self, naive_bayes_model_cls):
        self.labelEncoder = LabelEncoder()
        self.model = naive_bayes_model_cls()

    def _transform_text(self, text_array):
        def to_lower(text):
            return text.lower()
        
        def remove_spaces(text):
            return " ".join(text.split()).strip()
        
        def sanatize(text):
            return re.sub('[./:;$@&*\'"]', '', text) 
        
        return [to_lower(remove_spaces(sanatize(text))) for text in text_array]

    def train(self, csv_file, test_size=0.3):
        df = pd.read_csv(csv_file, header=0)

        x = self._transform_text(df[df.columns[0]])
        y = self.labelEncoder.fit_transform(df[df.columns[1]])

        xtrain, xtest, ytrain, ytest = x, x, y, y
        if (test_size>0):
            xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=test_size)
        
        self.model.fit(xtrain, ytrain)

        print("Training complete.")
        print("Best parameters:", self.model.gs.best_params_)
        print("Score on validation set:", self.model.score(xtest, ytest))

        return self.model.gs.best_params_, self.model.score(xtest, ytest)

    def test(self, csv_file):
        if not self.model.gs:
            raise ValueError("Model has not been trained yet.")

        df = pd.read_csv(csv_file, header=0)
        x_test = self._transform_text(df[df.columns[0]])
        y_test = df[df.columns[1]]

        y_pred = self.model.predict(x_test)

        print("Testing complete.")
        print("Best parameters:", self.model.gs.best_params_)
        print("Score on test set:", self.model.score(x_test, self.labelEncoder.transform(y_test)))

        table = pd.DataFrame(
            zip(x_test, [self.labelEncoder.classes_[result] for result in y_pred], y_test),
            columns=["Input", "Prediction", "Expected"]
        )

        accuracy = accuracy_score(self.labelEncoder.transform(y_test), y_pred)
        precision = precision_score(self.labelEncoder.transform(y_test), y_pred, average="micro")
        recall = recall_score(self.labelEncoder.transform(y_test), y_pred, average="macro")
        f1score = f1_score(self.labelEncoder.transform(y_test), y_pred, average="macro")

        print("Results: ")
        print("Accuracy = " + str(accuracy))
        print("Precision = " + str(precision))
        print("Recall = " + str(recall))
        print("F1-Score = " + str(f1score))

        return table, confusion_matrix(self.labelEncoder.transform(y_test), y_pred)

import json
import os

CHOICES_FILE = 'previous_choices.json'

def load_previous_choices():
    if os.path.exists(CHOICES_FILE):
        with open(CHOICES_FILE, 'r') as file:
            return json.load(file)
    return {
        'nb_choice': None,
        'train_full': None,
        'data_choice': None
    }

def save_previous_choices(nb_choice, train_full, data_choice):
    choices = {
        'nb_choice': nb_choice,
        'train_full': train_full,
        'data_choice': data_choice
    }
    with open(CHOICES_FILE, 'w') as file:
        json.dump(choices, file)

def get_user_input():
    previous_choices = load_previous_choices()

    if previous_choices['nb_choice'] is not None:
        reuse_choice = input("\nDo you want to reuse your previous choices?\n1. Yes (default)\n2. No\nEnter the number (1/2): ").strip()
        
        if reuse_choice != '2':
            return previous_choices['nb_choice'], previous_choices['train_full'], previous_choices['data_choice']

    nb_choice = input("\nChoose the Naive Bayes variant to use:\n1. Multinomial\n2. Complement\n3. Categorical\n4. Gaussian\n5. Bernoulli\nEnter the number (1/2/3/4/5): ").strip()
    full_data_choice = input("\nDo you want to train with full data?\n1. Yes\n2. No (70% training, 30% testing)\nEnter the number (1/2): ").strip()
    data_choice = input("\nWhich dataset do you want to use?\n1. BALANCED\n2. ORIGINAL\nEnter the number (1/2): ").strip()

    save_previous_choices(nb_choice, full_data_choice == '1', data_choice)

    return nb_choice, full_data_choice == '1', data_choice

def main():
    nb_choice, train_full, data_choice = get_user_input()

    if nb_choice == '1':
        NBVariantCls = MultinomialNB
    elif nb_choice == '2':
        NBVariantCls = ComplementNB
    elif nb_choice == '3':
        NBVariantCls = CategoricalNB
    elif nb_choice == '4':
        NBVariantCls = GaussianNB
    elif nb_choice == '5':
        NBVariantCls = BernoulliNB
    else:
        print("Invalid choice for Naive Bayes variant. Exiting...")
        return

    classifier = TextClassifier(NBVariantCls)
    csv_file = BALANCED_DATA_CSV if data_choice == '1' else ORIGINAL_DATA_CSV

    if train_full:
        print("\nTraining with full data...")
        classifier.train(csv_file, test_size=0)  
    else:
        print("\nTraining with 70% data...")
        classifier.train(csv_file, test_size=0.3) 

    print("Testing the model...")
    results = classifier.test(TEST_CSV)
    print(results)
    
main()
