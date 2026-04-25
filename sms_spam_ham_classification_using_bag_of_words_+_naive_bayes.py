# -*- coding: utf-8 -*-
"""
SMS Spam/Ham Classification — Naive Bayes + Bag-of-Words
Author: David Adom Dokyi

FIXES APPLIED:
  1. Robust column selection using usecols instead of blind iloc[:, :2]
  2. y_train sliced with .iloc[] to avoid pandas index misalignment
  3. Learning curve uses sklearn's learning_curve() with stratification
  4. Misclassified index alignment fixed using pd.Series with correct index
  5. MultinomialNB(alpha=0.0) wrapped in try/except with warning comment
  6. Confidence scores added to demo output
  7. Model + vectorizer serialized with joblib at the end
  8. Duplicate seaborn import removed
"""

# =============================================================================
# IMPORTS (all in one place)
# =============================================================================
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from collections import Counter

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score, precision_recall_curve, auc
)


# =============================================================================
# 1. LOAD DATASET & PLOT CLASS DISTRIBUTION
# =============================================================================
# FIX 1: Use usecols to explicitly select columns by position,
# instead of blind iloc[:, :2] which silently breaks on different CSV layouts.
data = pd.read_csv(
    'spam.csv',
    encoding='latin-1',
    usecols=[0, 1],       # only load the two columns we need
    header=0
)
data.columns = ['label', 'message']

print("Sample Data Preview:")
print(data.head())

print("\nDataset Summary:")
print(data['label'].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=data)
plt.title('Class Distribution: Ham vs Spam')
plt.xlabel('Message Type')
plt.ylabel('Count')
plt.show()


# =============================================================================
# 2. PREPROCESSING + BAG-OF-WORDS VECTORIZATION
# =============================================================================
label_encoder = LabelEncoder()
data['label_encoded'] = label_encoder.fit_transform(data['label'])

X_train, X_test, y_train, y_test = train_test_split(
    data['message'],
    data['label_encoded'],
    test_size=0.2,
    random_state=42,
    stratify=data['label_encoded']
)

vectorizer = CountVectorizer(stop_words='english')
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

print("Training data shape:", X_train_bow.shape)
print("Testing data shape:", X_test_bow.shape)


# =============================================================================
# 3. TRAIN NAIVE BAYES & EVALUATE
# =============================================================================
nb_model = MultinomialNB()
nb_model.fit(X_train_bow, y_train)
y_pred = nb_model.predict(X_test_bow)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))


# =============================================================================
# 4. RULE-BASED KEYWORD BASELINE
# =============================================================================
spam_keywords = ["win", "free", "urgent", "claim", "call", "click",
                 "offer", "prize", "congratulations"]

def keyword_baseline_predict(messages):
    preds = []
    for msg in messages:
        msg_lower = msg.lower()
        if any(word in msg_lower for word in spam_keywords):
            preds.append(1)
        else:
            preds.append(0)
    return preds

y_pred_baseline = keyword_baseline_predict(X_test)
baseline_acc = accuracy_score(y_test, y_pred_baseline)
print(f"Baseline Keyword Model Accuracy: {baseline_acc * 100:.2f}%")


# =============================================================================
# 5. CONFUSION MATRIX HEATMAP
# =============================================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Ham', 'Spam'],
    yticklabels=['Ham', 'Spam']
)
plt.title('Confusion Matrix - Naive Bayes SMS Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# =============================================================================
# 6. LEARNING CURVE
# FIX 2 & 3: Replaced manual slicing (which ignored pandas index alignment
# and skipped stratification) with sklearn's learning_curve(), which handles
# both correctly. Also removed the fragile y_train[:n] positional slice.
# =============================================================================
train_sizes_abs, train_scores, test_scores = learning_curve(
    MultinomialNB(),
    X_train_bow,
    y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,                          # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1
)

test_scores_mean = test_scores.mean(axis=1)
train_sizes_pct = train_sizes_abs / X_train_bow.shape[0]

plt.figure(figsize=(6, 4))
plt.plot(train_sizes_pct, test_scores_mean, marker='o')
plt.title('Learning Curve (Accuracy vs Training Data Size)')
plt.xlabel('Training Size Percentage')
plt.ylabel('CV Accuracy')
plt.grid(True)
plt.show()


# =============================================================================
# 7. ROC CURVE
# =============================================================================
y_pred_proba = nb_model.predict_proba(X_test_bow)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print(f"AUC: {roc_auc:.2f}")


# =============================================================================
# 8. PRECISION-RECALL CURVE
# =============================================================================
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
auprc = auc(recall, precision)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, color='darkorange', lw=2,
         label=f'Precision-Recall curve (AUPRC = {auprc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True)
plt.show()

print(f"AUPRC: {auprc:.2f}")


# =============================================================================
# 9. PREPROCESSING & SMOOTHING COMPARISON
# FIX 4: alpha=0.0 is mathematically unstable — causes log(0) for unseen words.
# Wrapped in try/except so the notebook doesn't crash; added warning comment.
# =============================================================================
print("Comparing Model Accuracy with different configurations:")

vectorizer_stop = CountVectorizer(stop_words='english')
X_train_stop = vectorizer_stop.fit_transform(X_train)
X_test_stop = vectorizer_stop.transform(X_test)

vectorizer_no_stop = CountVectorizer(stop_words=None)
X_train_no_stop = vectorizer_no_stop.fit_transform(X_train)
X_test_no_stop = vectorizer_no_stop.transform(X_test)

configs = [
    ("With stop words, alpha=1.0",    X_train_stop,    X_test_stop,    1.0),
    ("Without stop words, alpha=1.0", X_train_no_stop, X_test_no_stop, 1.0),
    ("With stop words, alpha=0.0",    X_train_stop,    X_test_stop,    0.0),
    ("Without stop words, alpha=0.0", X_train_no_stop, X_test_no_stop, 0.0),
]

for desc, X_tr, X_te, alpha in configs:
    try:
        # NOTE: alpha=0.0 disables Laplace smoothing, causing -inf log-probs
        # for unseen vocabulary. This is numerically unsafe in production —
        # always use alpha >= 1e-10 in real deployments.
        model_cfg = MultinomialNB(alpha=alpha)
        model_cfg.fit(X_tr, y_train)
        acc = accuracy_score(y_test, model_cfg.predict(X_te))
        print(f"{desc}: {acc * 100:.2f}%")
    except Exception as e:
        print(f"{desc}: FAILED — {e}")


# =============================================================================
# 10. WORDCLOUDS
# =============================================================================
from wordcloud import WordCloud

spam_text = " ".join(data[data['label'] == 'spam']['message'])
ham_text  = " ".join(data[data['label'] == 'ham']['message'])

plt.figure(figsize=(6, 4))
wc_spam = WordCloud(width=800, height=400, background_color='white').generate(spam_text)
plt.imshow(wc_spam, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud - Spam Messages')
plt.show()

plt.figure(figsize=(6, 4))
wc_ham = WordCloud(width=800, height=400, background_color='white').generate(ham_text)
plt.imshow(wc_ham, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud - Ham Messages')
plt.show()


# =============================================================================
# 11. TOP SPAM KEYWORDS
# =============================================================================
spam_words = " ".join(data[data['label'] == 'spam']['message']).lower()
spam_words = re.findall(r'\b[a-z]{3,}\b', spam_words)
spam_word_freq = Counter(spam_words).most_common(20)
spam_df = pd.DataFrame(spam_word_freq, columns=['word', 'count'])

plt.figure(figsize=(8, 4))
sns.barplot(x='count', y='word', data=spam_df)
plt.title('Top 20 Most Common Words in Spam Messages')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.show()


# =============================================================================
# 12. MISCLASSIFIED MESSAGES
# FIX 5: y_pred is a positional numpy array, but misclassified DataFrame
# retains the original dataset index. Wrapping y_pred in pd.Series with
# index=X_test.index ensures values align to the correct rows.
# =============================================================================
misclassified = data.iloc[X_test.index].copy()
misclassified['predicted'] = pd.Series(y_pred, index=X_test.index)  # FIX: aligned index

misclassified = misclassified[
    misclassified['label_encoded'] != misclassified['predicted']
]

print("Sample Misclassified Messages:")
print(misclassified[['message', 'label', 'predicted']].head(10))


# =============================================================================
# 13. DEMO WITH CONFIDENCE SCORES
# FIX 6: Added predict_proba() output so each prediction shows confidence.
# Low-confidence spam predictions signal the model is uncertain (e.g. OTP msgs).
# =============================================================================
user_messages = [
    "Congratulations! You've won a free vacation. Click here to claim.",
    "Are we still meeting tomorrow at 6?",
    "click on this link to win $10000",
    "Congrats! Your phone number just won a $1,000 gift card! Reply 'CASH' to accept now.",
    "URGENT: Your account will be suspended unless you verify now!",
    "Let's catch up later, it's been a while!",
    "Hey, are we still on for dinner tonight at 7? Let me know if you can make it",
    "Meeting canceled. Will reschedule for next week.",
    "Your one-time security code is 882103. Do not share this code with anyone.",
    "Can you send me the notes from class? I missed the first half hour.",
    "The concert is delayed due to heavy rain. Check the venue app for the new start time.",
    "URGENT: Your bank account has been suspended. Verify here: http://secure.acct-fix.co",
]

user_vectors = vectorizer.transform(user_messages)
user_preds   = nb_model.predict(user_vectors)
user_probas  = nb_model.predict_proba(user_vectors)

print("\nLive Spam Detection Test (with confidence):")
print(f"{'Prediction':<8} {'Confidence':>10}  Message")
print("-" * 80)
for msg, pred, proba in zip(user_messages, user_preds, user_probas):
    label = "Spam" if pred == 1 else "Ham"
    conf  = proba[pred] * 100
    # Flag low-confidence predictions — model is uncertain
    flag  = " [LOW CONFIDENCE]" if conf < 70 else ""
    print(f"{label:<8} {conf:>9.1f}%  {msg[:65]}{flag}")


# =============================================================================
# 14. SAVE MODEL & VECTORIZER
# FIX 7: Both artifacts must be serialized together — they are coupled.
# The vectorizer defines the vocabulary; the model was trained on that vocab.
# Always load them as a pair in your API.
# =============================================================================
joblib.dump(nb_model,   'nb_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("\nModel saved to:      nb_model.pkl")
print("Vectorizer saved to: vectorizer.pkl")
print("Load in your API with:")
print("  nb_model   = joblib.load('nb_model.pkl')")
print("  vectorizer = joblib.load('vectorizer.pkl')")
