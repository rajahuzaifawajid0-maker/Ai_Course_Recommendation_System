# ============================================================
# STUDENT AI COURSE RECOMMENDATION SYSTEM
# ============================================================

# Libraries import kar rahe hain jo data processing aur ML ke liye use hongi.
# Pandas data handle karega aur sklearn model training ke liye use hoga.

import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 1. DATASET LOAD
# ============================================================

# CSV file ko Pandas DataFrame mein load kar rahe hain.
# DataFrame ko hum "df" naam de rahe hain.

df = pd.read_csv("course_data.csv")


# Dataset ki first 5 rows dekh rahe hain.
# Isse humein dataset ka basic structure samajh aata hai.

print("\nFirst 5 Rows:")
print(df.head())


# Dataset mein kitni rows aur columns hain check kar rahe hain.
# Shape humein (rows, columns) format mein result deta hai.

print("\nDataset Shape:")
print(df.shape)


# Dataset ke columns aur data types check kar rahe hain.
# Isse pata chalega kis column mein numerical ya text data hai.

print("\nDataset Information:")
print(df.info())


# ============================================================
# 2. MISSING VALUES CHECK
# ============================================================

# Har column mein missing values check kar rahe hain.
# Missing value ka matlab hai ke kisi cell mein data available nahi.

print("\nMissing Values:")
print(df.isnull().sum())
    
# ============================================================
# 3. TARGET DISTRIBUTION
# ============================================================

# Dekh rahe hain ke har course ke kitne students hain.
# Isse pata chalta hai dataset balanced hai ya nahi.

print("\nCourse Distribution:")
print(df["course"].value_counts())


# ============================================================
# 4. LABEL ENCODING
# ============================================================

# Text values ko numerical values mein convert karne ke liye encoder bana rahe hain.
# Machine Learning models numerical data ko easily process karte hain.

education_encoder = LabelEncoder()
interest_encoder = LabelEncoder()
career_encoder = LabelEncoder()
course_encoder = LabelEncoder()


# Education column ko numbers mein convert kar rahe hain.
# Example: FSC, Bachelor's, Master's etc. numerical labels ban jayenge.

df["education"] = education_encoder.fit_transform(df["education"])


# Interest column ko numerical values mein convert kar rahe hain.
# Har different interest ko ek unique number milega.

df["interest"] = interest_encoder.fit_transform(df["interest"])


# Career goal ko numerical values mein convert kar rahe hain.
# Different career goals ko numerical labels milenge.

df["career_goal"] = career_encoder.fit_transform(df["career_goal"])


# Target course ko bhi numerical values mein convert kar rahe hain.
# Model prediction ke waqt in numbers ko dobara course names mein convert karenge.

df["course"] = course_encoder.fit_transform(df["course"])


# ============================================================
# 5. FEATURES AND TARGET
# ============================================================

# Course ko target variable bana rahe hain.
# Model ka main kaam student ke liye course predict karna hai.

X = df.drop("course", axis=1)
y = df["course"]


# Features aur target ka size check kar rahe hain.
# Isse confirm hota hai ke data correctly separate hua hai.

print("\nFeatures Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)


# ============================================================
# 6. TRAIN TEST SPLIT
# ============================================================

# Dataset ko training aur testing parts mein divide kar rahe hain.
# 80% data training aur 20% testing ke liye use hoga.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Training aur testing data ka size check kar rahe hain.
# Isse pata chalega model ko kitna data training ke liye mila.

print("\nTraining Data:")
print(X_train.shape)

print("\nTesting Data:")
print(X_test.shape)


# ============================================================
# 7. RANDOM FOREST MODEL
# ============================================================

# Random Forest classification model create kar rahe hain.
# Ye multiple decision trees combine karke prediction karta hai.

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# Model ko training data provide kar rahe hain.
# Model student features aur selected course ke relationship ko learn karega.

model.fit(X_train, y_train)


print("\nModel Training Completed!")


# ============================================================
# 8. PREDICTION
# ============================================================

# Testing data par model prediction kar raha hai.
# Model har student ke liye predicted course number return karega.

y_pred = model.predict(X_test)


# ============================================================
# 9. MODEL ACCURACY
# ============================================================

# Actual aur predicted values ko compare karke accuracy calculate kar rahe hain.
# Accuracy batati hai model ne kitni predictions correctly ki hain.

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)


# Accuracy percentage mein show kar rahe hain.
# Example: 0.95 ka matlab approximately 95% predictions correct hain.

print("\nAccuracy Percentage:")
print(accuracy * 100, "%")


# ============================================================
# 10. CLASSIFICATION REPORT
# ============================================================

# Precision, recall aur F1-score calculate kar rahe hain.
# Ye accuracy se zyada detailed model performance information deta hai.

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=course_encoder.classes_
))


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

# Confusion matrix actual aur predicted courses ko compare karti hai.
# Isse pata chalta hai model kis course ko kis course ke saath confuse kar raha hai.

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# 12. NEW STUDENT PREDICTION
# ============================================================

# Ab ek new student ka example bana rahe hain.
# Model is student ke liye suitable course predict karega.

new_student = pd.DataFrame({
    "age": [20],
    "education": [
        education_encoder.transform(["FSC"])[0]
    ],
    "percentage": [82],
    "python_level": [2],
    "programming_level": [2],
    "math_level": [3],
    "ai_knowledge": [2],
    "interest": [
        interest_encoder.transform(["Machine Learning"])[0]
    ],
    "career_goal": [
        career_encoder.transform(["ML Engineer"])[0]
    ]
})


# New student ke liye course prediction kar rahe hain.
# Prediction numerical form mein milegi.

prediction = model.predict(new_student)


# Numerical prediction ko original course name mein convert kar rahe hain.
# Ab user ko readable course name milega.

predicted_course = course_encoder.inverse_transform(prediction)


print("\n====================================")
print("STUDENT COURSE RECOMMENDATION")
print("====================================")

print("Recommended Course:", predicted_course[0])

print("====================================")

# ============================================================
# 13. SAVE TRAINED MODEL
# ============================================================

# Trained model ko file mein save kar rahe hain.
# Baad mein Flask website isi saved model ko use karegi.


os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/course_model.pkl")

joblib.dump(education_encoder, "model/education_encoder.pkl")
joblib.dump(interest_encoder, "model/interest_encoder.pkl")
joblib.dump(career_encoder, "model/career_encoder.pkl")
joblib.dump(course_encoder, "model/course_encoder.pkl")

print("\n====================================")
print("MODEL SAVED SUCCESSFULLY!")
print("====================================")
