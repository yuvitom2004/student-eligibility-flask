import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Step 1: Generate synthetic dataset
np.random.seed(42)
n = 200  # number of students

data = {
    "Internal_Marks": np.random.randint(0, 31, n),
    "Attendance": np.random.randint(40, 101, n),
    "Lab_Marks": np.random.randint(0, 21, n),
    "Discipline_Score": np.random.randint(0, 11, n),
    "Participation_Score": np.random.randint(0, 11, n),
    "Punctuality_Score": np.random.randint(0, 11, n)
}

df = pd.DataFrame(data)

def check_eligibility(row):
    if row["Attendance"] < 65 or row["Internal_Marks"] < 12 or row["Lab_Marks"] < 8:
        return 0
    overall = (row["Internal_Marks"]/30*0.4 +
               row["Attendance"]/100*0.2 +
               row["Lab_Marks"]/20*0.2 +
               row["Discipline_Score"]/10*0.05 +
               row["Participation_Score"]/10*0.05 +
               row["Punctuality_Score"]/10*0.1)
    return 1 if overall >= 0.5 else 0

df["Eligible"] = df.apply(check_eligibility, axis=1)

# Step 2: Train model
X = df.drop("Eligible", axis=1)
y = df["Eligible"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 3: Save model
pickle.dump(model, open("eligibility_model.pkl", "wb"))
print("Model trained and saved as eligibility_model.pkl ✅")
