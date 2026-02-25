import pandas as pd
from sklearn.ensemble import RandomForestClassifier

data = {
    "temperature":[50,52,49,80,85],
    "vibration":[0.2,0.3,0.25,0.9,1.1],
    "failure":[0,0,0,1,1]
}

df = pd.DataFrame(data)

X = df[["temperature","vibration"]]
y = df["failure"]

model = RandomForestClassifier()
model.fit(X,y)

print("Model trained successfully")