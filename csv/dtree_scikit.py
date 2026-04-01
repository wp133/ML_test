import pandas as pd, csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

df = pd.read_csv(
    "data.csv",
    sep=",",
    quotechar=None,
    quoting=csv.QUOTE_NONE,
    engine="python",
)
df.columns = df.columns.str.replace('"', '').str.strip()
df = df.map(lambda v: v.strip('"') if isinstance(v, str) else v)

#print(df.columns)
#print(df.head())    
#print(df.info())      

X = df.drop("species", axis=1) 
y = df["species"]               
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=420)
model = DecisionTreeClassifier(max_depth=5, min_samples_split=5, random_state=420)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

print(export_text(model, feature_names=list(X.columns)))

########
new_test = pd.DataFrame([{
    "sepal_length": 5.1,
    "sepal_width": 3.1,
    "petal_length": 1.42,
    "petal_width": 0.1,
}]) 
xy = pd.get_dummies(new_test)
xy = new_test.reindex(columns=X.columns, fill_value=0)
prediction = model.predict(xy)
print(prediction)
#######
