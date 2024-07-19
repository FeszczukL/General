import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:
def linear_feature(x_features,y_feature,df):
    X=df[x_features]
    y=df[y_feature]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    LR=LinearRegression()
    LR.fit(X_train,y_train)
    y_predict=LR.predict(X_test)
    print(LR.score(X_test,y_test))
    plt.scatter(X_test.Aces,y_predict)

df =pd.read_csv("tennis_stats.csv")
feature_set=['FirstServeReturnPointsWon','Aces']

linear_feature(feature_set,"Wins",df)

plt.show()


# perform exploratory analysis here:






















## perform single feature linear regressions here:






















## perform two feature linear regressions here:






















## perform multiple feature linear regressions here:
