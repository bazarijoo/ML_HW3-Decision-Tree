
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pydotplus
from IPython.display import Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report, confusion_matrix

def estimate_with_scikitlearn(data,criterion='gini'):
# train with scikit learn
    X = data.iloc[:, :-1].values
    y = data.iloc[:, 13].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

    tree = DecisionTreeClassifier(criterion=criterion,max_depth=3)
    tree.fit(X_train, y_train)

    dot_data=export_graphviz(
    tree,
    out_file=None,
    feature_names=list(data.columns[:-1]),
    class_names=['0','1'],
    filled=True,
    rounded=True)

    graph = pydotplus.graph_from_dot_data(dot_data)
    Image(graph.create_png())

    if criterion=='gini':
        graph.write_png("gini_tree.png")
    else:
        graph.write_png("entropy_tree.png")

    y_pred = tree.predict(X_test)

    print('scikit learn confusion metric with ',criterion,' criteria :')
    print(classification_report(y_test, y_pred))
    print("The prediction accuracy is: ", tree.score(X_test, y_test) * 100, "%")
