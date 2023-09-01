from sklearn.metrics import accuracy_score

def func(X, a, b):
    b_predict = X.predict(a)

    accuracy = accuracy_score(b, b_predict)
    
    return accuracy
