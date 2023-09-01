from sklearn.model_selection import train_test_split

def split_train_dev_test(X, y, test_size, dev_size):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + dev_size), random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=(test_size / (test_size + dev_size)), random_state=42)

    return X_train, X_dev, X_test, y_train, y_dev, y_test
