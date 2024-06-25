## helper functions 
# scikit-learn pipeline
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, log_loss

# general and visualization
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# source: extract CV average times sklearn: https://scikit-learn.org/stable/modules/cross_validation.html
def plot_time_curve(clf, X, y):
    # initialize empty lists for train time, valid time (mean and std)
    time_fit_mean, time_fit_std = [], []
    time_pred_mean, time_pred_std = [], []

    # extract total number of data points and create equally spaced size chunks
    num_points = y.shape[0]
    train_sizes = (num_points * np.linspace(.05, 1.0, 20)).astype('int')  
    for sizes in train_sizes:
        # extract a random (shuffled) subset of the incoming training size incrementally increasing the selected sample size
        samples = np.random.choice(X.shape[0], size=sizes, replace=False)
        X_partial = X.iloc[samples,:]
        y_partial = y.iloc[samples]

        # evaulate the default 5-fold cross-validation
        scores = cross_validate(clf, X_partial, y_partial, scoring='f1_weighted', return_train_score=True)

        # extract the training fitting time
        time_fit_mean.append(np.mean(scores['fit_time']))
        time_fit_std.append(np.std(scores['fit_time']))

        # extract the validation fitting time
        time_pred_mean.append(np.mean(scores['score_time']))
        time_pred_std.append(np.std(scores['score_time']))

    # convert appended times into array
    time_fit_mean = np.array(time_fit_mean)
    time_fit_std = np.array(time_fit_std)
    time_pred_mean = np.array(time_pred_mean)
    time_pred_std = np.array(time_pred_std)
    
    # plot training and validation time complexity (including mean and std)
    plt.figure(figsize=(6,4))
    plt.xlabel("Training Examples")
    plt.ylabel("Training Time (s)")
    plt.fill_between(train_sizes, time_fit_mean - 2*time_fit_std, time_fit_mean + 2*time_fit_std, alpha=0.25, color="b")
    plt.fill_between(train_sizes, time_pred_mean - 2*time_pred_std, time_pred_mean + 2*time_pred_std, alpha=0.25, color="g")
    plt.plot(train_sizes, time_fit_mean, 'o-', color="b", label="Training Time (s)")
    plt.plot(train_sizes, time_pred_mean, 'o-', color="g", label="Prediction Time (s)")
    plt.legend(loc="best")
    plt.show()
    
    return train_sizes, time_fit_mean, time_pred_mean, time_fit_std, time_pred_std
    

# source: Metrics sklearn: https://scikit-learn.org/stable/modules/model_evaluation.html
def final_classifier_evaluation(clf, X_train, X_test, y_train, y_test, data):
    
    # calculate the model fitting time
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    training_time = end - start
    print("Model Training Time (s)", training_time)
    
    # calcualte the model prediction time
    start = time.time()  
    y_pred = clf.predict(X_test)
    end = time.time()
    pred_time = end - start
    print("Model Prediction Time (s):",pred_time)

    # evaluate classification metrics (weighted for multi-class evaluation)
    if 'wine' in data:
        accuracy = accuracy_score(y_test,y_pred) 
        print("Accuracy:",accuracy)
        precision = precision_score(y_test,y_pred, average='weighted')
        print("Precision:",precision)
        recall = recall_score(y_test,y_pred, average='weighted')
        print("Recall:",recall)
        f1 = f1_score(y_test,y_pred, average='weighted') # harmonic mean between precision/recall
        print("F1-Score:",f1)
        f1_manual = 2*precision*recall/(precision+recall)   
        print("Manual F1 Score:",f1_manual)

    else:
        accuracy = accuracy_score(y_test,y_pred)
        print("Accuracy:",accuracy)
        precision = precision_score(y_test,y_pred)
        print("Precision:",precision)
        recall = recall_score(y_test,y_pred)
        print("Recall:",recall)
        f1 = f1_score(y_test, y_pred) # harmonic mean between precision/recall
        print("F1-Score:",f1)
        f1_manual = 2*precision*recall/(precision+recall) 
        print("Manual F1 Score:",f1_manual)
    

# source: Code stackoverflow: https://stackoverflow.com/questions/46912557/is-it-possible-to-get-test-scores-for-each-iteration-of-clfclassifier
def neural_net_epoch_learning(clf, X_train, y_train, X_test, y_test, epochs=500, batches=32):
    # ensure that data is properly standardized (used outside of sklearn pipeline)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_tr_scaled = scaler.transform(X_train)
    X_te_scaled = scaler.transform(X_test)

    # set initilization prameters
    n_train_samples = X_train.shape[0]
    n_epochs = epochs
    n_batch = batches
    n_batch = np.unique(y_train)

    # initialize empty lists for scoring and loss curves
    scores_train, scores_test = [], []
    loss_train, loss_test = [], []

    epoch = 0
    while epoch < n_epochs:
        if epoch % 10 == 0:
            print('epoch: ', epoch)

        # randomly shuffle dataset
        random_perm = np.random.permutation(X_train.shape[0])
        mini_batch_index = 0
        while True:
            # mini-batch iteratively fitting (update weights based on batch size)
            indices = random_perm[mini_batch_index:mini_batch_index + n_batch]
            clf.partial_fit(X_tr_scaled[indices], y_train.values[indices], classes=n_batch)
            mini_batch_index += n_batch

            if mini_batch_index >= n_train_samples:
                break

        # append scores (accuracy)
        scores_train.append(clf.score(X_tr_scaled, y_train.values))
        scores_test.append(clf.score(X_te_scaled, y_test.values))

        # append loss (cross-entropy loss)
        loss_train.append(log_loss(y_train.values, clf.predict_proba(X_tr_scaled)))
        loss_test.append(log_loss(y_test.values, clf.predict_proba(X_te_scaled)))

        epoch += 1

    # visualize the accuracy and loss curves
    plt.figure(figsize=(6,4))
    plt.plot(scores_train, color='green', alpha=0.8, label='Train')
    plt.plot(scores_test, color='magenta', alpha=0.8, label='Test')
    plt.title("Accuracy over epochs", fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.show()

    plt.figure(figsize=(6,4))
    plt.plot(loss_train, color='green', alpha=0.8, label='Train')
    plt.plot(loss_test, color='magenta', alpha=0.8, label='Test')
    plt.title("Loss over epochs", fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')
    plt.show()

    return scores_train, scores_test, loss_train, loss_test
