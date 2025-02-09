import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, log_loss, roc_auc_score, average_precision_score, precision_recall_curve, \
    accuracy_score


def calculate_roc_curve(x, y, classifier):
    predicted_y = classifier.predict_proba(x)[:, 1]
    fpr, tpr, _ = roc_curve(y, predicted_y)
    return fpr, tpr


def calculate_fpr_curve(x, y, classifier):
    predicted_y = classifier.predict_proba(x)[:, 1]
    precision, recall, _ = precision_recall_curve(y, predicted_y)
    return precision, recall


def calculate_metrics(x, y, classifier):
    predicted_y = classifier.predict_proba(x)[:, 1]
    accuracy = classifier.score(x, y)
    auc_score = roc_auc_score(y, predicted_y)
    loss = log_loss(y, predicted_y)
    pr_scor = average_precision_score(y, predicted_y)
    return [accuracy, auc_score, loss, pr_scor]


def calculate_accuracy(x, y, classifier, threshold):
    predicted_y = classifier.predict_proba(x)[:, 1] > threshold
    accuracy = accuracy_score(y, predicted_y)
    return accuracy


def calculate_plots(x_train, y_train, x_test, y_test, classifier):
    # roc curve
    fpr_train, tpr_train = calculate_roc_curve(x_train, y_train, classifier)
    fpr_test, tpr_test = calculate_roc_curve(x_test, y_test, classifier)

    # plot
    figure, axes = plt.subplots()
    axes.plot([0, 1], [0, 1], 'k--')
    axes.plot(fpr_train, tpr_train, label='train')
    axes.plot(fpr_test, tpr_test, label='test')
    axes.set_xlabel('False positive rate')
    axes.set_ylabel('True positive rate')
    axes.set_title('ROC curve')
    axes.legend(loc='best')
    figure.show()

    # pr curve
    precision_train, recall_train = calculate_fpr_curve(x_train, y_train, classifier)
    precision_test, recall_test = calculate_fpr_curve(x_test, y_test, classifier)

    # plot
    figure, axes = plt.subplots()
    axes.plot([0, 1], [np.mean(y_train), np.mean(y_train)], 'k--')
    axes.plot(recall_train[recall_train > 0.01], precision_train[recall_train > 0.01], label='train')
    axes.plot(recall_test[recall_test > 0.01], precision_test[recall_test > 0.01], label='test')
    axes.set_xlabel('Recall')
    axes.set_ylabel('Precision')
    axes.set_title('PR curve')
    axes.legend(loc='best')
    figure.show()


def performance_indicators(x_train, y_train, x_validation, y_validation, classifier):
    # metrics
    metrics_train = calculate_metrics(x_train, y_train, classifier)
    metrics_validation = calculate_metrics(x_validation, y_validation, classifier)
    metrics_table = pd.DataFrame([metrics_train, metrics_validation],
                                 columns=['accuracy', 'auc', 'logloss', 'aupr'],
                                 index=['train', 'validation'])

    return metrics_table


def macro_regions():
    return {'Sicilia': 'SI',
            'Piemonte': 'NW',
            'Marche': 'C',
            "Valle d'Aosta": 'NW',
            'Abruzzo': 'C',
            'Toscana': 'C',
            'Campania': 'SI',
            'Puglia': 'SI',
            'Lombardia': 'NW',
            'Veneto': 'NE',
            'Emilia Romagna': 'C',
            'Trentino Alto Adige': 'NE',
            'Sardegna': 'SI',
            'Molise': 'SI',
            'Calabria': 'SI',
            'Lazio': 'C',
            'Liguria': 'NW',
            'Friuli Venezia Giulia': 'NE',
            'Basilicata': 'SI',
            'Umbria': 'C'}
