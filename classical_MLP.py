# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 14:16:44 2022

@author: plini
"""

#Carregar os dados...

X_train, X_test, y_train, y_test = train_test_split(df_X, df_class, test_size = 0.2, random_state=18)

#classical MLP
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt 
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =8)

 #Usando o MLP
mlp_c3 = MLPClassifier()
mlp_c3.fit(X_train, y_train)
pred_mlp = mlp_c3.predict(X_test)
print("\n relatório de classificação")
print(classification_report(y_test, pred_mlp))
print("\nconfusion matrix")
print (pd.crosstab(y_test,pred_mlp, rownames=['Real'], colnames=['Predito'], margins=True))
print("\nBalanced Accuracy")
ba_c3 = balanced_accuracy_score(y_test, pred_mlp)
print(ba_c3)
plot_confusion_matrix(mlp_c3, X_test, y_test)  # doctest: +SKIP
plt.show()  # doctest: +SKIP
