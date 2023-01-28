{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "34ba07a8",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Import Data\n",
    "import pandas as pd\n",
    "survey_data = pd.read_csv(r'D:\\Google Drive\\Training\\Book\\0.Chapters\\Chapter4 Decison Trees\\Datasets\\Call_center_survey.csv')\n",
    "\n",
    "#total number of customers\n",
    "print(survey_data.shape)\n",
    "\n",
    "#Column names\n",
    "print(survey_data.columns)\n",
    "\n",
    "#Print Sample data\n",
    "pd.set_option('display.max_columns', None) #This option displays all the columns \n",
    "\n",
    "survey_data.head()\n",
    "\n",
    "#Sample summary\n",
    "summary=survey_data.describe()\n",
    "round(summary,2)\n",
    "\n",
    "#frequency counts table\n",
    "survey_data['Overall_Satisfaction'].value_counts()\n",
    "survey_data[\"Personal_loan_ind\"].value_counts()\n",
    "survey_data[\"Home_loan_ind\"].value_counts()\n",
    "survey_data[\"Prime_Customer_ind\"].value_counts()\n",
    "\n",
    "\n",
    "#Non numerical data need to be mapped to numerical data. \n",
    "survey_data['Overall_Satisfaction'] = survey_data['Overall_Satisfaction'].map( {'Dis Satisfied': 0, 'Satisfied': 1} ).astype(int)\n",
    "\n",
    "#number of satisfied customers\n",
    "survey_data['Overall_Satisfaction'].value_counts()\n",
    "\n",
    "#Defining Features and lables, ignoring cust_num and target variable\n",
    "features=list(survey_data.columns[1:6])\n",
    "print(features)\n",
    "#Preparing X and Y data\n",
    "#X = survey_data[[\"Age\", \"Account_balance\",\"Personal_loan_ind\",\"Home_loan_ind\",\"Prime_Customer_ind\"]]\n",
    "X = survey_data[features]\n",
    "y = survey_data['Overall_Satisfaction']\n",
    "\n",
    "#Building Tree Model\n",
    "from sklearn import tree\n",
    "DT_Model = tree.DecisionTreeClassifier(max_depth=2)\n",
    "DT_Model.fit(X,y)\n",
    "\n",
    "##Plotting the trees - Old Method\n",
    "\n",
    "#Before drawing the graph below command on anaconda console\n",
    "#pip install pydotplus \n",
    "#pip install graphviz\n",
    "\n",
    "from IPython.display import Image\n",
    "from six import StringIO\n",
    "\n",
    "import pydotplus\n",
    "dot_data = StringIO()\n",
    "tree.export_graphviz(DT_Model, #Mention the model here\n",
    "                     out_file = dot_data,\n",
    "                     filled=True, \n",
    "                     rounded=True,\n",
    "                     impurity=False,\n",
    "                     feature_names = features)\n",
    "\n",
    "graph = pydotplus.graph_from_dot_data(dot_data.getvalue())\n",
    "Image(graph.create_png())\n",
    "\n",
    "#Rules\n",
    "print(dot_data.getvalue())\n",
    "\n",
    "##Plotting the trees - New Method\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.tree import plot_tree, export_text\n",
    "plt.figure(figsize=(15,7))\n",
    "plot_tree(DT_Model, filled=True, \n",
    "                     rounded=True,\n",
    "                     impurity=False,\n",
    "                     feature_names = features)\n",
    "\n",
    "print(export_text(DT_Model, feature_names = features))\n",
    "\n",
    "\n",
    "#LAB : Tree Validation\n",
    "########################################\n",
    "##########Tree Validation\n",
    "#Tree Validation\n",
    "predict1 = DT_Model.predict(X)\n",
    "print(predict1)\n",
    "\n",
    "from sklearn.metrics import confusion_matrix \n",
    "cm = confusion_matrix(y, predict1)\n",
    "print(cm)\n",
    "\n",
    "total = sum(sum(cm))\n",
    "#####from confusion matrix calculate accuracy\n",
    "accuracy = (cm[0,0]+cm[1,1])/total\n",
    "print(accuracy)\n",
    "\n",
    "\n",
    "#LAB: Overfitting\n",
    "#LAB: The problem of overfitting\n",
    "############################################################################ \n",
    "##The problem of overfitting\n",
    "\n",
    "import pandas as pd\n",
    "overall_data = pd.read_csv(r\"D:\\Google Drive\\Training\\Book\\0.Chapters\\Chapter4 Decison Trees\\Datasets\\Customer_profile_data.csv\")\n",
    "\n",
    "##print train.info()\n",
    "print(overall_data.shape)\n",
    "\n",
    "#First few records\n",
    "print(overall_data.head())\n",
    "\n",
    "# the data have string values we need to convert them into numerical values\n",
    "overall_data['Gender'] = overall_data['Gender'].map( {'Male': 1, 'Female': 0} ).astype(int)\n",
    "overall_data['Bought'] = overall_data['Bought'].map({'Yes':1, 'No':0}).astype(int)\n",
    "\n",
    "#First few records\n",
    "print(overall_data.head())\n",
    "\n",
    "#Defining features, X and Y\n",
    "features = list(overall_data.columns[1:3])\n",
    "print(features)\n",
    "\n",
    "X = overall_data[features]\n",
    "y = overall_data['Bought']\n",
    "\n",
    "print(X.shape)\n",
    "print(y.shape)\n",
    "#Dividing X and y to train and test data parts. The function train_test_split() takes care of it. Mention the train data percentage in the parameter train_size. \n",
    "from sklearn import model_selection\n",
    "X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, train_size = 0.8 , random_state=5)\n",
    "\n",
    "print(\"X_train.shape\", X_train.shape)\n",
    "print(\"y_train.shape\",y_train.shape)\n",
    "print(\"X_test.shape\",X_test.shape)\n",
    "print(\"y_test.shape\",y_test.shape)\n",
    "\n",
    "##print train.info()\n",
    "##print test.info()\n",
    "\n",
    "from sklearn import tree\n",
    "#training Tree Model\n",
    "DT_Model1 = tree.DecisionTreeClassifier()\n",
    "DT_Model1.fit(X_train,y_train)\n",
    "\n",
    "#Plotting the trees\n",
    "from IPython.display import Image\n",
    "from six import StringIO\n",
    "import pydotplus\n",
    "dot_data = StringIO()\n",
    "tree.export_graphviz(DT_Model1,\n",
    "                     out_file = dot_data,\n",
    "                     feature_names = features,\n",
    "                     filled=True, rounded=True,\n",
    "                     impurity=False)\n",
    "\n",
    "graph = pydotplus.graph_from_dot_data(dot_data.getvalue())\n",
    "Image(graph.create_png())\n",
    "\n",
    "#Accuracy on train data\n",
    "from sklearn.metrics import confusion_matrix\n",
    "\n",
    "predict1 = DT_Model1.predict(X_train)\n",
    "cm1 = confusion_matrix(y_train,predict1)\n",
    "total1 = sum(sum(cm1))\n",
    "accuracy1 = (cm1[0,0]+cm1[1,1])/total1\n",
    "print(\"Train accuracy\", accuracy1)\n",
    "\n",
    "#Accuracy on test data\n",
    "predict2 = DT_Model1.predict(X_test)\n",
    "cm2 = confusion_matrix(y_test,predict2)\n",
    "total2 = sum(sum(cm2))\n",
    "#####from confusion matrix calculate accuracy\n",
    "accuracy2 = (cm2[0,0]+cm2[1,1])/total2\n",
    "print(\"Test accuracy\",accuracy2)\n",
    "\n",
    "####LAB: Pruning\n",
    "#### max_depth parameter \n",
    "DT_Model2 = tree.DecisionTreeClassifier(max_depth= 4)\n",
    "DT_Model2.fit(X_train,y_train)\n",
    "\n",
    "predict3 = DT_Model2.predict(X_train)\n",
    "predict4 = DT_Model2.predict(X_test)\n",
    "\n",
    "#Accuracy of the model on the train data\n",
    "cm1 = confusion_matrix(y_train,predict3)\n",
    "total1 = sum(sum(cm1))\n",
    "accuracy1 = (cm1[0,0]+cm1[1,1])/total1\n",
    "print(\"max_depth4 Train Accuracy\", accuracy1)\n",
    "\n",
    "#Accuracy of the model on the Test Data\n",
    "cm2 = confusion_matrix(y_test,predict4)\n",
    "total2 = sum(sum(cm2))\n",
    "accuracy2 = (cm2[0,0]+cm2[1,1])/total2\n",
    "print(\"max_depth4 Test Accuracy\", accuracy2)\n",
    "\n",
    "#### max_depth =2\n",
    "DT_Model2 = tree.DecisionTreeClassifier(max_depth= 2)\n",
    "DT_Model2.fit(X_train,y_train)\n",
    "\n",
    "predict3 = DT_Model2.predict(X_train)\n",
    "predict4 = DT_Model2.predict(X_test)\n",
    "\n",
    "#Accuracy of the model on the train data\n",
    "cm1 = confusion_matrix(y_train,predict3)\n",
    "total1 = sum(sum(cm1))\n",
    "accuracy1 = (cm1[0,0]+cm1[1,1])/total1\n",
    "print(\"max_depth2 Train Accuracy\", accuracy1)\n",
    "\n",
    "#Accuracy of the model on the Test Data\n",
    "cm2 = confusion_matrix(y_test,predict4)\n",
    "total2 = sum(sum(cm2))\n",
    "accuracy2 = (cm2[0,0]+cm2[1,1])/total2\n",
    "print(\"max_depth2 Test Accuracy\", accuracy2)\n",
    "\n",
    "#### The problem of underfitting\n",
    "#### max_depth =1\n",
    "DT_Model2 = tree.DecisionTreeClassifier(max_depth= 1)\n",
    "DT_Model2.fit(X_train,y_train)\n",
    "\n",
    "predict3 = DT_Model2.predict(X_train)\n",
    "predict4 = DT_Model2.predict(X_test)\n",
    "\n",
    "#Accuracy of the model on the train data\n",
    "cm1 = confusion_matrix(y_train,predict3)\n",
    "total1 = sum(sum(cm1))\n",
    "accuracy1 = (cm1[0,0]+cm1[1,1])/total1\n",
    "print(\"max_depth1 Train Accuracy\", accuracy1)\n",
    "\n",
    "#Accuracy of the model on the Test Data\n",
    "cm2 = confusion_matrix(y_test,predict4)\n",
    "total2 = sum(sum(cm2))\n",
    "accuracy2 = (cm2[0,0]+cm2[1,1])/total2\n",
    "print(\"max_depth1 Test Accuracy\", accuracy2)\n",
    "\n",
    "#### max_leaf_nodes =4\n",
    "DT_Model3 = tree.DecisionTreeClassifier(max_leaf_nodes= 3)\n",
    "DT_Model3.fit(X_train,y_train)\n",
    "\n",
    "predict3 = DT_Model3.predict(X_train)\n",
    "predict4 = DT_Model3.predict(X_test)\n",
    "\n",
    "#Accuracy of the model on the train data\n",
    "cm1 = confusion_matrix(y_train,predict3)\n",
    "total1 = sum(sum(cm1))\n",
    "accuracy1 = (cm1[0,0]+cm1[1,1])/total1\n",
    "print(accuracy1)\n",
    "\n",
    "#Accuracy of the model on the Test Data\n",
    "cm2 = confusion_matrix(y_test,predict4)\n",
    "total2 = sum(sum(cm2))\n",
    "accuracy2 = (cm2[0,0]+cm2[1,1])/total2\n",
    "print(accuracy2)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}