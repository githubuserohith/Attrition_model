import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier,RandomForestClassifier,VotingClassifier
import xgboost as xgm
import lightgbm as lg
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def fn_model(df):

    # nulls
    print(f"nulls {df.isna().sum()}")

    # display number of outliers
    df_num = df.select_dtypes(include='number')
    df_outlier = {}
    for item in df_num.columns:
        q3 = df_num[item].quantile(0.75)
        q1 = df_num[item].quantile(0.25)
        iqr = q3-q1
        ub = q3+(1.5*iqr)
        lb = q1-(1.5*iqr)
        count = 0
        for v in df[item]:
            if v<lb or v>ub:
                count = count+1
        df_outlier[item] = count
    df_outlier = pd.DataFrame(df_outlier, index=['Outliers_count'])
    print(df_outlier.T.sort_values(by='Outliers_count', ascending=False))

    # split into features and target
    
    X, pp = fn_pipe(df, 'train',0,0)
    y=  df.Attrition
    y = y.replace({'No':0,'Yes':1})

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Different models and their auc and precision
    models,precision,f1,auc = [],[],[],[]
    model_list = [DecisionTreeClassifier(),
                RandomForestClassifier(),
                GradientBoostingClassifier(), 
                AdaBoostClassifier(), 
                # xgm.XGBClassifier(), 
                lg.LGBMClassifier(),
                LogisticRegression()
                ]
    for model in model_list:
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
    #     print(classification_report(y_train, pred_train))
        pred_test = model.predict(X_test)
    #     print(classification_report(y_test, pred_test))
    #     print(confusion_matrix(y_test, pred_test))
    #     print(roc_auc_score(y_test, pred_test))
        models.append(str(model).split('(')[0])
        precision.append(precision_score(y_test, pred_test))
        # f1.append(f1_score(y_test, pred_test))
        auc.append(roc_auc_score(y_test, pred_test))
    print(pd.DataFrame({'model':models,'precision':precision,'auc':auc}).sort_values(by='auc'))

    # ensembler
    gb_clf = xgm.XGBClassifier()
    lr_clf = lg.LGBMClassifier(random_state=42)
    rf_clf = RandomForestClassifier(random_state=42)
    gb_clf  = AdaBoostClassifier(random_state=42)
    rf_clf = GradientBoostingClassifier()
    lr_clf  = LogisticRegression(random_state=42, class_weight="balanced")

    hard_voting_clf = VotingClassifier(estimators=[
        ('rf', rf_clf),
        ('gb', gb_clf),
        ('lr', lr_clf)
    #     ('svc', svc_clf)
    ], voting='soft')

    hard_voting_clf.fit(X_train, y_train)
    hard_voting_predictions = hard_voting_clf.predict(X_train)
    hard_voting_cr = classification_report(y_train, hard_voting_predictions)
    print(hard_voting_cr)

    hard_voting_predictions = hard_voting_clf.predict(X_test)
    hard_voting_cr = classification_report(y_test, hard_voting_predictions)
    print(hard_voting_cr)

    # selecting best threshold
    df['prob'] = hard_voting_clf.predict_proba(X)[:, 1]
    df_ROC = pd.DataFrame(columns=['Threshold', 'TP', 'TN', 'FP', 'FN', 'TPR', 'FPR'])

    for threshold in np.arange(0, 1.1, 0.1):
        y_pred = (df['prob'] > threshold).astype(int)
        cm = confusion_matrix(y, y_pred)
        TP = cm[1, 1]
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        df_ROC.loc[len(df_ROC)] = pd.Series({'Threshold': threshold, 
                                            'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 
                                            'TPR': TPR, 'FPR': FPR})

    # Compute AUC
    auc = roc_auc_score(y, df['prob'])

    # Print ROC curve and AUC
    print(df_ROC)
    print('AUC:', auc)

    # Plot ROC curve from df_ROC
    # plt.plot(df_ROC['FPR'], df_ROC['TPR'], label=f'AUC={auc:.2f}')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend()
    # plt.show()

    # Compute TPR - FPR
    df_ROC['TPR-FPR'] = df_ROC['TPR'] - df_ROC['FPR']

    # Select threshold with max TPR - FPR
    max_TPR_FPR_diff = df_ROC['TPR-FPR'].max()
    selected_threshold = df_ROC[df_ROC['TPR-FPR'] == max_TPR_FPR_diff]['Threshold'].values[0]
    print(f"Best threshold: {selected_threshold}")

    # Compute confusion matrix for this threshold
    y_pred_selected_threshold = (df['prob'] > selected_threshold).astype(int)
    cm_max_diff = confusion_matrix(y, y_pred_selected_threshold)

    # Calculate metrics from confusion matrix
    TN = cm_max_diff[0, 0]
    FP = cm_max_diff[0, 1]
    FN = cm_max_diff[1, 0]
    TP = cm_max_diff[1, 1]

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    misclassification_rate = 1 - accuracy
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    f1_score = 2 * precision * sensitivity / (precision + sensitivity)

    # Print metrics
    print('Accuracy:', accuracy)
    print('Misclassification Rate:', misclassification_rate)
    print('Sensitivity (Recall):', sensitivity)
    print('Specificity:', specificity)
    print('Precision:', precision)
    print('F1 Score:', f1_score)

    # model_list.append(hard_voting_clf)
    return (hard_voting_clf,pp,X_train,X_test,y_train,y_test,model_list)

def fn_pipe(df,type,model,pp):
    if type=="train":
        X = df.drop(['Attrition'], axis=1)
        preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), ['Department','EducationField','MaritalStatus']),
            ('num',MinMaxScaler(),['Age','DistanceFromHome','Education','MonthlyIncome','YearsAtCompany'])
        ]
        #     remainder='passthrough'  # passthrough other columns
        )
        pipeline = Pipeline([
            ('preprocessor', preprocessor)
            ])
        X1 = pd.DataFrame(pipeline.fit_transform(X), columns=preprocessor.get_feature_names_out(X.columns))
        X = pd.concat([X1,X[['EnvironmentSatisfaction','JobSatisfaction','NumCompaniesWorked','WorkLifeBalance']]], axis=1)

        return (X,pipeline)
    
    elif type=="test":
        X_test = df.drop(['Attrition'], axis=1)        
        X1 = pd.DataFrame(pp.transform(X_test), columns=pp.get_feature_names_out(X_test.columns))
        X1 = pd.concat([X1,X_test[['EnvironmentSatisfaction','JobSatisfaction','NumCompaniesWorked','WorkLifeBalance']]], axis=1)
        X_test['output'] = model.predict(X1)
        X_test.output = X_test.output.replace({0:'No',1:'Yes'})
        return (X_test)
