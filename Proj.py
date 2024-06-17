import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

tran_features = pd.read_csv('training_set_features.csv')
features = [
    "behavioral_antiviral_meds",  
    "behavioral_avoidance",
    "behavioral_face_mask",
    "behavioral_wash_hands",
    "behavioral_large_gatherings",
    "behavioral_outside_home",
    "behavioral_touch_face",
    "doctor_recc_xyz",
    "doctor_recc_seasonal",
    "chronic_med_condition",
    "child_under_6_months",
    "health_worker",
    "health_insurance"
]
tran_features.drop('age_group',axis=1,inplace=True)
# Data Cleaning
# sns.heatmap(tran_features.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# sns.set_style("whitegrid")
# sns.countplot(x='xyz_concern',hue='health_worker',data=tran_features,palette='RdBu_r')
# plt.figure(figsize=(12,7))
# sns.boxplot(x='xyz_concern',y='health_worker',data=tran_features)
# plt.show()
tran_features.fillna({
    'health_insurance': tran_features['health_insurance'].mode()[0],
    'opinion_xyz_vacc_effective': tran_features['opinion_xyz_vacc_effective'].mode()[0],
    'opinion_xyz_risk': tran_features['opinion_xyz_risk'].mode()[0],
    'opinion_xyz_sick_from_vacc': tran_features['opinion_xyz_sick_from_vacc'].mode()[0],
    'opinion_seas_vacc_effective': tran_features['opinion_seas_vacc_effective'].mode()[0],
    'opinion_seas_risk': tran_features['opinion_seas_risk'].mode()[0],
    'age_group': tran_features['age_group'].mode()[0],
    'opinion_seas_sick_from_vacc': tran_features['opinion_seas_sick_from_vacc'].mode()[0],
    'education': tran_features['education'].mode()[0],
    'income_poverty': tran_features['income_poverty'].mode()[0],
    'marital_status': tran_features['marital_status'].mode()[0],
    'rent_or_own': tran_features['rent_or_own'].mode()[0],
    'employment_status': tran_features['employment_status'].mode()[0],
    'employment_industry': 'unknown',
    'employment_occupation': 'unknown',
}, inplace=True)


categorical_features = ['education', 'race', 'sex', 'income_poverty',
                        'marital_status', 'rent_or_own', 'employment_status', 'hhs_geo_region',
                        'census_msa', 'employment_industry', 'employment_occupation']
numeric_features = [ "respondent_id",
    "xyz_concern",
    "xyz_knowledge",
    "behavioral_antiviral_meds",
    "behavioral_avoidance",
    "behavioral_face_mask",
    "behavioral_wash_hands",
    "behavioral_large_gatherings",
    "behavioral_outside_home",
    "behavioral_touch_face",
    "doctor_recc_xyz",
    "doctor_recc_seasonal",
    "chronic_med_condition",
    "child_under_6_months",
    "health_worker",
    "health_insurance",
    "opinion_xyz_vacc_effective",
    "opinion_xyz_risk",
    "opinion_xyz_sick_from_vacc",
    "opinion_seas_vacc_effective",
    "opinion_seas_risk",
    "opinion_seas_sick_from_vacc",
    "household_adults",
    "household_children"]
for i in numeric_features:
      tran_features[numeric_features] = tran_features[numeric_features].fillna(tran_features[numeric_features].median())
print(tran_features.isnull().sum())

model_features = numeric_features + categorical_features  

clf = RandomForestClassifier(random_state=42)
labels = tran_features[['xyz_concern', 'opinion_xyz_sick_from_vacc']]

roc_auc_scores = cross_val_score(clf, tran_features[model_features],labels, cv=5, scoring='roc_auc')

