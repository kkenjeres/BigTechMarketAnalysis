from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

# AdaBoost
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42, algorithm='SAMME')

# Extra Trees
et_model = ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42)

# Bagging Classifier
bag_model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)

estimators = [
    ('rf', rf_model),
    ('gb', gb_model),
    ('et', et_model)
]

# Voting Classifier (Hard)
vote_hard = VotingClassifier(estimators=estimators, voting='hard')

# Voting Classifier (Soft)
vote_soft = VotingClassifier(estimators=estimators, voting='soft')

all_models = {
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "AdaBoost": ada_model,
    "Extra Trees": et_model,
    "Bagging": bag_model,
    "Voting (Hard)": vote_hard,
    "Voting (Soft)": vote_soft
}
