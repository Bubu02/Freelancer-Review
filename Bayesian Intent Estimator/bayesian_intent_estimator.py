import numpy as np
import pandas as pd

n = 2500
random_ng = np.random.default_rng(42)

device = random_ng.choice(["mobile", "desktop"], size=n, p=[0.65, 0.35])
traffic_source = random_ng.choice(["search", "ads", "direct", "social"],
                                 size=2500, p=[0.45, 0.25, 0.20, 0.10])
search_use = (traffic_source == "search").astype(int)
filtering_actions = (search_use * (random_ng.random(2500) < 0.6)).astype(int)
basket_activity = ((filtering_actions | (random_ng.random(n) < 0.15)) * (random_ng.random(n) < 0.5)).astype(int)
checkout_behaviour = (basket_activity * (random_ng.random(n) < 0.55)).astype(int)
shipping_research = ((checkout_behaviour | (random_ng.random(n) < 0.1)) * (random_ng.random(n) < 0.7)).astype(int)
purchase_history = random_ng.choice([0, 1], size=n, p=[0.80, 0.20])

# Intent: mostly driven by checkout + some prior + tiny device bump
base = 0.05 + 0.40 * checkout_behaviour + 0.15 * purchase_history + 0.05 * (device == "desktop")
intent = (random_ng.random(n) < np.clip(base, 0, 0.95)).astype(int)



demo_data = pd.DataFrame(
        {
            "traffic_source": traffic_source,
            "device": device,
            "purchase_history": purchase_history,
            "filtering_actions": filtering_actions,
            "basket_activity": basket_activity,
            "checkout_behaviour": checkout_behaviour,
            "shipping_research": shipping_research,
            "intent": intent,
        })

demo_data.head(5)

from sklearn.model_selection import train_test_split

# feature and target seperation
features = [
        "traffic_source", "device", "purchase_history", "filtering_actions",
        "basket_activity", "checkout_behaviour", "shipping_research"
    ]
target = "intent"

# Standard split
X = demo_data[features].copy()
y = demo_data[target].astype(int)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)

# Combine features and target back into DataFrames
train_data = X_train.copy()
train_data['intent'] = y_train

from pgmpy.models.BayesianNetwork import BayesianNetwork
from pgmpy.estimators import BayesianEstimator

model = BayesianNetwork([
    ('traffic_source', 'filtering_actions'),
    ('filtering_actions', 'basket_activity'),
    ('basket_activity', 'checkout_behaviour'),
    ('checkout_behaviour', 'shipping_research'),
    ('checkout_behaviour', 'intent'),
    ('purchase_history', 'intent'),
    ('device', 'intent')
])
model.fit(train_data, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=10)

# Print one of the learned CPD
print('Print one of the learned CPD:')
print(model.cpds[0])

from pgmpy.inference import VariableElimination

inference = VariableElimination(model)

rows = []
for col in features:
    # states for this variable
    states = model.get_cpds(col).state_names[col]
    for val in states:
        q = inference.query(variables=["intent"], evidence={col: val}, show_progress=False)
        intent_states = list(q.state_names["intent"])

        p_high = float(q.values[intent_states.index(1)])
        p_low  = float(q.values[intent_states.index(0)])

        rows.append({
            "Attribute": col,
            "Value": {0: "No", 1: "Yes"}.get(val, val), # yes and no as in the documnt
            "Intent = High": round(p_high, 2),
            "Intent = Low": round(p_low, 2),
        })

cpt_df = pd.DataFrame(rows, columns=["Attribute", "Value", "Intent = High", "Intent = Low"])
print('CPT matrix for top attributes:')
print(cpt_df.to_string(index=False))
