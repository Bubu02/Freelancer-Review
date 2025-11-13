import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

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

from sklearn.preprocessing import LabelEncoder

cat_cols = ['traffic_source', 'device']

# Label encodding
labels_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    demo_data[col] = le.fit_transform(demo_data[col])
    labels_encoders[col] = le

from sklearn.model_selection import train_test_split

# Features target separation
features = [
        "traffic_source", "device", "purchase_history", "filtering_actions",
        "basket_activity", "checkout_behaviour", "shipping_research"
    ]
target = "intent"

# Standard split
X = demo_data[features].copy()
y = demo_data[target].astype(int)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)

from catboost import CatBoostClassifier


model = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6,
                           loss_function='MultiClass', random_state=42, verbose=0)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")