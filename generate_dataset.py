import pandas as pd
import numpy as np
alert_data = {
    "EAR": np.random.uniform(0.25, 0.40, 500),
    "BlinkRate": np.random.randint(10, 20, 500),
    "ClosureDuration": np.random.uniform(0.1, 0.4, 500),
    "PERCLOS": np.random.uniform(5, 15, 500),
    "Label": np.zeros(500)
}
drowsy_data = {
    "EAR": np.random.uniform(0.10, 0.22, 500),
    "BlinkRate": np.random.randint(20, 40, 500),
    "ClosureDuration": np.random.uniform(1, 5, 500),
    "PERCLOS": np.random.uniform(20, 70, 500),
    "Label": np.ones(500)
}
alert_df = pd.DataFrame(alert_data)

drowsy_df = pd.DataFrame(drowsy_data)

dataset = pd.concat(
    [alert_df, drowsy_df],
    ignore_index=True
)
dataset = dataset.sample(
    frac=1
).reset_index(drop=True)

dataset.to_csv(
    "dataset/fatigue_dataset.csv",
    index=False
)