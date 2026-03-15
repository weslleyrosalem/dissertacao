import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import mean_absolute_error

from prometheus_api_client import PrometheusConnect
from prometheus_api_client.metric_range_df import MetricRangeDataFrame
from prometheus_api_client.utils import parse_datetime
from datetime import timedelta
import pandas as pd
from IPython.display import Image
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

metric_df = pd.read_json("../metrics/202307020143.json")

# Filter out non-numerical values
#metric_df2 = metric_df2[metric_df2['value'].apply(lambda x: str(x).replace('.', '', 1).isdigit())]

# Make the dataframe
#metric_df = MetricRangeDataFrame(metric_df2)
metric_df.index = pd.to_datetime(metric_df.index, utc=True)

# Resample the data
ts = metric_df["value"].astype(float).resample("30min").mean()

# Plot the time series
sns.set()
ts.plot(figsize=(15, 10))
plt.title("Visualize time series")
plt.ylabel("Node memory active bytes")
plt.show()