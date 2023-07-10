from prometheus_api_client import PrometheusConnect
from prometheus_api_client.metric_range_df import MetricRangeDataFrame
from prometheus_api_client.utils import parse_datetime
from datetime import timedelta
import pandas as pd
from IPython.display import Image
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# Creating the prometheus connect object with the required parameter
prom_url = "http://demo.robustperception.io:9090"
pc = PrometheusConnect(url=prom_url, disable_ssl=True)

# Request last week's data
metric_data = pc.get_metric_range_data(
    "node_memory_Active_bytes",  # metric name and label config
    start_time=parse_datetime(
        "27 days ago"
    ),  # datetime object for metric range start time
    end_time=parse_datetime(
        "now"
    ),  # datetime object for metric range end time
    chunk_size=timedelta(
        days=1
    ),  # timedelta object for duration of metric data downloaded in one request
)

## Make the dataframe
metric_df = MetricRangeDataFrame(metric_data)
metric_df.head()
metric_df.index = pd.to_datetime(metric_df.index, unit="s", utc=True)

metric_df.head()
### 30 mins because it gets very difficult to model otherwise
ts = metric_df["value"].astype(float).resample("30min").mean()
sns.set()
ts.plot(figsize=(15, 10))
plt.title("Visualize time series")
plt.ylabel("Node memory active bytes")
plt.show()





# Request last week's data
metric_data = pc.get_metric_range_data(
    "instance:node_network_transmit_bytes_excluding_lo:rate5m",  # metric name and label config
    start_time=parse_datetime(
        "27 days ago"
    ),  # datetime object for metric range start time
    end_time=parse_datetime(
        "now"
    ),  # datetime object for metric range end time
    chunk_size=timedelta(
        days=1
    ),  # timedelta object for duration of metric data downloaded in one request
)

## Make the dataframe
metric_df = MetricRangeDataFrame(metric_data)
metric_df.head()
metric_df.index = pd.to_datetime(metric_df.index, unit="s", utc=True)
metric_df.head()
### 30 mins because it gets very difficult to model otherwise
ts = metric_df["value"].astype(float).resample("30min").mean()
sns.set()
ts.plot(figsize=(15, 10))
plt.title("node_network_transmit_bytes_excluding_lo")
plt.ylabel("node_network_transmit_bytes_excluding_lo")
plt.show()

# Request last week's data
metric_data = pc.get_metric_range_data(
    "instance:node_cpu_utilisation:rate5m",  # metric name and label config
    start_time=parse_datetime(
        "27 days ago"
    ),  # datetime object for metric range start time
    end_time=parse_datetime(
        "now"
    ),  # datetime object for metric range end time
    chunk_size=timedelta(
        days=1
    ),  # timedelta object for duration of metric data downloaded in one request
)

## Make the dataframe
metric_df = MetricRangeDataFrame(metric_data)
metric_df.head()
metric_df.index = pd.to_datetime(metric_df.index, unit="s", utc=True)
metric_df.head()

### 30 mins because it gets very difficult to model otherwise
ts = metric_df["value"].astype(float).resample("30min").mean()
sns.set()
ts.plot(figsize=(15, 10))
plt.title("instance:node_cpu_utilisation:rate5m")
plt.ylabel("instance:node_cpu_utilisation:rate5m")
plt.show()
