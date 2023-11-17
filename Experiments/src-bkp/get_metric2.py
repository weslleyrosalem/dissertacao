from prometheus_api_client import PrometheusConnect
from prometheus_api_client.metric_range_df import MetricRangeDataFrame
from prometheus_api_client.utils import parse_datetime
from datetime import timedelta
import pandas as pd
from IPython.display import Image
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

prom_url = "http://demo.robustperception.io:9090"

pc = PrometheusConnect(url=prom_url, disable_ssl=True)

metric_data = pc.get_metric_range_data(
    "node_memory_Active_bytes",  # metric name and label config
    start_time=parse_datetime(
        "27 days ago"
    ),  # datetime object for metric range start time
    end_time=parse_datetime(
        "now"
    ),  # datetime object for metric range end time
    chunk_size=timedelta(
        days=27
    ),  # timedelta object for duration of metric data downloaded in one request
    store_locally=True,
)

