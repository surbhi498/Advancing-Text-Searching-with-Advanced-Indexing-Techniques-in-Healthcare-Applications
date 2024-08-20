import requests

qdrant_url = "https://f1e9a70a-afb9-498d-b66d-cb248e0d5557.us-east4-0.gcp.cloud.qdrant.io:6333"
qdrant_api_key = "REXlX_PeDvCoXeS9uKCzC--e3-LQV0lw3_jBTdcLZ7P5_F6EOdwklA"

headers = {
    "api-key": qdrant_api_key
}

try:
    response = requests.get(qdrant_url, headers=headers)
    print(f"Status Code: {response.status_code}")
    print(f"Response Text: {response.text}")
except Exception as e:
    print(f"Connection Error: {e}")