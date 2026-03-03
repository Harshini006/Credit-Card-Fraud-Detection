import requests
import json

# Sample transaction data (legitimate-looking)
data = {
    "Amount": 150.0
}
# Add V1-V28
for i in range(1, 29):
    data[f"V{i}"] = 0.1 * i  # Just dummy values

print("Sending request to API...")
try:
    response = requests.post("http://127.0.0.1:8000/predict", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Failed to connect: {e}")
