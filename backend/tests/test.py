import requests

# FastAPI server URL (check if your route should end with a slash or not)
url = "http://localhost:8000/copilotkit_remote"  # without the trailing slash

# Payload data to test the query
payload = {
    "query": "machine learning",  # Example query for testing
    "max_results": 5
}

# Send POST request to the FastAPI endpoint
response = requests.post(url, json=payload)

# Print the response from the server
if response.status_code == 200:
    print("Response:", response.json())  # Successfully received the response
else:
    print(f"Failed to fetch data: {response.status_code}")
