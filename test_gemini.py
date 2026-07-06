import os
from google import genai
os.environ["GOOGLE_CLOUD_PROJECT"] = "data-it-with-gpu-acceleration"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"
try:
    client = genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"], location=os.environ["GOOGLE_CLOUD_LOCATION"])
    response = client.models.generate_content(model="gemini-1.5-flash-001", contents="Hello")
    print("1.5-flash-001 success:", response.text)
except Exception as e:
    print("1.5-flash-001 error:", str(e))

try:
    response = client.models.generate_content(model="gemini-1.5-flash", contents="Hello")
    print("1.5-flash success:", response.text)
except Exception as e:
    print("1.5-flash error:", str(e))
