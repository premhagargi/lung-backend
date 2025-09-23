import requests

url = "http://localhost:5000/predict"
image_path = "C:/Users/91805/Desktop/Lung/lung_ct_dataset/test/adenocarcinoma/000109 (4).png"  # Update with a valid image path

with open(image_path, "rb") as image_file:
    files = {"image": image_file}
    response = requests.post(url, files=files)

if response.status_code == 200:
    print(response.json())
else:
    print(f"Error: {response.status_code}, {response.json()}")