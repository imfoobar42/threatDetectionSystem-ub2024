import kagglehub

# Download latest version
path = kagglehub.dataset_download("toluwaniaremu/smartcity-cctv-violence-detection-dataset-scvd")

print("Path to dataset files:", path)