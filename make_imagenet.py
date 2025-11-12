import kagglehub

# Download latest version
path = kagglehub.dataset_download("../imagenet")

print("Path to dataset files:", path)