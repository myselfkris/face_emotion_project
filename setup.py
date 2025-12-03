import os

# Define project root folder
project_root = "project_root"

folders = [
    f"{project_root}/data/train",
    f"{project_root}/data/val",
    f"{project_root}/data/test",
    f"{project_root}/preprocessing",
    f"{project_root}/model",
]

files = [
    f"{project_root}/preprocessing/face_detector.py",
    f"{project_root}/preprocessing/face_align.py",
    f"{project_root}/preprocessing/face_preprocess.py",
    f"{project_root}/model/mobilenet_emotion.py",
    f"{project_root}/train.py",
    f"{project_root}/inference.py"
]

# Create all folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create empty files
for file in files:
    open(file, "w").close()

print("🚀 Project structure created successfully!")


print("🚀 Project structure created successfully!")
