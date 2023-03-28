import os

folder_path = "F:/DeepLearning/Dataset/Normal cases/Processed - Sao chép"

file_list = os.listdir(folder_path)

for file in file_list:
    if file.find("mask") != -1:
        os.remove(os.path.join(folder_path, file))
