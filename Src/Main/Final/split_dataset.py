import os
import shutil
import random
from sklearn.model_selection import train_test_split

# path to processed dataset
dataset_dir = 'F:/DeepLearning/Dataset/Normal cases/Processed'

# path to Train, Test và Validation
output_dir = 'F:/DeepLearning/Dataset/Split Dataset'

# tỉ lệ chia dataset
train_ratio = 0.8
test_ratio = 0.1
val_ratio = 0.1

# lấy danh sách các tệp trong dataset
file_list = os.listdir(dataset_dir)

# split dataset into 2 parts: Train set and the others
train_files, remaining_files = train_test_split(file_list, train_size=train_ratio)

# split the leftover into 2 sets: Test and Validation
test_files, val_files = train_test_split(remaining_files, train_size=test_ratio/(test_ratio+val_ratio))

# Create Train folder and move the Train files into
train_dir = os.path.join(output_dir, 'Train')
os.makedirs(train_dir, exist_ok=True)
for file_name in train_files:
    src_file_path = os.path.join(dataset_dir, file_name)
    dst_file_path = os.path.join(train_dir, file_name)
    shutil.copy(src_file_path, dst_file_path)

# Create Test folder and move the Test files into
test_dir = os.path.join(output_dir, 'Test')
os.makedirs(test_dir, exist_ok=True)
for file_name in test_files:
    src_file_path = os.path.join(dataset_dir, file_name)
    dst_file_path = os.path.join(test_dir, file_name)
    shutil.copy(src_file_path, dst_file_path)

# Create Validation folder and move the Validation files into
val_dir = os.path.join(output_dir, 'Validation')
os.makedirs(val_dir, exist_ok=True)
for file_name in val_files:
    src_file_path = os.path.join(dataset_dir, file_name)
    dst_file_path = os.path.join(val_dir, file_name)
    shutil.copy(src_file_path, dst_file_path)
