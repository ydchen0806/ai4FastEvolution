import os
import zipfile

# 指定目录路径
directory_path = r'E:\XMU\231002_reproduce\weather_1005'

# 获取目录下的所有zip文件
zip_files = [f for f in os.listdir(directory_path) if f.endswith('.zip')]

# 遍历所有zip文件并解压缩
for zip_file in zip_files:
    zip_file_path = os.path.join(directory_path, zip_file)
    
    # 创建解压缩目标文件夹，以zip文件的名称为基础
    extract_folder = os.path.splitext(zip_file)[0]
    extract_folder_path = os.path.join(directory_path, extract_folder)
    os.makedirs(extract_folder_path, exist_ok=True)
    
    # 打开zip文件并解压缩到目标文件夹
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder_path)
    
    # 删除原始zip文件
    os.remove(zip_file_path)
    
    print(f"已解压缩并删除文件: {zip_file}")

print("解压缩并删除完成")