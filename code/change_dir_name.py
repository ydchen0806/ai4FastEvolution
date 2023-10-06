import os
import re

# 定义要替换的字符串
old_string = "231002_reproduce"
new_string = "231002_reproduce"

# 定义要遍历的目录
directory = r"E:\XMU\231002_reproduce\code"  # 替换为您的目录路径

# 遍历目录中的文件
for root, dirs, files in os.walk(directory):
    for filename in files:
        if filename.endswith(".ipynb") or filename.endswith(".py"):
            file_path = os.path.join(root, filename)
            
            # 读取文件内容
            with open(file_path, "r", encoding="utf-8") as file:
                file_content = file.read()
            
            # 使用正则表达式替换字符串（考虑大小写敏感性）
            modified_content = re.sub(re.escape(old_string), new_string, file_content, flags=re.IGNORECASE)
            
            # 写入替换后的内容回文件
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(modified_content)
            
            print(f"替换文件 {filename} 中的字符串完成。")
