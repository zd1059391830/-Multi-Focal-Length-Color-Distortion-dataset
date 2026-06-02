#========================
#改名字变成能用的数据
#L：ground-truth, 1x
#R: 3x
#=========================


import os
import shutil

# 原始数据和目标数据的路径
src_root = "F:\\DJ\\Data\\Result\\Cutting"
dst_root = "F:\\DJ\\Data\\Result\\Artificial Dataset"

# 处理的子集
splits = ['train', 'test', 'validation']

for split in splits:
    tele_dir = os.path.join(src_root, split, 'tele')
    wide_dir = os.path.join(src_root, split, 'wide_1x')

    dst_split_dir = os.path.join(dst_root, split)
    os.makedirs(dst_split_dir, exist_ok=True)

    # 获取所有 tele 文件名，假设 wide_1x 同样有相同名字
    tele_files = sorted(os.listdir(tele_dir))
    wide_files = sorted(os.listdir(wide_dir))

    # 确保一对对应
    tele_files = [f for f in tele_files if f in wide_files]

    for idx, filename in enumerate(tele_files):
        basename = f"{idx:04d}"  # 生成四位数字，比如 0000

        src_tele_path = os.path.join(tele_dir, filename)
        src_wide_path = os.path.join(wide_dir, filename)

        dst_tele_path = os.path.join(dst_split_dir, f"{basename}_R.png")
        dst_wide_path = os.path.join(dst_split_dir, f"{basename}_L.png")

        shutil.copy(src_tele_path, dst_tele_path)
        shutil.copy(src_wide_path, dst_wide_path)

print("所有图片重命名并复制完成 ✅")