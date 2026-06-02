# ========================================================
# 直方图统计方法 - 批量处理版本
# 功能：将 1x 图像的颜色匹配到对应 3x 图像，并保存到 3x_method2 文件夹
# 数据结构：
# Cutting/
#   DJI/
#     train/1x, train/3x
#     test/1x, test/3x
#     validation/1x, validation/3x
#   rewhite/
#     train/1x, train/3x
#     test/1x, test/3x
#     validation/1x, validation/3x
# ========================================================

import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.exposure import match_histograms


# ========================================================
# RGB 直方图匹配
# src：需要被改变颜色的图像，例如 1x
# target：目标颜色图像，例如 3x
# 返回：颜色分布尽量接近 target 的 src 图像
# ========================================================
def white_balance(src, target):
    if src is None:
        raise ValueError("src 图像为空，请检查 1x 图像路径。")
    if target is None:
        raise ValueError("target 图像为空，请检查 3x 图像路径。")

    # OpenCV 读取为 BGR，这里先转换为 RGB 再逐通道匹配
    src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    target_rgb = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    matched_channels = []
    for c in range(3):
        matched = match_histograms(src_rgb[:, :, c], target_rgb[:, :, c])
        matched = np.clip(matched, 0, 255).astype(np.uint8)
        matched_channels.append(matched)

    matched_rgb = cv2.merge(matched_channels)
    result = cv2.cvtColor(matched_rgb, cv2.COLOR_RGB2BGR)
    return result


# ========================================================
# 读取图像，支持中文路径/特殊路径
# ========================================================
def imread_unicode(path):
    path = str(path)
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


# ========================================================
# 保存图像，支持中文路径/特殊路径
# ========================================================
def imwrite_unicode(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ext = path.suffix
    if ext == "":
        ext = ".png"
        path = path.with_suffix(ext)

    success, encoded = cv2.imencode(ext, image)
    if not success:
        raise IOError(f"图像编码失败：{path}")
    encoded.tofile(str(path))


# ========================================================
# 获取图片文件
# ========================================================
def list_images(folder, exts):
    folder = Path(folder)
    image_files = []
    for ext in exts:
        image_files.extend(folder.glob(f"*{ext}"))
        image_files.extend(folder.glob(f"*{ext.upper()}"))
    return sorted(set(image_files))


# ========================================================
# 处理一个 split，例如 DJI/train 或 rewhite/test
# ========================================================
def process_one_split(split_dir, exts, overwrite=False):
    split_dir = Path(split_dir)
    dir_1x = split_dir / "1x"
    dir_3x = split_dir / "3x"
    out_dir = split_dir / "3x_method2"

    if not dir_1x.exists():
        print(f"[跳过] 未找到 1x 文件夹：{dir_1x}")
        return 0, 0, 0
    if not dir_3x.exists():
        print(f"[跳过] 未找到 3x 文件夹：{dir_3x}")
        return 0, 0, 0

    files_1x = list_images(dir_1x, exts)
    if len(files_1x) == 0:
        print(f"[跳过] 1x 文件夹中没有图片：{dir_1x}")
        return 0, 0, 0

    out_dir.mkdir(parents=True, exist_ok=True)

    n_success = 0
    n_missing = 0
    n_failed = 0

    for path_1x in files_1x:
        path_3x = dir_3x / path_1x.name
        out_path = out_dir / path_1x.name

        if out_path.exists() and not overwrite:
            n_success += 1
            continue

        if not path_3x.exists():
            print(f"[缺少对应 3x] {path_3x}")
            n_missing += 1
            continue

        img_1x = imread_unicode(path_1x)
        img_3x = imread_unicode(path_3x)

        if img_1x is None:
            print(f"[读取失败] 1x：{path_1x}")
            n_failed += 1
            continue
        if img_3x is None:
            print(f"[读取失败] 3x：{path_3x}")
            n_failed += 1
            continue

        try:
            # 核心：将 3x 的颜色匹配到 1x
            result = white_balance(img_3x, img_1x)
            imwrite_unicode(out_path, result)
            n_success += 1
        except Exception as e:
            print(f"[处理失败] {path_1x} -> {out_path}，原因：{e}")
            n_failed += 1

    print(f"[完成] {split_dir} | 成功/已存在: {n_success}, 缺少3x: {n_missing}, 失败: {n_failed}, 输出: {out_dir}")
    return n_success, n_missing, n_failed


# ========================================================
# 批量处理 Cutting 下的 DJI 和 rewhite
# ========================================================
def batch_process(root_dir, datasets, splits, exts, overwrite=False):
    root_dir = Path(root_dir)

    total_success = 0
    total_missing = 0
    total_failed = 0

    print("=" * 80)
    print(f"Root folder: {root_dir}")
    print(f"Datasets: {datasets}")
    print(f"Splits: {splits}")
    print("=" * 80)

    for dataset in datasets:
        for split in splits:
            split_dir = root_dir / dataset / split
            s, m, f = process_one_split(
                split_dir=split_dir,
                exts=exts,
                overwrite=overwrite,
            )
            total_success += s
            total_missing += m
            total_failed += f

    print("=" * 80)
    print("全部处理完成")
    print(f"总成功/已存在: {total_success}")
    print(f"总缺少对应 3x: {total_missing}")
    print(f"总失败: {total_failed}")
    print("=" * 80)


# ========================================================
# 主函数
# ========================================================
def main():
    current_dir = Path(__file__).resolve().parent
    default_root = current_dir / "Data/Cutting"

    parser = argparse.ArgumentParser(description="Batch RGB histogram matching: convert 1x colors to corresponding 3x colors.")
    parser.add_argument(
        "--root",
        type=str,
        default=str(default_root),
        help="Cutting 文件夹路径。默认使用当前脚本同级目录下的 Cutting 文件夹。",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["DJI", "rewhite"],
        help="需要处理的数据集文件夹名称。默认：DJI rewhite。",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test", "validation"],
        help="需要处理的数据划分。默认：train test validation。",
    )
    parser.add_argument(
        "--exts",
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"],
        help="需要处理的图片后缀。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果输出图像已经存在，是否覆盖。默认不覆盖。",
    )

    args = parser.parse_args()

    batch_process(
        root_dir=args.root,
        datasets=args.datasets,
        splits=args.splits,
        exts=args.exts,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
