#========================================================
# 功能：将 3x 图像的颜色改变为对应 1x 图像的颜色，直方图+Lab色空间
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
#========================================================

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from skimage.exposure import match_histograms
from luxpy import srgb_to_xyz, xyz_to_lab, lab_to_xyz, xyz_to_srgb


# 支持的图片格式
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def imread_unicode(path):
    """兼容中文路径的图像读取。"""
    path = str(path)
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def imwrite_unicode(path, img):
    """兼容中文路径的图像保存。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def white_balance(src, target):
    """
    Lab 直方图匹配。

    参数：
        src    : 需要被改变颜色的图像，BGR格式。
        target : 颜色参考图像，BGR格式。

    返回：
        result : 将 src 的颜色分布匹配到 target 后的图像，BGR格式。
    """
    # BGR -> RGB
    src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    target_rgb = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

    # RGB -> XYZ -> Lab
    src_XYZ = srgb_to_xyz(src_rgb)
    target_XYZ = srgb_to_xyz(target_rgb)

    src_shape = src_XYZ.shape
    src_XYZ_x = src_XYZ.reshape(-1, 3)
    target_XYZ_x = target_XYZ.reshape(-1, 3)

    src_Lab = xyz_to_lab(src_XYZ_x)
    target_Lab = xyz_to_lab(target_XYZ_x)

    # 分别对 L、a、b 通道做直方图匹配
    matched_L = match_histograms(src_Lab[:, 0], target_Lab[:, 0])
    matched_a = match_histograms(src_Lab[:, 1], target_Lab[:, 1])
    matched_b = match_histograms(src_Lab[:, 2], target_Lab[:, 2])

    matched_Lab = np.vstack((matched_L, matched_a, matched_b)).T

    # Lab -> XYZ -> RGB
    matched_XYZ = lab_to_xyz(matched_Lab)
    matched_XYZ = matched_XYZ.reshape(src_shape)
    matched_rgb = xyz_to_srgb(matched_XYZ)

    matched_rgb = np.clip(matched_rgb, 0, 255).astype(np.uint8)

    # RGB -> BGR
    result = cv2.cvtColor(matched_rgb, cv2.COLOR_RGB2BGR)
    return result


def collect_images(folder):
    """收集一个文件夹下的图片，并用文件名建立索引。"""
    folder = Path(folder)
    image_dict = {}
    for ext in IMAGE_EXTS:
        for p in folder.glob(f"*{ext}"):
            image_dict[p.name] = p
        for p in folder.glob(f"*{ext.upper()}"):
            image_dict[p.name] = p
    return image_dict


def process_one_split(split_dir, overwrite=False):
    """
    处理某个 split，例如：Cutting/DJI/train。

    读取：split_dir/3x
    参考：split_dir/1x
    保存：split_dir/3x_method3
    """
    split_dir = Path(split_dir)
    dir_1x = split_dir / "1x"
    dir_3x = split_dir / "3x"
    save_dir = split_dir / "3x_method3"
    save_dir.mkdir(parents=True, exist_ok=True)

    if not dir_1x.exists():
        print(f"[跳过] 找不到 1x 文件夹：{dir_1x}")
        return 0, 0, 0
    if not dir_3x.exists():
        print(f"[跳过] 找不到 3x 文件夹：{dir_3x}")
        return 0, 0, 0

    images_1x = collect_images(dir_1x)
    images_3x = collect_images(dir_3x)

    total = len(images_3x)
    success = 0
    skipped = 0

    for name, path_3x in sorted(images_3x.items()):
        path_1x = images_1x.get(name)
        if path_1x is None:
            print(f"[跳过] 3x 中的图片没有找到同名 1x 参考图：{path_3x}")
            skipped += 1
            continue

        save_path = save_dir / name
        if save_path.exists() and not overwrite:
            print(f"[已存在] {save_path}，如需覆盖请加 --overwrite")
            skipped += 1
            continue

        img_3x = imread_unicode(path_3x)   # 需要被改变颜色的图
        img_1x = imread_unicode(path_1x)   # 颜色参考图

        if img_3x is None:
            print(f"[失败] 无法读取 3x 图片：{path_3x}")
            skipped += 1
            continue
        if img_1x is None:
            print(f"[失败] 无法读取 1x 图片：{path_1x}")
            skipped += 1
            continue

        # 关键方向：把 3x 的颜色改变为 1x 的颜色
        result = white_balance(img_3x, img_1x)

        ok = imwrite_unicode(save_path, result)
        if ok:
            success += 1
        else:
            print(f"[失败] 无法保存结果：{save_path}")
            skipped += 1

    print(f"[完成] {split_dir} | 3x图片数：{total}，成功：{success}，跳过/失败：{skipped}，保存到：{save_dir}")
    return total, success, skipped


def process_all(root, datasets=None, splits=None, overwrite=False):
    """批量处理 Cutting 根目录下的所有数据集和划分。"""
    root = Path(root)
    datasets = datasets or ["DJI", "rewhite"]
    splits = splits or ["train", "test", "validation"]

    if not root.exists():
        raise FileNotFoundError(f"找不到 Cutting 根目录：{root}")

    all_total = 0
    all_success = 0
    all_skipped = 0

    for dataset in datasets:
        for split in splits:
            split_dir = root / dataset / split
            total, success, skipped = process_one_split(split_dir, overwrite=overwrite)
            all_total += total
            all_success += success
            all_skipped += skipped

    print("=" * 80)
    print(f"全部处理完成 | 3x图片总数：{all_total}，成功：{all_success}，跳过/失败：{all_skipped}")


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir / "Data/Cutting"

    parser = argparse.ArgumentParser(description="批量将 3x 图像颜色匹配到同名 1x 图像，并保存到 3x_method3。")
    parser.add_argument(
        "--root",
        type=str,
        default=str(default_root),
        help="Cutting 根目录。默认使用当前脚本所在目录下的 Cutting 文件夹。"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["DJI", "rewhite"],
        help="需要处理的数据集文件夹名称，默认：DJI rewhite。"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test", "validation"],
        help="需要处理的数据划分，默认：train test validation。"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果输出图片已经存在，是否覆盖。默认不覆盖。"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_all(
        root=args.root,
        datasets=args.datasets,
        splits=args.splits,
        overwrite=args.overwrite
    )
