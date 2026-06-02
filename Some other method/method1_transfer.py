#===========================================================
# 方法1：传统的 lαβ 颜色映射方法 - 批量处理版本
# 功能：将 3x 图像颜色转换为对应 1x 图像的颜色风格，并批量保存到 3x_method1
# 数据结构：
#   Cutting/
#       DJI/
#           train/1x, train/3x
#           test/1x, test/3x
#           validation/1x, validation/3x
#       rewhite/
#           train/1x, train/3x
#           test/1x, test/3x
#           validation/1x, validation/3x
#===========================================================

import argparse
import os
from pathlib import Path

import cv2
from color_transfer import color_transfer


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def str2bool(v):
    """兼容命令行中的 True/False 输入。"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def read_image_bgr(image_path):
    """读取图像，并在读取失败时给出明确报错。"""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"图像读取失败：{image_path}")
    return image


def find_reference_image(ref_dir, image_name):
    """
    在 1x 文件夹中寻找与 3x 同名的参考图。
    优先完全同名匹配；如果后缀不同，则按 stem 匹配常见图像后缀。
    """
    ref_path = ref_dir / image_name
    if ref_path.exists():
        return ref_path

    stem = Path(image_name).stem
    candidates = []
    for ext in IMAGE_EXTENSIONS:
        p = ref_dir / f"{stem}{ext}"
        if p.exists():
            candidates.append(p)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(f"找到多个同名参考图，请检查文件：{stem}，目录：{ref_dir}")

    return None


def process_one_pair(path_1x, path_3x, save_path, clip=True, preserve_paper=True):
    """
    将 3x 图像颜色转换为 1x 图像颜色。

    color_transfer(source, target) 的使用逻辑：
        source: 颜色参考图，这里是 1x
        target: 需要被改变颜色的图，这里是 3x
    """
    img_1x = read_image_bgr(path_1x)
    img_3x = read_image_bgr(path_3x)

    # 把 3x 的图颜色改变为 1x 的颜色
    try:
        result = color_transfer(img_1x, img_3x, clip=clip, preserve_paper=preserve_paper)
    except TypeError:
        # 兼容某些 color_transfer.py 只支持 color_transfer(source, target) 的版本
        result = color_transfer(img_1x, img_3x)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(save_path), result)
    if not ok:
        raise IOError(f"结果保存失败：{save_path}")


def process_split(split_dir, overwrite=False, clip=True, preserve_paper=True):
    """处理单个 split，例如 Cutting/DJI/train。"""
    dir_1x = split_dir / "1x"
    dir_3x = split_dir / "3x"
    save_dir = split_dir / "3x_method1"

    if not dir_1x.exists():
        print(f"[跳过] 找不到 1x 文件夹：{dir_1x}")
        return 0, 0
    if not dir_3x.exists():
        print(f"[跳过] 找不到 3x 文件夹：{dir_3x}")
        return 0, 0

    images_3x = sorted([p for p in dir_3x.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
    if len(images_3x) == 0:
        print(f"[跳过] 3x 文件夹中没有图像：{dir_3x}")
        return 0, 0

    success_count = 0
    fail_count = 0

    for path_3x in images_3x:
        path_1x = find_reference_image(dir_1x, path_3x.name)
        if path_1x is None:
            print(f"[缺少参考图] 3x 图像没有对应 1x：{path_3x.name}")
            fail_count += 1
            continue

        save_path = save_dir / path_3x.name
        if save_path.exists() and not overwrite:
            print(f"[已存在，跳过] {save_path}")
            continue

        try:
            process_one_pair(
                path_1x=path_1x,
                path_3x=path_3x,
                save_path=save_path,
                clip=clip,
                preserve_paper=preserve_paper,
            )
            success_count += 1
        except Exception as e:
            print(f"[失败] {path_3x.name}: {e}")
            fail_count += 1

    print(f"[完成] {split_dir} -> {save_dir}，成功 {success_count} 张，失败 {fail_count} 张")
    return success_count, fail_count


def main():
    parser = argparse.ArgumentParser(
        description="批量将 Cutting 中 3x 图像颜色转换为对应 1x 图像颜色，并保存到 3x_method1。"
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Cutting 文件夹路径。默认使用当前脚本所在目录下的 Cutting 文件夹。",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["DJI", "rewhite"],
        help="需要处理的数据文件夹名称，默认：DJI rewhite。",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test", "validation"],
        help="需要处理的子集，默认：train test validation。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果结果已存在，是否覆盖保存。默认不覆盖。",
    )
    parser.add_argument(
        "--clip",
        type=str2bool,
        default=True,
        help="是否对 Lab 数值进行 clip，默认 True。",
    )
    parser.add_argument(
        "--preservePaper",
        type=str2bool,
        default=True,
        help="是否严格遵循原论文方法，默认 True。",
    )
    args = parser.parse_args()

    if args.root is None:
        root = Path(__file__).resolve().parent / "Data/Cutting"
    else:
        root = Path(args.root)

    print(f"Root: {root}")
    print("任务：把 3x 图像颜色改变为对应 1x 图像颜色")
    print("输出文件夹：3x_method1")

    total_success = 0
    total_fail = 0

    for dataset in args.datasets:
        for split in args.splits:
            split_dir = root / dataset / split
            success, fail = process_split(
                split_dir=split_dir,
                overwrite=args.overwrite,
                clip=args.clip,
                preserve_paper=args.preservePaper,
            )
            total_success += success
            total_fail += fail

    print("=" * 60)
    print(f"全部处理完成：成功 {total_success} 张，失败 {total_fail} 张")
    print("=" * 60)


if __name__ == "__main__":
    main()
