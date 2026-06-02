#========================================================
# 功能：
#   以 1x 图像作为颜色参考，将同名 3x 图像的颜色改变为 1x 图像的颜色。
#   不同颜色迁移方法的结果分别保存到：
#       3x_method4, 3x_method5, 3x_method6, 3x_method7, 3x_method8
#   如启用 DMSCT 神经网络方法，则保存到：
#       3x_method9
#
# 数据结构：
#   Data/Cutting/
#       DJI/
#           train/1x, train/3x
#           test/1x, test/3x
#           validation/1x, validation/3x
#       rewhite/
#           train/1x, train/3x
#           test/1x, test/3x
#           validation/1x, validation/3x
#
# 运行示例：
#   python method4_transfer_batch.py
#   python method4_transfer_batch.py --root E:/Code/DJI/Data/Cutting
#   python method4_transfer_batch.py --overwrite
#   python method4_transfer_batch.py --run_dmsct --ckpt_path F:/DJ/Code/colour_transfer/method4_color-transfer-master/lightning_logs/version_9/checkpoints/epoch=9-step=2640.ckpt
#========================================================

import argparse
from pathlib import Path
import traceback

import numpy as np
from PIL import Image
from skimage.util import img_as_float

# 原项目中的传统方法
from methods.linear import color_transfer_between_images as ct
from methods.linear import color_transfer_in_correlated_color_space as ct_ccs
from methods.linear import monge_kantorovitch_color_transfer as mkct
from methods.iterative import iterative_distribution_transfer as idt
from methods.iterative import automated_color_grading as acg


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def infer_default_root():
    """
    自动推断默认 Data/Cutting 路径。
    如果本脚本放在 E:/Code/DJI/method4_color-transfer-master/ 下，
    默认会优先寻找 E:/Code/DJI/Data/Cutting。
    """
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "Data" / "Cutting",
        script_dir.parent / "Data" / "Cutting",
        Path.cwd() / "Data" / "Cutting",
        Path.cwd() / "Cutting",
    ]

    for p in candidates:
        if p.exists():
            return p

    # 如果都不存在，返回最符合当前工程结构的默认路径
    return script_dir.parent / "Data" / "Cutting"


def load_rgb_float(image_path):
    """
    读取图像，返回 RGB float 图像，范围约为 [0, 1]。
    """
    img = Image.open(image_path).convert("RGB")
    return img_as_float(img)


def save_rgb_float(image, save_path):
    """
    保存 RGB float 图像。
    自动处理 NaN、Inf、越界值，并转换到 uint8。
    """
    image = np.asarray(image)
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    image = np.clip(image, 0.0, 1.0)

    image_u8 = (image * 255.0 + 0.5).astype(np.uint8)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_u8, mode="RGB").save(save_path)


def build_stem_map(folder):
    """
    建立 stem -> path 的映射。
    这样即使同名图像扩展名略有不同，也可以匹配。
    """
    mapping = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            mapping[p.stem] = p
    return mapping


def get_methods():
    """
    method4: Global Linear Method - color_transfer_between_images
    method5: Global Linear Method - color_transfer_in_correlated_color_space
    method6: Global Linear Method - monge_kantorovitch_color_transfer
    method7: Iterative Local Method - iterative_distribution_transfer
    method8: Iterative Local Method - automated_color_grading
    """
    return [
        ("3x_method4", "color_transfer_between_images", ct),
        ("3x_method5", "color_transfer_in_correlated_color_space", ct_ccs),
        ("3x_method6", "monge_kantorovitch_color_transfer", mkct),
        ("3x_method7", "iterative_distribution_transfer", idt),
        ("3x_method8", "automated_color_grading", acg),
    ]


def load_dmsct_runner(ckpt_path, device_name=None):
    """
    可选：加载 DMSCT 神经网络方法。
    如果不使用 --run_dmsct，则不会加载 torch/kornia，也不会要求安装这些包。
    """
    import torch
    from kornia import image_to_tensor, tensor_to_image
    from methods.dmsct import DMSCT

    if device_name is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(device_name)

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 DMSCT 权重文件：{ckpt_path}")

    dmsct = DMSCT.load_from_checkpoint(str(ckpt_path), map_location=device)
    dmsct.to(device)
    dmsct.eval()

    @torch.no_grad()
    def run_dmsct(target, reference):
        target_t = image_to_tensor(target, keepdim=False).float().to(device)
        reference_t = image_to_tensor(reference, keepdim=False).float().to(device)
        result = dmsct(target_t, reference_t)
        return tensor_to_image(result)

    return run_dmsct


def process_one_pair(target_3x_path, reference_1x_path, output_paths, methods, overwrite=False):
    """
    处理一组同名图像：
    target_3x_path: 被改变颜色的 3x 图像
    reference_1x_path: 颜色参考 1x 图像
    output_paths: 每个方法对应的输出路径
    """
    target = load_rgb_float(target_3x_path)
    reference = load_rgb_float(reference_1x_path)

    for (out_dir_name, method_name, method_func), save_path in zip(methods, output_paths):
        if save_path.exists() and not overwrite:
            continue

        try:
            # 关键方向：
            # target 是 3x，被改变颜色；
            # reference 是 1x，作为颜色参考。
            result = method_func(target, reference)
            save_rgb_float(result, save_path)
        except Exception as e:
            print(f"[失败] {method_name}: {target_3x_path.name}")
            print(f"       原因: {e}")


def process_split(root, dataset_name, split_name, methods, overwrite=False):
    split_dir = root / dataset_name / split_name
    dir_1x = split_dir / "1x"
    dir_3x = split_dir / "3x"

    if not dir_1x.exists() or not dir_3x.exists():
        print(f"[跳过] 找不到目录：{dir_1x} 或 {dir_3x}")
        return 0, 0

    map_1x = build_stem_map(dir_1x)
    map_3x = build_stem_map(dir_3x)

    if len(map_1x) == 0 or len(map_3x) == 0:
        print(f"[跳过] 空目录：{dir_1x} 或 {dir_3x}")
        return 0, 0

    common_stems = sorted(set(map_1x.keys()) & set(map_3x.keys()))
    missing_1x = sorted(set(map_3x.keys()) - set(map_1x.keys()))
    missing_3x = sorted(set(map_1x.keys()) - set(map_3x.keys()))

    if missing_1x:
        print(f"[警告] {dataset_name}/{split_name}: 有 {len(missing_1x)} 张 3x 图像找不到同名 1x 参考图")
    if missing_3x:
        print(f"[警告] {dataset_name}/{split_name}: 有 {len(missing_3x)} 张 1x 图像找不到同名 3x 图像")

    for out_dir_name, _, _ in methods:
        (split_dir / out_dir_name).mkdir(parents=True, exist_ok=True)

    total_pairs = len(common_stems)
    processed_pairs = 0

    print(f"\n[开始] {dataset_name}/{split_name}: 共 {total_pairs} 组同名图像")

    for idx, stem in enumerate(common_stems, start=1):
        path_1x = map_1x[stem]
        path_3x = map_3x[stem]

        output_paths = [
            split_dir / out_dir_name / path_3x.name
            for out_dir_name, _, _ in methods
        ]

        process_one_pair(
            target_3x_path=path_3x,
            reference_1x_path=path_1x,
            output_paths=output_paths,
            methods=methods,
            overwrite=overwrite,
        )

        processed_pairs += 1
        if idx == 1 or idx % 20 == 0 or idx == total_pairs:
            print(f"  进度: {idx}/{total_pairs}")

    return processed_pairs, total_pairs


def main():
    parser = argparse.ArgumentParser(description="Batch color transfer: change 3x images to match 1x color reference.")
    parser.add_argument(
        "--root",
        type=str,
        default=str(infer_default_root()),
        help="Root directory of dataset. Default tries to locate Data/Cutting automatically.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["DJI", "rewhite"],
        help="Dataset folders under root.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test", "validation"],
        help="Split folders under each dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output images.",
    )
    parser.add_argument(
        "--run_dmsct",
        action="store_true",
        help="Also run DMSCT neural network method and save to 3x_method9.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="F:/DJ/Code/colour_transfer/method4_color-transfer-master/lightning_logs/version_9/checkpoints/epoch=9-step=2640.ckpt",
        help="Checkpoint path for DMSCT. Only used when --run_dmsct is enabled.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for DMSCT, e.g. cuda or cpu. Default: cuda if available else cpu.",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    print(f"[数据根目录] {root}")

    if not root.exists():
        raise FileNotFoundError(
            f"找不到数据根目录：{root}\n"
            f"请确认 Data/Cutting 是否存在，或使用 --root 手动指定，例如：\n"
            f"python method4_transfer_batch.py --root E:/Code/DJI/Data/Cutting"
        )

    methods = get_methods()

    if args.run_dmsct:
        try:
            dmsct_runner = load_dmsct_runner(args.ckpt_path, args.device)
            methods.append(("3x_method9", "DMSCT", dmsct_runner))
            print("[DMSCT] 已启用，结果将保存到 3x_method9")
        except Exception:
            print("[DMSCT] 加载失败，已跳过 DMSCT。详细错误如下：")
            traceback.print_exc()

    total_processed = 0
    total_pairs = 0

    for dataset_name in args.datasets:
        for split_name in args.splits:
            processed, pairs = process_split(
                root=root,
                dataset_name=dataset_name,
                split_name=split_name,
                methods=methods,
                overwrite=args.overwrite,
            )
            total_processed += processed
            total_pairs += pairs

    print("\n[完成]")
    print(f"共处理图像组数：{total_processed}/{total_pairs}")
    print("输出文件夹：")
    for out_dir_name, method_name, _ in methods:
        print(f"  {out_dir_name}: {method_name}")


if __name__ == "__main__":
    main()
