import os
import re
import shutil
from collections import defaultdict

# ==============================
# 需要的 accuracy 顺序
# ==============================
accuracy_priority = [
    "accuracy_1_task1",
    "accuracy_1_task2",
    "accuracy_2_task2",
    "accuracy_1_task3",
    "accuracy_2_task3",
    "accuracy_3_task3",
]

# ==============================
# 不需要求平均和偏差的字段
# 这些一般是配置项：同一组实验中应该是完全一致的
# ==============================
NO_AVG_KEYS = {"n_epochs", "lr", "batch_size", "buffer_size", "model"}

# ==============================
# 通用：对一个字段的一批值求均值和标准差
# values 可以是 ['99.9', '81.5', '98.8'] 或 [99.9, 81.5, 98.8] 等
# ==============================
def format_mean_and_diffs(key, values):
    """
    返回形如：
        key: mean ± std

    如果不是纯数字，就原样打印列表。

    对于在 NO_AVG_KEYS 里的字段，不做均值和标准差，直接输出：
        - 如果所有值都相同： key: value
        - 否则： key: [v1, v2, v3, ...]
    """
    # 配置字段：不做均值和标准差
    if key in NO_AVG_KEYS:
        # 去掉 None
        non_none_vals = [v for v in values if v is not None]
        if not non_none_vals:
            return f"{key}: {values}"

        # 如果全部相同，打印单个值
        if all(v == non_none_vals[0] for v in non_none_vals):
            return f"{key}: {non_none_vals[0]}"
        # 否则就直接把列表打印出来
        return f"{key}: {non_none_vals}"

    # 下面是“真正要算均值和标准差”的字段
    nums = []
    for v in values:
        if v is None:
            return f"{key}: {values}"
        s = str(v)
        # 从字符串中提取第一个数字，比如 '99.9', ' 99.9 ' 都可以
        m = re.search(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", s)
        if not m:
            return f"{key}: {values}"
        nums.append(float(m.group(0)))

    if not nums:
        return f"{key}: {values}"

    # 计算均值
    mean_val = sum(nums) / len(nums)

    # 计算标准差（使用样本标准差；如果只有一个值，则 std=0）
    if len(nums) == 1:
        std_val = 0.0
    else:
        var = sum((x - mean_val) ** 2 for x in nums) / (len(nums) - 1)
        std_val = var ** 0.5

    return f"{key}: {mean_val:.6f} ± {std_val:.6f}"


# ==============================
# 字段提取
# ==============================
def extract_fields(path):
    with open(path, "rb") as f:
        content = f.read().decode("utf-8", errors="ignore")

    result = {}

    # 字符串字段
    str_fields = ["model"]
    for field in str_fields:
        result[field] = re.findall(rf"'{field}'\s*:\s*'([^']+)'", content)

    # 普通数字字段
    num_fields = ["n_epochs", "lr", "buffer_size", "batch_size"]
    for field in num_fields:
        result[field] = re.findall(rf"'{field}'\s*:\s*([0-9.+-eE]+)", content)

    # 精度字段（按你指定顺序）
    for field in accuracy_priority:
        result[field] = re.findall(rf"'{field}'\s*:\s*([0-9.+-eE]+)", content)

    # np.float64 字段
    np_fields = [
        "accmean_task1",
        "accmean_task2",
        "accmean_task3",
        "forward_transfer",
        "backward_transfer",
        "forgetting",
    ]
    for field in np_fields:
        result[field] = re.findall(rf"'{field}'\s*:\s*np\.float64\(([^)]+)\)", content)

    return result


# =========================================================
# 1. 遍历大文件夹 class-il 下的所有 seq-nct-224* 子文件夹
# 2. 同名 model 聚合在一起
# =========================================================
base_dir = "class-il"
reg_folders = [
    d
    for d in os.listdir(base_dir)
    if d.startswith("seq-nct-224") and os.path.isdir(os.path.join(base_dir, d))
]

# aggregated[model_name][reg_name] = [ run1_dict, run2_dict, ... ]
aggregated = defaultdict(lambda: defaultdict(list))

for reg_name in reg_folders:
    reg_path = os.path.join(base_dir, reg_name)
    # reg_path 下面就是各个 model，比如 dualprompt
    for model_name in os.listdir(reg_path):
        model_path = os.path.join(reg_path, model_name)
        logs_file = os.path.join(model_path, "logs.pyd")
        if not os.path.isfile(logs_file):
            continue

        fields = extract_fields(logs_file)
        num_models = len(fields["model"])

        # 把每一次实验（索引 idx）保存为一个 dict
        for idx in range(num_models):
            run_data = {}
            for key, values in fields.items():
                run_data[key] = values[idx] if idx < len(values) else None

            # 让 accmean_average 和 accmean_task3 相同
            accmean_t3_val = run_data.get("accmean_task3")
            run_data["accmean_average"] = accmean_t3_val

            aggregated[model_name][reg_name].append(run_data)

# =========================================================
# 原始结果打印（不 merge）：按 model / reg 分组
# （这里我保持你原来的逐条打印，不做均值）
# =========================================================
for model_name, reg_dict in aggregated.items():
    print(f"\n\n==================== MODEL: {model_name} ====================")
    for reg_name, runs in reg_dict.items():
        print(f"\n---------- Nomin method: {reg_name} ----------")
        for i, run in enumerate(runs, 1):
            print(f"\n===== Experiment {i} ({reg_name})=====")
            for key, value in run.items():
                print(f"{key}: {value}")

# =========================================================
# Merge 版本：相同配置（最多 3 次）合并，输出到 resultYZ 目录
# 每个字段：均值 + 标准差（但 NO_AVG_KEYS 中的不算）
# =========================================================

# 定义“配置相同”的字段（可按需增减）
config_fields = ["model", "n_epochs", "lr", "buffer_size", "batch_size"]

# 结果根目录
result_root = "resultYZ"

# 每次运行都完全刷新 resultYZ
if os.path.isdir(result_root):
    shutil.rmtree(result_root)
os.makedirs(result_root, exist_ok=True)

# 按 model / reg 输出到各自文件
for model_name, reg_dict in aggregated.items():
    # 为每个 model 建一个子文件夹
    model_dir = os.path.join(result_root, model_name)
    os.makedirs(model_dir, exist_ok=True)

    for reg_name, runs in reg_dict.items():
        # 每个 reg_name 一个 .txt 文件
        merge_lines = []
        merge_lines.append(f"==================== MODEL: {model_name} ====================")
        merge_lines.append(f"\n---------- Nomin method: {reg_name} ----------")

        # 按配置分组：key = (model, n_epochs, lr, buffer_size, batch_size)
        config_groups = defaultdict(list)
        for run in runs:
            cfg_key = tuple(run.get(field) for field in config_fields)
            config_groups[cfg_key].append(run)

        merged_idx = 0

        # 对每一组相同配置的实验进行 merge（每组最多 3 个为一批）
        for cfg_key, run_list in config_groups.items():
            for start in range(0, len(run_list), 3):
                chunk = run_list[start : start + 3]
                if not chunk:
                    continue

                merged_idx += 1
                merge_count = len(chunk)
                merge_lines.append(
                    f"\n===== MERGED Experiment {merged_idx} ({reg_name}) | merge_count = {merge_count} ====="
                )

                # 收集该批次所有出现过的字段
                all_keys = set()
                for r in chunk:
                    all_keys.update(r.keys())

                # -------------------------
                # 先打印“配置”部分：优先 n_epochs 和 lr
                # -------------------------
                merge_lines.append("Config:")
                config_priority = ["n_epochs", "lr"]
                printed_keys = set()

                for key in config_priority:
                    if key in all_keys:
                        values = [r.get(key) for r in chunk]
                        line = format_mean_and_diffs(key, values)
                        merge_lines.append(" " + line)
                        printed_keys.add(key)

                # 也把其余配置字段（batch_size, buffer_size, model）往后补上
                for key in ["batch_size", "buffer_size", "model"]:
                    if key in all_keys and key not in printed_keys:
                        values = [r.get(key) for r in chunk]
                        line = format_mean_and_diffs(key, values)
                        merge_lines.append(" " + line)
                        printed_keys.add(key)

                # -------------------------
                # 再按指定顺序输出 accuracy_X_X
                # -------------------------
                for key in accuracy_priority:
                    if key in all_keys and key not in printed_keys:
                        values = [r.get(key) for r in chunk]
                        line = format_mean_and_diffs(key, values)
                        merge_lines.append(line)
                        printed_keys.add(key)

                # -------------------------
                # 最后输出剩余字段（包括 accmean_*、forgetting 等）
                # -------------------------
                other_keys = sorted(all_keys - printed_keys)
                for key in other_keys:
                    values = [r.get(key) for r in chunk]
                    line = format_mean_and_diffs(key, values)
                    merge_lines.append(line)

        # 写入当前 model / reg 的文件
        out_path = os.path.join(model_dir, f"{reg_name}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(merge_lines))

print(f"\n\n[INFO] Merge results written to folder: {result_root}")
