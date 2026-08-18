# test/ — 共享算子测试框架

给所有算子共用的正确性验证 + 性能基准框架（CPU / GPU）。GEMM 是第一个接入的算子。

## 1. 目录结构

```
test/
├── README.md      ← 本文件
├── __init__.py    ← 公共 API 导出（from test import *）
├── config.py      ← 全局配置：数据类型、容差、TEST_BACKENDS、空闲GPU阈值、
│                    OPERATORS（算子来源登记）、BASELINES（基线覆盖）
├── gpu.py         ← 空闲 GPU 自动选择（nvidia-smi，无空闲则打印状态并退出）
├── ops.py         ← load_op()：按 config.OPERATORS 加载算子文件夹中的 op.py
├── registry.py    ← 注册表：register_op / register_baseline、OpImpl / OpResult / SweepResult
├── verify.py      ← 正确性验证：与算子 reference（NumPy）比对
├── benchmark.py   ← 性能基准：GPU CUDA Event / CPU 自适应 wall-clock
├── plot.py        ← 绘图：plot_comparison（分后端 GFLOPS + speedup）、plot_roofline
└── native.py      ← 通用 C++/CUDA 构建加载：NativeLib（版本化 .so + mtime 自动重编译）、
                     compile_cpp / compile_cu
```

## 2. 接入一个新算子（三步）

### Step 1 — 在 `test/config.py` 登记算子来源

```python
OPERATORS = {
    "gemm":    "1_operators/1.1-GEMM",   # 算子名 → 算子文件夹（相对仓库根目录）
    "softmax": "1_operators/1.4-Softmax",
}
```

### Step 2 — 在算子文件夹中写 `op.py`

实现一个小协议（模块级函数，必选项会由 `load_op` 校验）：

| 属性 | 必选 | 说明 |
|---|---|---|
| `NAME` | ✓ | 与 config.OPERATORS 的 key 对应 |
| `DEFAULT_SIZES` | ✓ | verify_all 的默认尺寸列表（元素为任意 tuple） |
| `gen_inputs(size, backend, seed)` | ✓ | 生成输入：`backend="gpu"` 返回 cupy 数组、`"cpu"` 返回 numpy 数组 |
| `metric(size, time_ms)` | ✓ | 指标值（GFLOPS / GB/s / tokens/s …） |
| `register()` | ✓ | 用 `register_op` 注册实现、用 `register_baseline` 指定默认基线 |
| `reference(*args)` | 推荐 | 正确性参考（NumPy FP32）；缺失时退回用 CPU 基线实现的输出 |
| `prepare()` | 可选 | 构建/加载/注册原生实现（C++/CUDA）；Results cell 调用 |
| `SIZE_LABEL` | 可选 | x 轴标签（默认 "Size"） |
| `SIZE_COLS` | 可选 | 表格尺寸列名（默认 ["Size"]，渲染为 "128×256×512"） |
| `METRIC_LABEL` | 可选 | 指标名（默认 "Metric"） |
| `METRIC_FMT` | 可选 | 指标格式（默认 ".3f"，写无花括号的格式体） |
| `size_key(size)` | 可选 | 绘图 x 值（默认 size[0]） |
| `flops(size)` / `bytes_read(size)` | 可选 | 仅 roofline 用（缺失时 roofline 会报错） |

关键约定：**实现函数签名是 `fn(*args) -> out`**，args 来自 `gen_inputs`；
标量参数（M/N/K 等）一律从数组 shape 推导，因此注册的实现必须对尺寸通用。
原生实现（C++/CUDA）用 `test.native` 的 `NativeLib` 构建加载（版本化 `.so`，
改源码后 `prepare()` 自动重编译并重载），ctypes wrapper 写在 op.py 里
（各算子签名不同）。参考实现见 `1_operators/1.1-GEMM/op.py`。

### Step 3 — 在 notebook 中调用

```python
# 把仓库根目录加入 sys.path（向上找包含 test/ 的目录）
import sys, pathlib, os
_nb = pathlib.Path(globals().get("__vsc_ipynb_file__") or (globals().get("_dh") or [""])[0] or os.getcwd())
_d = _nb.parent if str(_nb).endswith(".ipynb") else _nb
while not (_d / "test" / "ops.py").exists() and _d.parent != _d:
    _d = _d.parent
if str(_d) not in sys.path:
    sys.path.insert(0, str(_d))

from test import *
import test.config as config   # 动态配置（改这里无需重启 kernel）

op = load_op("gemm")           # 注册 Python 实现与基线
device = select_gpu()          # 自动选择空闲 GPU（CPU-only 模式跳过）

op.prepare()                   # 构建/加载/注册 C++/CUDA 实现

verify_result = verify_all(op, backends="gpu")     # 或 "cpu" / "all" / None
sweep = benchmark_all(op, SQUARE_SIZES, backends="gpu")
plot_comparison(sweep, title="GEMM Performance")
plot_roofline(sweep)            # 仅 GFLOPS 算子（需 flops/bytes_read）
```

常用配置（`test/config.py`，均支持运行时改 `config.X` 动态生效）：

- `TEST_BACKENDS` — "all" / "gpu" / "cpu"：只测哪些后端；"cpu" 模式完全不碰 GPU
- `BASELINES` — 覆盖算子默认 speedup 基线，如 `{"gemm": {"gpu": "cuBLAS (cupy)"}}`
- `IDLE_UTIL_MAX` / `IDLE_MEM_MAX_GB` / `IDLE_SAMPLES` — 空闲 GPU 判定阈值

## 3. 设计要点

- **基线两层机制**：正确性参考（`op.reference`，NumPy）与 speedup 基线
  （注册实现，`register_baseline` 指定默认 + `config.BASELINES` 覆盖）刻意分开——
  保持"手写实现 vs 官方实现"有真实数值误差的现状
- **后端自适应计时**：GPU 用 CUDA Event；CPU 用 `perf_counter` + 自适应迭代次数
  （每个数据点约 1.5s）；慢实现用 `max_size` 门控
- **版本化编译产物**：`<name>.<mtime>-<size>.so`，改源码 → `prepare()` 重编译到
  新路径并重新 dlopen（同路径 dlopen 会返回旧映射），旧产物自动清理
- **live config**：框架模块运行时读 `config.X`，never star-import 快照
