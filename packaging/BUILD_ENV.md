# SHKit 打包环境（出安装程序用）

出安装程序的生产环境。**只在打包时需要**；终端用户装 `SHKit_Setup_v2.0.1.exe` 即可，
不需要 Python。

## 1. 建环境（conda 建环境，包用 pip 装）

```powershell
# conda 环境（放在 anaconda3\envs 下）
conda create -n shkit-gui -y -c conda-forge python=3.12 pip

# 其余全部用 pip 装：NumPy/SciPy 走 PyPI 的 OpenBLAS 构建，
# 比 conda 的 MKL 构建小 ~500 MB，也不会带进 conda 的 UCRT/运行库冲突
$py = "$env:USERPROFILE\anaconda3\envs\shkit-gui\python.exe"
& $py -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple `
    numpy scipy matplotlib PySide6-Essentials pandas xarray netCDF4 psutil openpyxl pyinstaller

# conda 版 Python 的 _ctypes 需要 Library\bin\ffi-8.dll，pip 不管这个，必须补
conda install -n shkit-gui -y -c conda-forge libffi
```

> **为什么 `pyside6` 不用 conda 装**：conda-forge 的 `pyside6` 会一起拉 `qt6-main`，
> 它塞进 `Library\bin` 的 Qt6 DLL 与 pip 版 PySide6 需要的版本对不上，
> 结果 `import QtWidgets` 直接 `DLL load failed`（WinError 127）。
> 全用 pip 装就没有这个问题。

## 2. 一键出安装程序

```powershell
cd <项目>\SHKit
powershell -ExecutionPolicy Bypass -File packaging\build_installer.ps1 -VerifyInstall
```

做五件事：

1. PyInstaller 按 `packaging\shkit.spec` 打 onedir 目录包 → `D:\SHKit_build\dist\SHKit\`
   （入口是 `packaging\shkit_launcher.py`，**不是** `shkit/gui/app.py`，见下面「踩过的坑」）
2. 校验目录包：说明书/配图/二维码/许可是否都在；说明书里有没有不该出现的开发者内容；
   启动一次 exe；**再跑一遍冻结版自检** `SHKit.exe --self-test`
3. **把整个目录包复制到一个带空格的路径**（`<构建根>\dist space 带空格\SHKit`）下，
   从那里再跑一遍自检 —— 安装路径带空格是常态，这一步专门守它（见下面 §2.1）
4. Inno Setup 编译 `installer_shkit.iss` → `D:\SHKit_build\dist\SHKit_Setup_v2.0.1.exe`
5. （`-VerifyInstall`）把安装程序静默装到一个**目录名带空格**的路径
   （`<构建根>\_install test 带空格`）、校验文件、启动一次、跑一次自检、再卸载

### 2.1 安装路径带空格（必须守住的一条）

用户可能装在 `C:\Program Files\SHKit`、`%LOCALAPPDATA%\Programs\...`，或者用户名本身
带空格。这条路上有三处经典坑，现在都被打包流程**硬性检查**：

| 位置 | 坑 | 现在怎么守 |
| --- | --- | --- |
| `build_installer.ps1` 的静默安装 | `Start-Process -ArgumentList` 是**按空格拆分**的，`/DIR=$TestDir` 没整体加引号就会装到错误的目录 | 写成 `/DIR=`"$TestDir`""，而且验证目录名里**故意留空格**；装完先查 `<dir>\SHKit.exe` 在不在 |
| 冻结版自己启动 | exe 路径带空格时，任何"把路径拼进命令行字符串"的写法都会崩 | 运行代码里**没有** `shell=True`/`os.system`（`validate_v2.py` 里有断言守住）；自检把整个目录包**复制到带空格路径**再跑一遍 |
| `SHKit.exe --self-test` | 只检查"进程活着"抓不到导入期崩溃 | 自检里有两步是带空格路径的（网格 nc 写→读逐值一致、逐历元系数缓存写→读命中），还有一步真的**spawn 一个子进程**并要它返回结果 |

> 冻结版自检那两步（带空格路径、独立子进程）在**目录包**与**装完的版本**上各查一遍，
> 缺任何一项打包脚本就红。

### 冻结版自检（`SHKit.exe --self-test`）

进程"活着"**不代表没崩**：如果崩在导入期，窗口根本不会出现，而 `Start-Process`
可能已经拿到进程句柄、`HasExited` 还是 `false`。所以冻结版必须自己报结果 ——
`--self-test` 会跑完整条关键路径并以退出码表态：

```
[PASS] 主窗口显示  1440x900
[PASS] 生成示例数据  2500 点
[PASS] 运行分析  method=quadrature
[PASS] 渲染地图 / 逐阶谱 / 诊断报告 / 系数表
[PASS] 渲染说明书（含配图）  27 张图，正文 15360 字符
[PASS] 带空格路径：网格 nc 写→读逐值一致  网格 文件.nc 逐值一致；可执行文件 <路径>（含空格=True）
[PASS] 带空格路径：逐历元系数缓存写→读命中  ...shkit-coeffs.npz
[PASS] 独立子进程能起来并返回结果  子进程 pid=… 解出 nmax=2
[PASS] 导出系数文件  4074 字节
```

任何一项失败都会让打包脚本直接报错退出，不会把坏包交出去。

**中间产物一律放在项目外的 `D:\SHKit_build\`**：项目在网盘同步目录里，
同步驱动会锁文件（PyInstaller 清理时会报 `PermissionError: WinError 32`），
而且 netCDF4 在同步盘上写不出文件（`Errno 13`）。

### 常用参数

| 参数 | 作用 |
| --- | --- |
| `-SkipBuild` | 复用已有目录包，只重编安装程序（改 `.iss` 后用） |
| `-SkipInstaller` | 只出目录包，不编安装程序（出绿色版用） |
| `-VerifyInstall` | 额外安装-校验-卸载一遍 |
| `-EnvName xxx` | 换 conda 环境名（默认 `shkit-gui`） |
| `-BuildRoot X:\path` | 换构建根目录（默认 `D:\SHKit_build`） |

## 3. 环境 / 工具要求

| 项目 | 版本 / 位置 |
| --- | --- |
| Python | 3.12（conda 环境 `shkit-gui`） |
| PySide6 | 6.11.2（PySide6-Essentials，pip） |
| PyInstaller | 6.22.x |
| Inno Setup | 6.4.3，`E:\Inno Setup 6\ISCC.exe`（脚本会自动找几个常见位置） |
| 中文语言文件 | `E:\Inno Setup 6\Languages\ChineseSimplified.isl`（Inno 自带没有，需另下） |

## 4. 踩过的坑（都已在脚本/spec 里修好）

| 现象 | 原因 | 修法 |
| --- | --- | --- |
| **exe 双击一闪就没了**，报 `ImportError: attempted relative import with no known parent package` | PyInstaller 把**入口脚本**当顶层脚本执行（没有父包）。入口原本是 `shkit/gui/app.py`，里面的 `from .. import __version__` 找不到父包 | 入口改成包外的 `packaging/shkit_launcher.py`：先 `import shkit.gui.app`（包被正常导入）再调 `main()`。`tests/validate_core.py` 有 6 条断言守住这条 |
| 冻结后启动即崩，`ImportError: DLL load failed while importing _ctypes` | conda 版 Python 的 `_ctypes.pyd` 依赖 `<env>\Library\bin\ffi-8.dll`，PyInstaller 只扫 `DLLs/` 与 site-packages，收不到 | `shkit.spec` 里把 `Library\bin\*.dll` 显式加进 `binaries` |
| 源环境 `import QtWidgets` 就 WinError 127 | conda-forge 的 `qt6-main` 与 pip 版 PySide6 的 Qt6 DLL/运行库版本冲突 | 别用 conda 装 pyside6/qt6-main，全走 pip |
| PyInstaller 中止：`attempt to collect multiple Qt bindings` | 打包环境里同时有 PySide6 与 PyQt5/PyQt6 | `shkit.spec` 的 `ALWAYS_EXCLUDE` |
| 目录包 1.38 GB | Anaconda 版 NumPy 走 Intel MKL（mkl_*.dll ~520 MB）+ 被拖进来的 `node.exe` 88 MB | 用 PyPI 的 NumPy（OpenBLAS）：目录包降到 ~300 MB |
| `ISCC ... 系统找不到指定的文件` | `.iss` 里的相对路径按脚本所在目录解析；项目路径含中文 | 用 `/DProjDir=... /DSourceDir=...` 传绝对路径 |
| `build_installer.ps1` 中文乱码、解析报错 | Windows PowerShell 5.1 把无 BOM 的 UTF-8 当 GBK 读 | 脚本存成 **UTF-8 with BOM** |
| 自检日志中文乱码 | 冻结版（windowed）的 stdout 是 GBK 管道，`PYTHONIOENCODING` 对它无效 | 自检里 `sys.stdout.reconfigure(encoding="utf-8")` |

## 5. 发行包里有什么

```
SHKit.exe                     程序本体
_internal\                    运行所需的全部组件（Python 运行时、Qt、绘图库…）
_internal\data\               勒夫数表 + 离线海岸线（只画海岸线，不画国界）
_internal\docs\               使用说明.html + 使用说明_img\ + 公众号二维码 + 纯文本版
_internal\licenses\           LGPLv3 / GPLv3 / NOTICE
source\shkit\                 软件运行代码（Python 包）
source\tools\ source\tests\ source\examples\   工具脚本、测试文件、示例
source\pyproject.toml source\requirements.txt
```

**不进包**：`方案调研.md`、`SHKit方法总结.html`、`build_guide.py`、探测脚本、
`物理量与单位换算.md`、`许可与闭源商用说明.md`、`说明书备份_*`。
`build_installer.ps1` 会逐项校验这件事。

## 6. 打包链路实测（2026-09-13，本机，端到端跑通）

```powershell
cd SHKit
$env:QT_QPA_PLATFORM = "offscreen"      # 让脚本里的两次"启动存活"检查不弹窗
powershell -ExecutionPolicy Bypass -File packaging\build_installer.ps1 -VerifyInstall
```

| 步骤 | 实测 |
| --- | --- |
| 解释器 | `C:\Users\<你>\anaconda3\envs\shkit-gui\python.exe`（Python 3.12.14，PySide6 6.11.2，PyInstaller 6.22.3，pip/OpenBLAS NumPy） |
| 1/3 PyInstaller | 成功；目录包 **317 MB / 1318 个文件**，`SHKit.exe` **17.1 MB** |
| 2/3 目录包校验 | 10 项必备文件全 OK；说明书引用的图**全部在包里**；不该进包的开发资料一个都没进；说明书内容边界检查通过；exe 启动存活 10 s；**冻结版 `--self-test` 9/9（exit=0）** |
| 3/3 Inno Setup | `ISCC` = `E:\Inno Setup 6\ISCC.exe`；编译 **141 s**；产物 `D:\SHKit_build\dist\SHKit_Setup_v2.0.1.exe` **92 MB** |
| 附加：装一遍再验 | 静默装到 `D:\SHKit_build\_install_test`：9 项文件核对全 OK → 装完的 exe 正常启动 → 装完的版本 `--self-test` **9/9（exit=0）** → 自动卸载并清理，**SCRIPT_EXIT=0** |
| 参照 | 同一个 spec 用**基础 Anaconda 环境**（MKL NumPy）打：目录包 **2078 MB** —— 环境选错只是变大，不影响能不能跑 |

### 两个必须自己补的前提（都会让打包在这一步直接失败）

| 前提 | 不到位的后果 | 怎么办 |
| --- | --- | --- |
| **`ChineseSimplified.isl`** | Inno 自带语言包里**没有简体中文**；缺它 ISCC 在 `[Languages]` 处报错，安装程序编不出来 | 从 <https://jrsoftware.org/files/istrans/>（或社区维护的 kira-96 译本）取 6.5.0+ 版本，放到 `E:\Inno Setup 6\Languages\`。本机已放入 |
| **`build_installer.ps1` / `installer_shkit.iss` 的 UTF-8 BOM** | 脚本交给 Windows PowerShell 5.1 跑，**没有 BOM 就按 GBK 解码** → 中文变乱码 → `Unexpected token` **解析阶段就失败，一行都不执行**；`.iss` 没有 BOM 则 ISCC 按 ANSI 读，中文菜单名乱码 | 保持两个文件为 **UTF-8 with BOM**。BOM 是**看不见的**，任何"以 UTF-8 无 BOM 写回"的编辑器都会静默弄丢它 —— 所以 `tests/validate_v2.py` 里有两条断言**直接查文件前三个字节是否为 `EF BB BF`**，在改完版本号之类的文本编辑之后跑一次就能立刻发现 |

> `SHKit_Setup_v2.0.1.exe` 由 Inno Setup 编译 `.iss` 产生，没有 `ISCC.exe` 时脚本会
> **明确说明并跳过这一步**（`exit 2`），而不是假装成功。此时可退到
> 「冻结目录 + `SHKit.exe --self-test`」这一层：它已经覆盖绝大多数真实故障
> （缺 DLL、缺 hiddenimport、说明书/数据没进包、入口相对导入崩）：

```powershell
cd SHKit
python -m PyInstaller --noconfirm --clean --log-level WARN `
    --distpath D:\SHKit_build\dist --workpath D:\SHKit_build\build packaging\shkit.spec
$exe = "D:\SHKit_build\dist\SHKit\SHKit.exe"
$env:QT_QPA_PLATFORM = "offscreen"      # 无显示环境也能跑
& $exe --self-test ; "EXIT=$LASTEXITCODE"     # 0 = 冻结版关键路径全通
```

`tests/validate_v2.py` 另外把 spec 的 `datas` 清单**求值后逐项核对存在**（6 项）、检查
GPL-only 的 Qt 模块确实在排除表里、只用 onedir（LGPL 要求终端用户能替换 Qt 库），
并守住上面那条 BOM 前提 —— 这三件事不需要 Inno Setup 也能验。
