# =============================================================================
#  SHKit —— 一键出安装程序
# =============================================================================
#  用法（在 SHKit 目录下）：
#      powershell -ExecutionPolicy Bypass -File packaging\build_installer.ps1
#
#  默认用 conda 环境  C:\Users\<你>\anaconda3\envs\shkit-gui
#  （conda 建环境，包用 pip 装；numpy/scipy 用 PyPI 的 OpenBLAS 构建，
#    比 conda 的 MKL 构建小很多，也不会带进 conda 的 UCRT 运行时冲突）
#  中间产物放在项目**外**的  D:\SHKit_build\
#  （项目在网盘同步目录里：放外面可避免同步锁文件，netCDF4 也才能正常写盘）
#
#  参数：
#    -Python         指定解释器（默认自动找 conda 环境）
#    -EnvName        conda 环境名（默认 shkit-gui）
#    -BuildRoot      项目外的构建根目录（默认 D:\SHKit_build）
#    -SkipBuild      跳过 PyInstaller（复用已有目录包）
#    -SkipInstaller  只出目录包，不编译安装程序
#    -VerifyInstall  额外把生成的安装程序静默装到临时目录、校验、再卸载
# =============================================================================
[CmdletBinding()]
param(
    [string]$Python = "",
    [string]$EnvName = "shkit-gui",
    [string]$BuildRoot = "D:\SHKit_build",
    [switch]$SkipBuild,
    [switch]$SkipInstaller,
    [switch]$VerifyInstall
)

$ErrorActionPreference = "Stop"
$Project   = Split-Path -Parent $PSScriptRoot          # …\SHKit
$DistRoot  = Join-Path $BuildRoot "dist"
$Dist      = Join-Path $DistRoot "SHKit"
$Work      = Join-Path $BuildRoot "build"
$Icon      = Join-Path $BuildRoot "SHKit.ico"

function Find-Python {
    if ($Python) { return $Python }
    $cands = @(
        "C:\Users\$env:USERNAME\anaconda3\envs\$EnvName\python.exe",
        "C:\ProgramData\anaconda3\envs\$EnvName\python.exe",
        "D:\anaconda3\envs\$EnvName\python.exe"
    )
    foreach ($c in $cands) {
        if ($c -and (Test-Path $c)) {
            & $c -c "import PySide6, matplotlib, PyInstaller" 2>$null
            if ($LASTEXITCODE -eq 0) { return $c }
        }
    }
    throw "找不到 conda 环境 '$EnvName'（或其中缺 PySide6/matplotlib/PyInstaller）。建法见 packaging\BUILD_ENV.md"
}

function Find-Iscc {
    $cands = @(
        "E:\Inno Setup 6\ISCC.exe",
        "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        "C:\Program Files\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    ) + (Get-Command ISCC -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)
    foreach ($c in $cands) { if ($c -and (Test-Path $c)) { return $c } }
    return ""
}

Write-Host "项目目录  : $Project" -ForegroundColor Cyan
Write-Host "构建根目录: $BuildRoot" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null

# ----------------------------------------------------------------- 1. 目录包
$Py = Find-Python
Write-Host "`n=== 1/4  PyInstaller ===" -ForegroundColor Cyan
Write-Host "解释器：$Py"
if (-not $SkipBuild) {
    & $Py -m PyInstaller --clean --noconfirm `
        --distpath $DistRoot --workpath $Work `
        (Join-Path $Project "packaging\shkit.spec")
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller 失败（exit $LASTEXITCODE）" }
} else {
    Write-Host "（-SkipBuild：复用已有目录包）"
}

if (-not (Test-Path (Join-Path $Dist "SHKit.exe"))) { throw "没有生成 $Dist\SHKit.exe" }
$sz = Get-ChildItem $Dist -Recurse -File | Measure-Object -Property Length -Sum
Write-Host ("目录包：{0:N0} MB，{1} 个文件" -f ($sz.Sum/1MB), $sz.Count) -ForegroundColor Green

# ------------------------------------------------------------- 2. 校验目录包
Write-Host "`n=== 2/4  校验 PyInstaller 目录包 ===" -ForegroundColor Cyan
$bad = 0
$must = @(
    "SHKit.exe",
    "_internal\data\coastline_110m.npz",
    "_internal\data\love_numbers.npy",
    "_internal\licenses\LGPL-3.0.txt",
    "_internal\licenses\NOTICE.txt",
    "_internal\docs\使用说明.html",
    "_internal\docs\使用说明_GUI.md",
    "_internal\docs\地球重力与人类生活TVGG.jpg",
    "_internal\docs\使用说明_img\gui_01_demo_main.png",
    "_internal\docs\使用说明_img\panel_params.png"
)
foreach ($rel in $must) {
    if (Test-Path (Join-Path $Dist $rel)) {
        Write-Host ("  OK   {0}" -f $rel) -ForegroundColor DarkGreen
    } else {
        Write-Host ("  缺失 {0}" -f $rel) -ForegroundColor Red; $bad++
    }
}

# 说明书引用的每张图都必须在包里
$htmlPath = Join-Path $Dist "_internal\docs\使用说明.html"
$html = Get-Content $htmlPath -Raw -Encoding UTF8
$refs = [regex]::Matches($html, 'src="([^"]+)"') | ForEach-Object { $_.Groups[1].Value }
$missImg = @($refs | Where-Object { -not (Test-Path (Join-Path $Dist "_internal\docs\$_")) })
if ($missImg.Count) {
    Write-Host ("  说明书引用的图缺失：{0}" -f ($missImg -join ', ')) -ForegroundColor Red; $bad++
} else {
    Write-Host ("  OK   说明书引用的 {0} 张图全部在包里" -f $refs.Count) -ForegroundColor DarkGreen
}

# 不该进发行包的开发资料
foreach ($rel in @("_internal\docs\方案调研.md", "_internal\docs\SHKit方法总结.html",
                   "_internal\docs\build_guide.py", "_internal\docs\物理量与单位换算.md",
                   "_internal\docs\许可与闭源商用说明.md", "_internal\docs\说明书备份_20260913")) {
    if (Test-Path (Join-Path $Dist $rel)) {
        Write-Host ("  不该出现：{0}" -f $rel) -ForegroundColor Red; $bad++
    }
}

# 说明书内容边界（面向第三方使用者的版本）
foreach ($w in @("Slepian", "pip install", "命令行", "PyInstaller", "docs/方案调研",
                 "build_guide", "check_licensing")) {
    if ($html.Contains($w)) { Write-Host ("  说明书里不该出现：{0}" -f $w) -ForegroundColor Red; $bad++ }
}

# 启动一次，确认 GUI 真能起来
$exe = Join-Path $Dist "SHKit.exe"
$p = Start-Process $exe -PassThru
Start-Sleep -Seconds 10
if ($p.HasExited) {
    Write-Host ("  exe 启动即退出（exit={0}）" -f $p.ExitCode) -ForegroundColor Red; $bad++
} else {
    Write-Host "  OK   exe 启动正常（GUI 存活 10 秒）" -ForegroundColor DarkGreen
    Stop-Process -Id $p.Id -Force
}

# 冻结版自检：进程"活着"不代表没崩。导入期就崩的话窗口根本不会出现，
# 而 Start-Process 可能已经拿到了进程句柄、HasExited 还是 false（踩过这个坑：
# 入口用了相对导入，exe 一启动就 ImportError，却"存活"了 10 秒）。
# 所以让 exe 自己把关键路径跑一遍并报结果。
$stOut = Join-Path $BuildRoot "selftest.log"
$stErr = Join-Path $BuildRoot "selftest.err"
Remove-Item $stOut, $stErr -Force -ErrorAction SilentlyContinue
$env:PYTHONIOENCODING = "utf-8"      # 否则冻结版把中文按 GBK 写进日志，读出来是乱码
$env:PYTHONUTF8 = "1"
$st = Start-Process $exe -ArgumentList "--self-test" -PassThru -Wait `
      -RedirectStandardOutput $stOut -RedirectStandardError $stErr
$lines = @(Get-Content $stOut -Encoding UTF8 -ErrorAction SilentlyContinue)
$passed = @($lines | Where-Object { $_ -match '^\[PASS\]' }).Count
$failed = @($lines | Where-Object { $_ -match '^\[FAIL\]' })
Write-Host ("  冻结版自检：{0} 项通过，{1} 项失败（exit={2}）" -f $passed, $failed.Count, $st.ExitCode) `
    -ForegroundColor $(if ($st.ExitCode -eq 0) { 'DarkGreen' } else { 'Red' })
foreach ($l in $failed) { Write-Host ("    " + $l) -ForegroundColor Red }
if ($st.ExitCode -ne 0) {
    $bad++
    Get-Content $stErr -Encoding UTF8 -ErrorAction SilentlyContinue |
        Select-Object -First 15 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkRed }
}

# 文件清单与自检都过了（带空格路径的专项检查在下一步）。
Write-Host "目录包文件校验通过。" -ForegroundColor Green

# ------------------------------------------- 3/4 带空格路径的自检（必检）
# 用户可能装在 `C:\Program Files\SHKit`、`%LOCALAPPDATA%\Programs\...`，或者用户名
# 里本身带空格。冻结版从这种路径启动时，任何"把路径拼进命令字符串""没引号的相对
# 路径""spawn 重新拉起自己时丢了引号"的写法都会直接崩 —— 所以这里把整个目录包
# **复制到一个带空格（且带中文）的路径**下，再从那里跑一遍它自己的自检。
Write-Host "`n=== 3/4  带空格路径下的自检（安装路径很可能有空格）===" -ForegroundColor Cyan
$SpaceDir = Join-Path $BuildRoot "dist space 带空格"
$SpaceDist = Join-Path $SpaceDir "SHKit"
Remove-Item $SpaceDir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $SpaceDir | Out-Null
Copy-Item $Dist $SpaceDist -Recurse -Force
Write-Host "复制到（含空格）：$SpaceDist"
$spOut = Join-Path $BuildRoot "selftest_space.log"
$spErr = Join-Path $BuildRoot "selftest_space.err"
Remove-Item $spOut, $spErr -Force -ErrorAction SilentlyContinue
$sp = Start-Process (Join-Path $SpaceDist "SHKit.exe") -ArgumentList "--self-test" `
      -PassThru -Wait -RedirectStandardOutput $spOut -RedirectStandardError $spErr
$spLines = @(Get-Content $spOut -Encoding UTF8 -ErrorAction SilentlyContinue)
$spPass = @($spLines | Where-Object { $_ -match '^\[PASS\]' }).Count
$spFail = @($spLines | Where-Object { $_ -match '^\[FAIL\]' })
Write-Host ("  带空格路径自检：{0} 项通过，{1} 项失败（exit={2}）" -f $spPass, $spFail.Count, $sp.ExitCode) `
    -ForegroundColor $(if ($sp.ExitCode -eq 0) { 'DarkGreen' } else { 'Red' })
foreach ($l in $spFail) { Write-Host ("    " + $l) -ForegroundColor Red }
if ($sp.ExitCode -ne 0 -or $spPass -lt 8) {
    $bad++
    Get-Content $spOut, $spErr -Encoding UTF8 -ErrorAction SilentlyContinue |
        Select-Object -Last 20 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkRed }
}
# 自检里那两步专门证明"带空格路径"这件事真的被走过了
foreach ($needle in @("带空格路径：网格 nc 写→读逐值一致",
                      "带空格路径：逐历元系数缓存写→读命中",
                      "独立子进程能起来并返回结果")) {
    if (@($spLines | Where-Object { $_ -match '^\[PASS\]' -and $_.Contains($needle) }).Count -eq 1) {
        Write-Host ("  OK   {0}" -f $needle) -ForegroundColor DarkGreen
    } else {
        Write-Host ("  没走通：{0}" -f $needle) -ForegroundColor Red; $bad++
    }
}
Remove-Item $SpaceDir -Recurse -Force -ErrorAction SilentlyContinue
if ($bad) { throw "带空格路径校验未通过（$bad 项）" }

if ($SkipInstaller) { Write-Host "`n（-SkipInstaller：不编译安装程序）"; exit 0 }

# ------------------------------------------------------------- 4/4 安装程序
Write-Host "`n=== 4/4  Inno Setup ===" -ForegroundColor Cyan
$Iscc = Find-Iscc
if (-not $Iscc) {
    Write-Host "未找到 Inno Setup（ISCC.exe）。请从 https://jrsoftware.org/isdl.php 安装后重跑：" -ForegroundColor Yellow
    Write-Host "  powershell -ExecutionPolicy Bypass -File packaging\build_installer.ps1 -SkipBuild" -ForegroundColor Yellow
    exit 2
}
if (-not (Test-Path $Icon)) {
    Copy-Item (Join-Path $Project "..\grace_icon.ico") $Icon -Force -ErrorAction SilentlyContinue
}
Write-Host "ISCC：$Iscc"
& $Iscc "/DProjDir=$Project" "/DSourceDir=$Dist" "/DOutputDir=$DistRoot" "/DIconFile=$Icon" `
        (Join-Path $Project "installer_shkit.iss")
if ($LASTEXITCODE -ne 0) { throw "Inno Setup 编译失败（exit $LASTEXITCODE）" }

$Setup = Join-Path $DistRoot "SHKit_Setup_v2.0.1.exe"
if (-not (Test-Path $Setup)) { throw "安装程序没有生成" }
Write-Host ("`n安装程序：{0}  ({1:N0} MB)" -f $Setup, ((Get-Item $Setup).Length/1MB)) -ForegroundColor Green

# ------------------------------------------- 4. 装一遍再验（可选，最硬的验证）
if ($VerifyInstall) {
    Write-Host "`n=== 额外：静默安装到**带空格**的临时目录并校验 ===" -ForegroundColor Cyan
    # ★ 目录名里刻意留一个空格：安装路径带空格是**常态**（Program Files、用户名带
    #   空格…），而 Inno 的 /DIR 参数一旦没整体加引号，就会被拆成两个参数，
    #   于是"装到了错误的目录"或直接失败。这里同时验证引号与运行。
    $TestDir = Join-Path $BuildRoot "_install test 带空格"
    Remove-Item $TestDir -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host ("安装到：{0}" -f $TestDir)
    $ip = Start-Process $Setup -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART",
        "/NOICONS", "/DIR=`"$TestDir`"" -PassThru -Wait
    if ($ip.ExitCode -ne 0) { throw "静默安装失败（exit=$($ip.ExitCode)）" }
    if (-not (Test-Path (Join-Path $TestDir "SHKit.exe"))) {
        throw "安装路径不对：$TestDir 下没有 SHKit.exe（/DIR 带空格时被拆开了？）"
    }

    $bad2 = 0
    foreach ($rel in @(
        "SHKit.exe",
        "_internal\docs\使用说明.html",
        "_internal\docs\使用说明_img\panel_params.png",
        "_internal\licenses\LGPL-3.0.txt",
        "source\shkit\gui\main_window.py",
        "source\shkit\io.py",
        "source\tests\test_gui_smoke.py",
        "source\tools\check_licensing.py",
        "source\pyproject.toml")) {
        if (Test-Path (Join-Path $TestDir $rel)) {
            Write-Host ("  已安装 OK   {0}" -f $rel) -ForegroundColor DarkGreen
        } else {
            Write-Host ("  已安装 缺失 {0}" -f $rel) -ForegroundColor Red; $bad2++
        }
    }
    $ip2 = Start-Process (Join-Path $TestDir "SHKit.exe") -PassThru
    Start-Sleep -Seconds 10
    if ($ip2.HasExited) { Write-Host "  装完的 exe 启动即退出" -ForegroundColor Red; $bad2++ }
    else { Write-Host "  OK   装完的 exe 正常启动" -ForegroundColor DarkGreen; Stop-Process -Id $ip2.Id -Force }

    # 装完的版本也跑一遍自检
    $io2 = Join-Path $BuildRoot "selftest_installed.log"
    $ie2 = Join-Path $BuildRoot "selftest_installed.err"
    $ist = Start-Process (Join-Path $TestDir "SHKit.exe") -ArgumentList "--self-test" `
           -PassThru -Wait -RedirectStandardOutput $io2 -RedirectStandardError $ie2
    $iLines = @(Get-Content $io2 -Encoding UTF8 -ErrorAction SilentlyContinue)
    $ok2 = @($iLines | Where-Object { $_ -match '^\[PASS\]' }).Count
    Write-Host ("  装完的版本自检：{0} 项通过（exit={1}）" -f $ok2, $ist.ExitCode) `
        -ForegroundColor $(if ($ist.ExitCode -eq 0) { 'DarkGreen' } else { 'Red' })
    # 装到带空格的路径下，这三步必须真的走过（它们是本次专项回归的锚点）
    foreach ($needle in @("带空格路径：网格 nc 写→读逐值一致",
                          "带空格路径：逐历元系数缓存写→读命中",
                          "独立子进程能起来并返回结果")) {
        if (@($iLines | Where-Object { $_ -match '^\[PASS\]' -and $_.Contains($needle) }).Count -eq 1) {
            Write-Host ("  OK   装到带空格路径后：{0}" -f $needle) -ForegroundColor DarkGreen
        } else {
            Write-Host ("  装到带空格路径后没走通：{0}" -f $needle) -ForegroundColor Red
            $bad2++
        }
    }
    if ($ist.ExitCode -ne 0) {
        $bad2++
        Get-Content $io2, $ie2 -Encoding UTF8 -ErrorAction SilentlyContinue |
            Select-Object -First 15 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkRed }
    }

    $un = Join-Path $TestDir "unins000.exe"
    if (Test-Path $un) { Start-Process $un -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES" -Wait | Out-Null }
    Remove-Item $TestDir -Recurse -Force -ErrorAction SilentlyContinue
    if ($bad2) { throw "安装结果校验未通过（$bad2 项）" }
    Write-Host "安装结果校验通过（已卸载并清理）。" -ForegroundColor Green
}

Write-Host ("`n全部完成：{0}" -f $Setup) -ForegroundColor Green
