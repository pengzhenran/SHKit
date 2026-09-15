; =============================================================================
;  Inno Setup script — SHKit v2.0.1
; =============================================================================
;  用法：
;    1. 先出 PyInstaller 目录包（在 SHKit/ 下执行）：
;         pyinstaller --clean --noconfirm packaging\shkit.spec
;       → dist\SHKit\SHKit.exe + dist\SHKit\_internal\
;    2. 再编译本脚本：
;         ISCC.exe installer_shkit.iss
;       → dist\SHKit_Setup_v2.0.1.exe
;    或者直接跑 packaging\build_installer.ps1（两步一起做）。
;
;  发行包内容（见 packaging/shkit.spec 的 datas）：
;    SHKit.exe + _internal\   运行所需的全部组件（Python 运行时、Qt、绘图库…）
;    _internal\data\          勒夫数表 + 离线海岸线
;    _internal\docs\          使用说明.html + 使用说明_img\ + 公众号二维码 + 纯文本版
;    _internal\licenses\      LGPLv3 等许可全文
;    source\                  软件运行代码（shkit 包）+ 测试文件 + 工具脚本
; =============================================================================

#define MyAppName       "SHKit"
;  补丁号也带上：安装程序名、Windows「应用和功能」里显示的版本都跟 SHKIT_VERSION 一致，
;  用户一眼能看出装的是 2.0 还是 2.0.1（validate_v2 守着"两处必须相同"）。
#define MyAppVersion    "2.0.1"
#define MyAppPublisher  "彭桢燃  Zhenran Peng  (China University of Geosciences, Wuhan)"
#define MyAppExeName    "SHKit.exe"

; ---- 路径 -------------------------------------------------------------------
; 三个都可以在命令行上覆盖，方便把产物放到项目外（避开网盘同步目录）：
;   ISCC.exe /DProjDir="D:\...\SHKit" /DSourceDir="D:\SHKit_build\dist\SHKit" ^
;            /DOutputDir="D:\SHKit_build\dist" /DIconFile="D:\SHKit_build\SHKit.ico" ^
;            installer_shkit.iss
#ifndef ProjDir
  #define ProjDir       "."
#endif
#ifndef SourceDir
  #define SourceDir     ProjDir + "\dist\SHKit"
#endif
#ifndef OutputDir
  #define OutputDir     ProjDir + "\dist"
#endif
#ifndef IconFile
  #define IconFile      ProjDir + "\..\grace_icon.ico"
#endif

[Setup]
; 每次发布换一个 GUID：Inno Setup 菜单 Tools -> Generate GUID
AppId={{7C41E9A5-3D62-4F18-9B07-2A5E6C1D84F3}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
; 默认按用户安装（不弹管理员）：装到 %LOCALAPPDATA%\Programs\SHKit
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=SHKit_Setup_v{#MyAppVersion}
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes
; 安装包里已有中英文语言文件，界面按系统语言走
ShowLanguageDialog=auto

[Languages]
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式 / Create a &desktop shortcut"; GroupDescription: "附加任务 / Additional shortcuts:"; Flags: checkedonce

[Files]
; --- 程序本体（PyInstaller onedir 输出） -----------------------------------
Source: "{#SourceDir}\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs

; --- 软件运行代码与测试文件（随包附源码，便于用户核对与复现） ----------------
Source: "{#ProjDir}\shkit\*";    DestDir: "{app}\source\shkit";    Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "__pycache__\*,*.pyc"
Source: "{#ProjDir}\tools\*";    DestDir: "{app}\source\tools";    Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "__pycache__\*,*.pyc"
Source: "{#ProjDir}\tests\*";    DestDir: "{app}\source\tests";    Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "__pycache__\*,*.pyc,_gui_shots\*,_tmp_cli\*,_tmp_roundtrip\*,_tmp_sandbox\*"
Source: "{#ProjDir}\examples\*"; DestDir: "{app}\source\examples"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "__pycache__\*,*.pyc"
Source: "{#ProjDir}\pyproject.toml"; DestDir: "{app}\source"; Flags: ignoreversion
Source: "{#ProjDir}\requirements.txt"; DestDir: "{app}\source"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\使用说明 (HTML)"; Filename: "{app}\_internal\docs\使用说明.html"
Name: "{group}\卸载 {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent
Filename: "{app}\_internal\docs\使用说明.html"; Description: "打开使用说明"; Flags: shellexec nowait postinstall skipifsilent unchecked

