[Setup]
AppName=YOLO Streamlit App
AppVersion=1.0
DefaultDirName={pf}\YOLOApp
DefaultGroupName=YOLOApp
OutputDir=.
OutputBaseFilename=YOLOAppInstaller
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64

[Files]
Source: "launch.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "ultralytics\*"; DestDir: "{app}\Ultralytics"; Flags: recursesubdirs
Source: "app.py"; DestDir: "{app}"
Source: "models.pt"; DestDir: "{app}"
Source: "chromedriver.exe"; DestDir: "{app}"
Source: "rpa.py"; DestDir: "{app}"
Source: "runtime.txt"; DestDir: "{app}"

[Icons]
Name: "{group}\YOLO Streamlit"; Filename: "{app}\launch.exe"

[Run]
Filename: "{app}\launch.exe"; Description: "Run YOLO Streamlit"; Flags: nowait postinstall skipifsilent
