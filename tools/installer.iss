; GGS installer (Inno Setup 6). Build the app folder first:  .venv\Scripts\python tools\build_app.py
; then:  ISCC tools\installer.iss   ->  dist\GGS-Setup-<version>.exe (+ .bin slices: the app is ~5 GB with CUDA)
; Per-user install, no admin rights. AI models are not included; the app downloads them on first run into
; %USERPROFILE%\.ggs\models, which uninstalling leaves in place.

#define AppName "GGS - Game Graphic Studio"
#define AppVersion Trim(FileRead(FileOpen(SourcePath + "\..\dist\GGS\VERSION")))

[Setup]
AppId={{6B0E8F43-2C1D-4E55-9A7B-5D3C1F0A9E21}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=GGS
DefaultDirName={localappdata}\Programs\GGS
DefaultGroupName=GGS
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=GGS-Setup-{#AppVersion}
SetupIconFile=..\dist\GGS\GGS.ico
UninstallDisplayIcon={app}\GGS.ico
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; the CUDA libraries make the app larger than one setup file may be
DiskSpanning=yes
DiskSliceSize=max
Compression=lzma2/fast
SolidCompression=no
LZMAUseSeparateProcess=yes
ChangesEnvironment=yes

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "addtopath"; Description: "Add the 'ggs' command line tool to PATH"; GroupDescription: "Command line:"; Flags: unchecked

[Files]
Source: "..\dist\GGS\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\GGS"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m ggs.ui.app"; WorkingDir: "{app}"; IconFilename: "{app}\GGS.ico"; Comment: "Game Graphic Studio"
Name: "{autodesktop}\GGS"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m ggs.ui.app"; WorkingDir: "{app}"; IconFilename: "{app}\GGS.ico"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; Tasks: addtopath; Check: NeedsAddPath(ExpandConstant('{app}'))

[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: "-m ggs.ui.app"; WorkingDir: "{app}"; Description: "Start GGS"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; byte-code caches written while the app ran
Type: filesandordirs; Name: "{app}\python"

[Code]
// Windows paths stop at 260 characters and the deepest bundled file is ~140 characters below the app folder.
function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if (CurPageID = wpSelectDir) and (Length(WizardDirValue) > 110) then
  begin
    MsgBox('Please choose a shorter folder path (at most 110 characters).', mbError, MB_OK);
    Result := False;
  end;
end;


function NeedsAddPath(Dir: string): Boolean;
var
  Paths: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', Paths) then
  begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(Paths) + ';') = 0;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Paths, Dir: string;
  P: Integer;
begin
  if CurUninstallStep <> usPostUninstall then
    exit;
  if not RegQueryStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', Paths) then
    exit;
  Dir := ';' + ExpandConstant('{app}');
  P := Pos(Uppercase(Dir), Uppercase(Paths));
  if P > 0 then
  begin
    Delete(Paths, P, Length(Dir));
    RegWriteExpandStringValue(HKEY_CURRENT_USER, 'Environment', 'Path', Paths);
  end;
end;
