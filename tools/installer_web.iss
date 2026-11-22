; GGS small installer (Inno Setup 6): the GGS source and data plus uv, a few tens of MB. During setup it installs
; Python 3.13, the libraries (CUDA torch with an NVIDIA GPU) and the motion library, and optionally the AI models,
; into the app folder (tools\setup_env.py). Stage it first:   python tools\stage_app.py --windows
; then:   ISCC tools\installer_web.iss   ->  dist\GGS-Setup-<version>.exe
; Per-user install, no admin rights. Models go to %USERPROFILE%\.ggs\models (kept when uninstalling).

#define AppName "GGS - Game Graphic Studio"
#define AppVersion Trim(FileRead(FileOpen(SourcePath + "\..\dist\stage\GGS\VERSION")))

[Setup]
; same AppId as the full installer: installing this one upgrades it
AppId={{6B0E8F43-2C1D-4E55-9A7B-5D3C1F0A9E21}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=GGS
AppPublisherURL=https://github.com/shahind/ggs
DefaultDirName={localappdata}\Programs\GGS
DefaultGroupName=GGS
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=GGS-Setup-{#AppVersion}
SetupIconFile=..\dist\stage\GGS\GGS.ico
UninstallDisplayIcon={app}\GGS.ico
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2/max
SolidCompression=yes
ChangesEnvironment=yes

[Tasks]
Name: "models"; Description: "Download the AI models now (about 11 GB with an NVIDIA GPU; the app can also do it later)"; GroupDescription: "Downloads:"
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "addtopath"; Description: "Add the 'ggs' command line tool to PATH"; GroupDescription: "Command line:"; Flags: unchecked

[InstallDelete]
; the full installer's portable Python (this installer makes env\ instead)
Type: filesandordirs; Name: "{app}\python\Lib"
Type: filesandordirs; Name: "{app}\python\DLLs"
Type: files; Name: "{app}\python\*.exe"
Type: files; Name: "{app}\python\*.dll"
Type: files; Name: "{app}\python\*._pth"
Type: filesandordirs; Name: "{app}\src"

[Files]
Source: "..\dist\stage\GGS\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\GGS"; Filename: "{app}\Start GGS.cmd"; WorkingDir: "{app}"; IconFilename: "{app}\GGS.ico"; Comment: "Game Graphic Studio"; Flags: runminimized
Name: "{autoprograms}\GGS Repair"; Filename: "{app}\Repair GGS.cmd"; WorkingDir: "{app}"; IconFilename: "{app}\GGS.ico"; Comment: "Reinstall GGS's Python and libraries"
Name: "{autodesktop}\GGS"; Filename: "{app}\Start GGS.cmd"; WorkingDir: "{app}"; IconFilename: "{app}\GGS.ico"; Tasks: desktopicon; Flags: runminimized

[Registry]
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; Tasks: addtopath; Check: NeedsAddPath(ExpandConstant('{app}'))

[Run]
Filename: "{app}\Start GGS.cmd"; WorkingDir: "{app}"; Description: "Start GGS"; Flags: nowait postinstall skipifsilent runminimized

[UninstallDelete]
; everything setup downloaded into the app folder (Python, libraries, motion library); models in ~\.ggs stay
Type: filesandordirs; Name: "{app}"

[Code]
var
  SetupFailed: Boolean;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if (CurPageID = wpSelectDir) and (Length(WizardDirValue) > 110) then
  begin
    MsgBox('Please choose a shorter folder path (at most 110 characters).', mbError, MB_OK);
    Result := False;
  end;
end;

// Python, the libraries and the motion library come from the internet now, in a window that shows progress.
procedure CurStepChanged(CurStep: TSetupStep);
var
  Params: string;
  ResultCode: Integer;
begin
  if CurStep <> ssPostInstall then
    exit;
  Params := '';
  if WizardIsTaskSelected('models') then
    Params := '--models required';
  WizardForm.StatusLabel.Caption := 'Downloading Python, the libraries and the motion library (a few GB)...';
  if not Exec(ExpandConstant('{app}\Repair GGS.cmd'), Params, ExpandConstant('{app}'), SW_SHOW,
              ewWaitUntilTerminated, ResultCode) or (ResultCode <> 0) then
  begin
    SetupFailed := True;
    MsgBox('GGS was installed, but downloading its Python and libraries did not finish.' + #13#10 +
           'Check the internet connection, then run "GGS Repair" from the Start menu.', mbError, MB_OK);
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
