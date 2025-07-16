# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['video_to_image.py'],
    pathex=['D:\\ProgramData\\anaconda3\\envs\\auto_ppt\\Lib\\site-packages', 'D:\\GitHub\\image-to-ppt\\tools'],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [('D:\\ProgramData\\anaconda3\\envs\\auto_ppt\\python.exe', None, 'OPTION')],
    name='video_to_ppt',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['D:\\GitHub\\image-to-ppt\\data\\icon.ico'],
)
