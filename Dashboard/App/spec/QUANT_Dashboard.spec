# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['C:\\Users\\Leon\\Desktop\\QUANT\\Dashboard\\Main.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Users\\Leon\\Desktop\\QUANT\\Dashboard\\Main.py', 'Dashboard/Main.py'), ('C:\\Users\\Leon\\Desktop\\QUANT\\Dashboard\\Building_Blocks', 'Dashboard/Building_Blocks'), ('C:\\Users\\Leon\\Desktop\\QUANT\\Data_Center', 'Data_Center'), ('C:\\Users\\Leon\\Desktop\\QUANT\\System_Info', 'System_Info')],
    hiddenimports=['tkinter', 'tkinter.ttk', 'tkinter.messagebox', 'tkinter.filedialog', 'sqlite3', 'json', 'pathlib', 'importlib', 'importlib.util', 'pandas', 'numpy', 'matplotlib', 'matplotlib.backends.backend_tkagg', 'matplotlib.figure'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'tensorflow', 'keras', 'jax', 'jaxlib', 'transformers', 'sklearn', 'scipy', 'IPython', 'notebook', 'jupyter'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='QUANT_Dashboard',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='QUANT_Dashboard',
)
