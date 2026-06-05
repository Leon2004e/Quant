# -*- coding: utf-8 -*-
# ============================================================
# CODE_REGISTRY
# ============================================================
# script_id: build_quant_dashboard_app
# script_name: QUANT Dashboard App Builder
# owner: Leon
# status: active
# layer: Dashboard
# domain: Deployment
# asset_type: Builder
# purpose: Build QUANT Dashboard into a Windows Desktop App and store every build artifact only inside Dashboard/App
# inputs:
#   - Dashboard/Main.py
#   - Dashboard/Building_Blocks
#   - Data_Center
#   - System_Info
#   - tools
# outputs:
#   - Dashboard/App/dist/QUANT_Dashboard/QUANT_Dashboard.exe
#   - Dashboard/App/build/
#   - Dashboard/App/spec/
# dependencies:
#   - pyinstaller
# schedule: manual
# version: v2.1.0_dashboard_app_folder_only
# last_reviewed: 2026-06-04
# ============================================================

from __future__ import annotations

import os
import sys
import shutil
import platform
import subprocess
from pathlib import Path
from typing import List, Optional


# Kein __pycache__ neben Builder_App.py erzeugen.
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "QUANT_Dashboard"

# Wenn True: die App öffnet ohne Terminal-Fenster.
# Wenn die App nicht startet und du Fehler sehen willst: auf False setzen.
WINDOWED_MODE = True

# Wenn True: löscht nur alte Build-Dateien in Dashboard/App.
CLEAN_OLD_BUILDS = True

# Wenn True: löscht KEINE alten root build/dist Ordner.
# Wichtig: Der Builder schreibt ab jetzt nichts mehr in QUANT/build oder QUANT/dist.
# Alte Ordner kannst du manuell löschen, wenn sie noch von früher existieren.
CLEAN_LEGACY_ROOT_BUILDS = False

# Wenn True: Torch/TensorFlow/Keras werden ausgeschlossen.
# Das macht den Build schneller und kleiner.
EXCLUDE_HEAVY_AI_LIBS = True

# Wenn True: öffnet nach dem Build den Dashboard/App/dist/QUANT_Dashboard Ordner.
OPEN_DIST_FOLDER_AFTER_BUILD = True

# Wenn True: startet nach dem Build direkt die EXE.
RUN_APP_AFTER_BUILD = False

# Optional: icon.ico in QUANT/Dashboard/icon.ico ablegen.
ICON_FILE_NAME = "icon.ico"


# ============================================================
# ROOT DETECTION
# ============================================================

def find_quant_root() -> Path:
    """
    Sucht den QUANT Root.

    Erwartet:
    QUANT/
    ├── Dashboard/
    └── Data_Center/
    """
    current = Path(__file__).resolve().parent

    for p in [current] + list(current.parents):
        if (p / "Dashboard").is_dir() and (p / "Data_Center").is_dir():
            return p.resolve()

    raise RuntimeError(
        "QUANT root not found. Expected Dashboard/ and Data_Center/."
    )


# ============================================================
# PATHS
# ============================================================

ROOT = find_quant_root()

DASHBOARD_DIR = ROOT / "Dashboard"
MAIN_FILE = DASHBOARD_DIR / "Main.py"

# ALLE Build-Artefakte gehen ausschließlich hier rein:
APP_WORKSPACE_DIR = DASHBOARD_DIR / "App"

BUILD_DIR = APP_WORKSPACE_DIR / "build"
DIST_DIR = APP_WORKSPACE_DIR / "dist"
SPEC_DIR = APP_WORKSPACE_DIR / "spec"
LOG_DIR = APP_WORKSPACE_DIR / "logs"

SPEC_FILE = SPEC_DIR / f"{APP_NAME}.spec"

FINAL_APP_DIR = DIST_DIR / APP_NAME
FINAL_EXE = FINAL_APP_DIR / f"{APP_NAME}.exe"

ICON_FILE = DASHBOARD_DIR / ICON_FILE_NAME

# Alte Legacy-Pfade nur zur optionalen Diagnose.
LEGACY_ROOT_BUILD_DIR = ROOT / "build"
LEGACY_ROOT_DIST_DIR = ROOT / "dist"
LEGACY_ROOT_SPEC_FILE = ROOT / f"{APP_NAME}.spec"


# ============================================================
# TERMINAL HELPERS
# ============================================================

def line() -> None:
    print("=" * 80)


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def fail(msg: str) -> None:
    print(f"[ERROR] {msg}")


def pause() -> None:
    if os.name == "nt":
        try:
            input("\nPress ENTER to close...")
        except Exception:
            pass


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ============================================================
# VALIDATION
# ============================================================

def validate_project() -> None:
    line()
    info("VALIDATING PROJECT STRUCTURE")
    line()

    info(f"ROOT              = {ROOT}")
    info(f"DASHBOARD_DIR     = {DASHBOARD_DIR}")
    info(f"MAIN_FILE         = {MAIN_FILE}")
    info(f"APP_WORKSPACE_DIR = {APP_WORKSPACE_DIR}")
    info(f"BUILD_DIR         = {BUILD_DIR}")
    info(f"DIST_DIR          = {DIST_DIR}")
    info(f"SPEC_DIR          = {SPEC_DIR}")

    if not MAIN_FILE.exists():
        raise FileNotFoundError(f"Main.py not found:\n{MAIN_FILE}")

    if not (DASHBOARD_DIR / "Building_Blocks").exists():
        warn("Dashboard/Building_Blocks not found.")

    if not (ROOT / "Data_Center").exists():
        raise FileNotFoundError(f"Data_Center not found:\n{ROOT / 'Data_Center'}")

    ensure_dir(APP_WORKSPACE_DIR)
    ensure_dir(BUILD_DIR)
    ensure_dir(DIST_DIR)
    ensure_dir(SPEC_DIR)
    ensure_dir(LOG_DIR)

    ok("Project structure valid.")


def ensure_pyinstaller() -> None:
    """
    Kein direkter import PyInstaller.__main__.
    Dadurch keine gelbe Editor-Warnung.
    """
    info("Checking PyInstaller...")

    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--version"],
        text=True,
        capture_output=True,
    )

    if result.returncode == 0:
        ok(f"PyInstaller installed: {result.stdout.strip()}")
        return

    warn("PyInstaller not found. Installing...")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pyinstaller",
    ])

    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--version"],
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        raise RuntimeError("PyInstaller installation failed.")

    ok(f"PyInstaller installed: {result.stdout.strip()}")


# ============================================================
# CLEAN
# ============================================================

def remove_path(path: Path) -> None:
    if not path.exists():
        return

    if path.is_dir():
        info(f"Removing folder: {path}")
        shutil.rmtree(path, ignore_errors=True)
    else:
        info(f"Removing file: {path}")
        try:
            path.unlink()
        except Exception:
            pass


def clean_old_builds() -> None:
    if not CLEAN_OLD_BUILDS:
        return

    line()
    info("CLEANING OLD BUILDS IN DASHBOARD/APP ONLY")
    line()

    remove_path(BUILD_DIR)
    remove_path(DIST_DIR)
    remove_path(SPEC_DIR)

    ensure_dir(APP_WORKSPACE_DIR)
    ensure_dir(BUILD_DIR)
    ensure_dir(DIST_DIR)
    ensure_dir(SPEC_DIR)
    ensure_dir(LOG_DIR)

    if CLEAN_LEGACY_ROOT_BUILDS:
        warn("CLEAN_LEGACY_ROOT_BUILDS=True: removing old root build/dist/spec.")
        remove_path(LEGACY_ROOT_BUILD_DIR)
        remove_path(LEGACY_ROOT_DIST_DIR)
        remove_path(LEGACY_ROOT_SPEC_FILE)
    else:
        if LEGACY_ROOT_BUILD_DIR.exists() or LEGACY_ROOT_DIST_DIR.exists() or LEGACY_ROOT_SPEC_FILE.exists():
            warn("Old legacy root build/dist/spec still exists from earlier builds.")
            warn("This builder will NOT use them. Delete manually if you want a clean project root.")

    ok("Old Dashboard/App build files removed.")


# ============================================================
# PYINSTALLER ARG HELPERS
# ============================================================

def add_data_arg(source: Path, target: str) -> Optional[str]:
    """
    PyInstaller add-data syntax:
    Windows: source;target
    Linux/Mac: source:target
    """
    if not source.exists():
        warn(f"Skipped missing data path: {source}")
        return None

    sep = ";" if os.name == "nt" else ":"
    return f"{source}{sep}{target}"


def build_command() -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(MAIN_FILE),

        f"--name={APP_NAME}",
        "--onedir",
        "--clean",
        "--noconfirm",

        # Wichtig:
        # Nichts landet mehr in QUANT/build, QUANT/dist oder QUANT/*.spec
        f"--distpath={DIST_DIR}",
        f"--workpath={BUILD_DIR}",
        f"--specpath={SPEC_DIR}",
    ]

    if WINDOWED_MODE:
        cmd.append("--noconsole")
    else:
        cmd.append("--console")

    if ICON_FILE.exists():
        cmd.append(f"--icon={ICON_FILE}")
        info(f"Using icon: {ICON_FILE}")
    else:
        warn("No icon.ico found. Using default icon.")

    # Wichtig:
    # Nicht das ganze Dashboard/ als add-data nehmen,
    # weil sonst Dashboard/App rekursiv in die App gepackt werden kann.
    # Deshalb nur Main.py + Building_Blocks separat.
    data_items = [
        (DASHBOARD_DIR / "Main.py", "Dashboard/Main.py"),
        (DASHBOARD_DIR / "Building_Blocks", "Dashboard/Building_Blocks"),
        (ROOT / "Data_Center", "Data_Center"),
        (ROOT / "System_Info", "System_Info"),
        (ROOT / "tools", "tools"),
    ]

    for source, target in data_items:
        arg = add_data_arg(source, target)
        if arg is not None:
            cmd.append(f"--add-data={arg}")
            info(f"Included: {source} -> {target}")

    hidden_imports = [
        "tkinter",
        "tkinter.ttk",
        "tkinter.messagebox",
        "tkinter.filedialog",
        "sqlite3",
        "json",
        "pathlib",
        "importlib",
        "importlib.util",
        "pandas",
        "numpy",
        "matplotlib",
        "matplotlib.backends.backend_tkagg",
        "matplotlib.figure",
    ]

    for module in hidden_imports:
        cmd.append(f"--hidden-import={module}")

    if EXCLUDE_HEAVY_AI_LIBS:
        excluded = [
            "torch",
            "tensorflow",
            "keras",
            "jax",
            "jaxlib",
            "transformers",
            "sklearn",
            "scipy",
            "IPython",
            "notebook",
            "jupyter",
        ]
        for module in excluded:
            cmd.append(f"--exclude-module={module}")

    return cmd


# ============================================================
# BUILD
# ============================================================

def run_build() -> None:
    cmd = build_command()

    line()
    info("BUILDING QUANT DASHBOARD INTO DASHBOARD/APP")
    line()

    print("Command:")
    print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
    print()

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    log_file = LOG_DIR / "last_build_command.txt"
    log_file.write_text(
        " ".join(f'"{x}"' if " " in x else x for x in cmd),
        encoding="utf-8",
    )

    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        env=env,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"PyInstaller build failed with exit code {result.returncode}."
        )

    if not FINAL_EXE.exists():
        raise FileNotFoundError(
            f"Build finished but EXE was not found:\n{FINAL_EXE}"
        )

    ok("Build finished successfully.")


def open_dist_folder() -> None:
    if not OPEN_DIST_FOLDER_AFTER_BUILD:
        return

    if not FINAL_APP_DIR.exists():
        return

    try:
        if os.name == "nt":
            os.startfile(str(FINAL_APP_DIR))
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(FINAL_APP_DIR)])
        else:
            subprocess.Popen(["xdg-open", str(FINAL_APP_DIR)])
    except Exception as exc:
        warn(f"Could not open dist folder: {exc}")


def run_app() -> None:
    if not RUN_APP_AFTER_BUILD:
        return

    if not FINAL_EXE.exists():
        return

    try:
        subprocess.Popen([str(FINAL_EXE)], cwd=str(FINAL_APP_DIR))
    except Exception as exc:
        warn(f"Could not start app: {exc}")


# ============================================================
# FINAL OUTPUT
# ============================================================

def print_final_output() -> None:
    line()
    print("BUILD FINISHED")
    line()
    print()
    print("APP WORKSPACE:")
    print(APP_WORKSPACE_DIR)
    print()
    print("APP ORDNER:")
    print(FINAL_APP_DIR)
    print()
    print("START DATEI:")
    print(FINAL_EXE)
    print()
    print("WICHTIG:")
    print("- Alles vom Builder liegt unter Dashboard/App.")
    print("- Öffne die App mit Doppelklick auf QUANT_Dashboard.exe.")
    print("- Kopiere immer den gesamten Ordner Dashboard/App/dist/QUANT_Dashboard.")
    print("- Nicht nur die EXE kopieren.")
    print("- Main.py bleibt unverändert.")
    print("- Der Builder nutzt NICHT mehr QUANT/build oder QUANT/dist.")
    print()


# ============================================================
# MAIN
# ============================================================

def build() -> None:
    validate_project()
    ensure_pyinstaller()
    clean_old_builds()
    run_build()
    print_final_output()
    open_dist_folder()
    run_app()


if __name__ == "__main__":
    try:
        build()
    except Exception as exc:
        line()
        fail(str(exc))
        line()
        pause()
        raise SystemExit(1)
