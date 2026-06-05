Ich brauche einen professionellen QUANT Baseline Code.

Projektstruktur:

QUANT/
├── Dashboard/
└── Data_Center/
    ├── Backend_Management/
    │   └── 2_Baseline/
    │       └── <DOMAIN>/
    │           └── <BASELINE_NAME>/
    │               └── code.py
    │
    └── Data/
        ├── 1_Pipeline/
        ├── 2_Baseline/
        ├── 3_Features/
        ├── 4_Analytics/
        └── 5_Reports/

Ziel:

<BESCHREIBUNG>

Input:

<Data_Center/Data/1_Pipeline/...>

Output:

<Data_Center/Data/2_Baseline/...>

Anforderungen:

1. Produktionsfähiger Code
2. Keine Pseudocode-Blöcke
3. Keine Platzhalter im finalen Code
4. Fehlerbehandlung
5. Logging
6. Schema Validation
7. Data Validation
8. Automatische Ordnererstellung
9. Inkrementelle Verarbeitung wenn möglich
10. Vollständige Registry Integration

CODE_REGISTRY HEADER:

# ============================================================
# CODE_REGISTRY
# ============================================================
# script_id: <SCRIPT_ID>
# script_name: <SCRIPT_NAME>
# owner: Leon Everts
# status: active
# layer: 2_Baseline
# domain: <DOMAIN>
# asset_type: Baseline
# purpose: <PURPOSE>
# inputs:
#   - <INPUT_1>
# outputs:
#   - <OUTPUT_1>
# upstream_data:
#   - 1_Pipeline
# downstream_data:
#   - 3_Features
# dependencies:
#   - pandas
#   - pathlib
# schedule: manual
# version: v1.0.0
# last_reviewed: 2026-06-01
# business_criticality: high
# environment: desktop
# registry_group: baseline
# author: Leon Everts
# reviewer: ChatGPT
# created_date: 2026-06-01
# tags:
#   - baseline
#   - normalization
#   - data_cleaning
# notes:
#   - Generates clean standardized baseline datasets.
# ============================================================

Runtime Registry:

CODE_REGISTRY = {
    "script_id": "<SCRIPT_ID>",
    "script_name": "<SCRIPT_NAME>",
    "owner": "Leon Everts",
    "status": "active",
    "layer": "2_Baseline",
    "domain": "<DOMAIN>",
    "asset_type": "Baseline",
    "purpose": "<PURPOSE>",
    "inputs": ["<INPUT_1>"],
    "outputs": ["<OUTPUT_1>"],
    "upstream_data": ["1_Pipeline"],
    "downstream_data": ["3_Features"],
    "dependencies": ["pandas", "pathlib"],
    "schedule": "manual",
    "version": "v1.0.0",
    "last_reviewed": "2026-06-01",
    "business_criticality": "high",
    "environment": "desktop",
    "registry_group": "baseline",
    "author": "Leon Everts",
    "reviewer": "ChatGPT",
    "created_date": "2026-06-01",
    "tags": [
        "baseline",
        "normalization",
        "data_cleaning"
    ],
}

Pflichtfunktionen:

- find_quant_root()
- validate_input_schema()
- validate_output_schema()
- discover_input_files()
- process_file()
- save_output()
- main()

Root Detection:

QUANT Root enthält:

Dashboard/
Data_Center/

Verarbeitung:

1_Pipeline
    ↓
Validation
    ↓
Normalization
    ↓
Cleaning
    ↓
IS/OOS Split
    ↓
2_Baseline

Zusätzlich:

- Summary Report erzeugen
- Fehlerprotokoll erzeugen
- Metadata-Datei erzeugen
- Registry kompatibel zu code_registry.db

Bitte vollständigen lauffähigen code.py erzeugen.
Keine Ausschnitte.
Keine Erklärungen außerhalb des Codes außer einem kurzen Hinweis.