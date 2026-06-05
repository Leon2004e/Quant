Ich brauche einen kompletten professionellen Python-Code für eine neue QUANT Pipeline.

Ziel-Datei:

QUANT/Data_Center/Backend_Management/1_Pipelines/<DOMAIN>/<PIPELINE_NAME>/code.py

Projektstruktur:

QUANT/
├── Dashboard/
└── Data_Center/
    ├── Backend_Management/
    │   └── 1_Pipelines/
    │       └── <DOMAIN>/
    │           └── <PIPELINE_NAME>/
    │               └── code.py
    └── Data/
        └── 1_Pipeline/
            └── <DOMAIN>/

Pipeline Name:

<PIPELINE_NAME>

Domain:

<DOMAIN>

Funktion:

<BESCHREIBUNG WAS DIE PIPELINE MACHEN SOLL>

Input:

<INPUT_DATENQUELLE>

Output:

<OUTPUT_PFAD>

Schedule:

manual / realtime / daily / weekly

Wichtig:

Der Code muss produktionsfähig sein und zur QUANT-Struktur passen.

Pflicht:

1. CODE_REGISTRY Header ganz oben im Docstring.
2. Runtime CODE_REGISTRY Dictionary im Code.
3. find_quant_root(start: Path) Funktion.
4. Output-Pfade relativ zu QUANT/Data_Center/Data/1_Pipeline/.
5. Keine alten FTMO_ROOT oder Quant_Structure Pfade.
6. Fehlerbehandlung bei Datei-, DB-, API- und Datenproblemen.
7. Atomic Writes für CSV/JSON/Parquet, wenn Dateien geschrieben werden.
8. Logging/Prints mit [INFO], [OK], [WARN], [ERROR].
9. main() Funktion.
10. if __name__ == "__main__": main()
11. Keine hardcodierten sensiblen Zugangsdaten im Code.
12. Zugangsdaten nur über ENV-Variablen.
13. Bestehende Daten sollen inkrementell aktualisiert werden, wenn sinnvoll.
14. Schema-Migrationen einbauen, falls SQLite genutzt wird.
15. Summary/Metadata-Datei schreiben, wenn sinnvoll.

CODE_REGISTRY Header Format:

# ============================================================
# CODE_REGISTRY
# ============================================================
# script_id: <SCRIPT_ID>
# script_name: <SCRIPT_NAME>
# owner: Leon Everts
# status: active
# layer: 1_Pipeline
# domain: <DOMAIN>
# asset_type: Pipeline
# purpose: <PURPOSE>
# inputs:
#   - <INPUT_1>
# outputs:
#   - <OUTPUT_1>
# upstream_data:
#   - <UPSTREAM_DATA>
# downstream_data:
#   - Feature Engineering
#   - Analytics
#   - Dashboards
# dependencies:
#   - pathlib
#   - pandas optional
#   - sqlite3 optional
#   - json optional
# schedule: <SCHEDULE>
# version: v1.0.0
# last_reviewed: 2026-06-01
# business_criticality: <low/medium/high/critical>
# environment: desktop/server
# registry_group: pipeline
# author: Leon Everts
# reviewer: ChatGPT
# created_date: 2026-06-01
# tags:
#   - <TAG_1>
# notes:
#   - <NOTE_1>
# ============================================================

Runtime Registry:

CODE_REGISTRY = {
    "script_id": "<SCRIPT_ID>",
    "script_name": "<SCRIPT_NAME>",
    "owner": "Leon Everts",
    "status": "active",
    "layer": "1_Pipeline",
    "domain": "<DOMAIN>",
    "asset_type": "Pipeline",
    "purpose": "<PURPOSE>",
    "inputs": ["<INPUT_1>"],
    "outputs": ["<OUTPUT_1>"],
    "upstream_data": ["<UPSTREAM_DATA>"],
    "downstream_data": ["Feature Engineering", "Analytics", "Dashboards"],
    "dependencies": ["pathlib"],
    "schedule": "<SCHEDULE>",
    "version": "v1.0.0",
    "last_reviewed": "2026-06-01",
    "business_criticality": "<low/medium/high/critical>",
    "environment": "desktop",
    "registry_group": "pipeline",
    "author": "Leon Everts",
    "reviewer": "ChatGPT",
    "created_date": "2026-06-01",
    "tags": ["<TAG_1>"],
    "notes": ["<NOTE_1>"],
}

Root Detection:

def find_quant_root(start: Path) -> Path:
    """
    QUANT Root muss enthalten:
    - Dashboard/
    - Data_Center/
    - Data_Center/Data/
    - Data_Center/Backend_Management/
    """
    ...

Bitte gib mir den vollständigen fertigen code.py Code.

Keine Ausschnitte.
Keine Platzhalter.
Keine Erklärung außerhalb des Codes außer einem kurzen Hinweis.