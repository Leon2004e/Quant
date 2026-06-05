# ============================================================

# CODE_REGISTRY

# ============================================================

# script_id: <SCRIPT_ID>

# script_name: <SCRIPT_NAME>

# owner: Leon

# status: active

# layer: Dashboard

# domain: <DOMAIN>

# asset_type: Dashboard

# purpose: <PURPOSE>

# inputs:

# - <INPUT_1>

# outputs:

# - Dashboard UI

# dependencies:

# - tkinter

# - pathlib

# - pandas (optional)

# - sqlite3 (optional)

# schedule: manual

# version: v1.0.0

# last_reviewed: 2026-06-01

# ============================================================

Ich brauche einen kompletten Python-Tkinter-Code für ein neues QUANT Dashboard Building Block.

Projektstruktur:

Business_Code/
├── QUANT
│   ├── Dashboard
│   │   ├── Building_Blocks
│   │   │   └── <BLOCK_NAME>
│   │   │       └── code.py
│   │   └── Main.py
│   └── Data_Center
│       ├── Backend_Management
│       └── Data

Erstelle die komplette Datei:

QUANT/Dashboard/Building_Blocks/<BLOCK_NAME>/code.py

Dashboard Name:

<BLOCK_NAME>

Dashboard Beschreibung:

<BLOCK_DESCRIPTION>

Datenquelle:

<DATA_SOURCE>

Scanner:

<SCANNER_PATH ODER "kein Scanner">

Anforderungen:

1. Vollständiger lauffähiger Code.
2. Keine Platzhalter im finalen Code.
3. Keine Pseudocode-Blöcke.
4. Fehlerbehandlung für alle Datei- und Datenzugriffe.
5. Kompatibel mit Main.py.
6. Responsive Layout.
7. Dark Bloomberg Style.
8. Keine externen UI Libraries.
9. Keine festen Fenstergrößen.
10. Scrollbars für alle Tabellen.
11. Keine Inhalte dürfen abgeschnitten werden.

Pflichtfunktionen:

* find_quant_root()
* refresh_data()
* load_data()
* update_table()
* update_details()
* build_panel()

Pflicht API:

def build_panel(parent, repository=None, **kwargs):
return <DashboardClass>(parent, repository=repository, **kwargs)

Root Detection:

Der QUANT Root enthält:

Dashboard/
Data_Center/

Nutze:

def find_quant_root(start: Path) -> Path:
...

Dashboard Layout:

1. Header

   * Titel
   * Untertitel
   * Refresh Button
   * Optional Scanner Button

2. KPI Section

   * 4 bis 8 KPI Cards
   * Responsive Wrapping

3. Search Section

   * Suchfeld
   * Filter
   * Summary Counter

4. Main Section (PanedWindow)

Links:

* Navigation
* Kategorien
* Ordnerstruktur

Mitte:

* Haupttabelle
* Sortierbar
* Scrollbar

Rechts:

* Details Panel
* Inputs
* Outputs
* Metadaten

Responsive Verhalten:

* Kleine Breite → KPI Grid umbrechen
* Kleine Breite → Details Panel ausblendbar
* Kleine Breite → PanedWindow vertikal
* Tabelle immer scrollbar

Registry Integration:

Das Dashboard muss vorbereitet sein für:

code_registry.db

Tabellenbeispiele:

scripts
script_dependencies
script_inputs
script_outputs

Wenn vorhanden:

* Script Name anzeigen
* Domain anzeigen
* Layer anzeigen
* Asset Type anzeigen
* Inputs anzeigen
* Outputs anzeigen
* Dependencies anzeigen

Visual Flow Bereich:

Zusätzlich einen Bereich vorsehen für:

INPUTS
↓
SCRIPT
↓
OUTPUTS

damit später Datenflüsse visualisiert werden können.

Bitte liefere den vollständigen Inhalt der Datei:

QUANT/Dashboard/Building_Blocks/<BLOCK_NAME>/code.py

Keine Erklärungen außerhalb des Codes außer einem kurzen Hinweis.
