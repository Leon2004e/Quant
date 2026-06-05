# WORKFLOW_04_DASHBOARD.md

# PURPOSE

Standardisierter Workflow zur Erstellung neuer QUANT Dashboards.

Ziel:

* Einheitliche Dashboards
* Konsistentes UI
* Mockup vor Code
* Standardisierte Builder-Nutzung
* Schnellere Entwicklung
* Weniger Layout-Fehler

---

# PHASE 01 — DASHBOARD REQUEST

## USER

Liefert:

### DATA CENTER TREE

Aktuelle Systemstruktur.

### MARKED ITEMS

Aktueller Arbeitsbereich.

### CODE LOCATION

Vollständiger Pfad.

Beispiel:

```text
C:\Users\Leon\Desktop\QUANT\Dashboard\Main.py
```

oder

```text
C:\Users\Leon\Desktop\QUANT\Dashboard\Building_Blocks\MyDashboard\code.py
```

### INPUTS

Alle Eingabedaten.

### OUTPUTS

Alle Ausgabedaten.

### PURPOSE

Was soll das Dashboard lösen?

### FEATURES

Gewünschte Funktionen.

### DASHBOARD TEMPLATE

Aktueller Dashboard Builder.

---

# PHASE 02 — SYSTEM ANALYSIS

## CHATGPT

Analysiert:

* Arbeitsbereich
* Layer
* Domain
* Inputs
* Outputs
* Datenstruktur
* KPIs
* Visualisierungsmöglichkeiten
* Dashboard-Typ

Ergebnis:

Dashboard Analyse.

---

# PHASE 03 — COMPONENT DESIGN

## CHATGPT

Definiert Dashboard Komponenten.

### HEADER

* Title
* Subtitle
* Refresh
* Scanner

### KPI SECTION

* KPI Cards
* KPI Mapping

### SEARCH SECTION

* Search
* Filter
* Summary

### NAVIGATION

* Tree
* Folder View
* Categories

### MAIN CONTENT

* Table
* Charts
* Heatmaps

### DETAILS PANEL

* Metadata
* Inputs
* Outputs
* Dependencies

### VISUAL FLOW

```text
INPUT
↓
SCRIPT
↓
OUTPUT
```

### ACTIONS

* Refresh
* Export
* Open
* Scanner

---

# PHASE 04 — ASCII WIREFRAME

## CHATGPT

Erstellt Wireframe.

Beispiel:

```text
┌──────────────────────────────────────┐
│ HEADER                               │
├──────────────────────────────────────┤
│ KPI KPI KPI KPI KPI                  │
├──────────┬──────────────┬────────────┤
│ NAV      │ MAIN TABLE   │ DETAILS    │
└──────────┴──────────────┴────────────┘
```

Zusätzlich:

* Panelgrößen
* Breiten
* Responsive Verhalten

---

# PHASE 05 — ASCII REVIEW

## USER

Prüft Wireframe.

Antwort:

```text
ASCII FREIGEGEBEN
```

oder

```text
ÄNDERUNGEN

...
```

Keine Mockups vor ASCII-Freigabe.

---

# PHASE 06 — VISUAL MOCKUP

## CHATGPT

Erstellt Dashboard Mockup.

Pflicht.

Mockup Standard:

* Dark Bloomberg Style
* Corporate Design
* Institutional Look
* Responsive Layout
* Professional Appearance

Mockup zeigt:

* Header
* KPI Cards
* Search
* Navigation
* Main Content
* Details Panel
* Visual Flow
* Buttons

---

# PHASE 07 — MOCKUP REVIEW

## USER

Prüft Mockup.

Antwort:

```text
MOCKUP FREIGEGEBEN
```

oder

```text
ÄNDERUNGEN

...
```

---

# PHASE 08 — FINAL DESIGN

## CHATGPT

Erstellt finale Dashboard Spezifikation.

Definiert:

* Komponenten
* Layout
* Datenquellen
* KPI Mapping
* Buttons
* Interaktionen
* Responsive Verhalten

---

# PHASE 09 — FINAL APPROVAL

## USER

Antwort:

```text
DESIGN FREIGEGEBEN
```

Erst danach darf Code erzeugt werden.

---

# PHASE 10 — BUILDER GENERATION

## CHATGPT

Verwendet:

* Dashboard Template
* Finales Design
* Dashboard Spezifikation

Erstellt finalen Dashboard Builder Prompt.

---

# PHASE 11 — CODE GENERATION

## CHATGPT

Erstellt vollständigen Produktionscode.

Pflicht:

* Registry Header
* find_quant_root()
* load_data()
* refresh_data()
* update_table()
* update_details()
* build_panel()

Zusätzlich:

* Error Handling
* Scrollbars
* Responsive Layout
* Main.py Kompatibilität

---

# PHASE 12 — TESTING

## USER

Startet Dashboard.

Liefert:

* Screenshot

oder

* Fehlermeldung

---

# PHASE 13 — FIX ROUND

## CHATGPT

Behebt:

* Fehler
* Layout Probleme
* Responsive Probleme
* Datenprobleme

---

# PHASE 14 — FINAL REVIEW

## CHATGPT

Prüft:

* Registry Header
* Main.py Kompatibilität
* Responsive Layout
* Scrollbars
* Search
* Filter
* Buttons
* Details Panel
* Visual Flow
* Error Handling

---

# PHASE 15 — PRODUCTION

## USER

Integriert Dashboard in QUANT.
ok 
Dashboard gilt als abgeschlossen.

---

# GOLDEN RULE

```text
KEIN CODE

OHNE

ASCII FREIGABE

UND

MOCKUP FREIGABE
```
