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
* Bloomberg-konforme Dashboard-Struktur
* Wiederverwendbare Dashboard-Komponenten

---

# REQUIRED CONTEXT

Vor Beginn müssen folgende Dokumente geladen sein.

## SYSTEM OVERVIEW

Definiert:

* Architektur
* Ordnerstruktur
* Datenfluss
* Layer

---

## WORKFLOW_04_DASHBOARD

Definiert:

* Dashboard Prozess
* Freigaben
* Reihenfolge

---

## QUANT TERMINAL UI DESIGN STANDARD

Definiert:

* Farben
* Fonts
* Tabellen
* Layout
* Bloomberg Style
* OMS/PMS Style
* Research Style

Pflicht für:

* Component Design
* ASCII Wireframe
* Mockup
* Final Design
* Code Generation

---

## DASHBOARD BUILDER TEMPLATE

Definiert:

* Registry Header
* Dashboard Struktur
* Pflichtfunktionen
* Main.py Integration
* Root Detection
* Standard Layout
* Error Handling

Pflicht für:

* Builder Generation
* Code Generation

---

# PHASE 01 — DASHBOARD REQUEST

## USER

Liefert:

### DATA CENTER TREE

Aktuelle Systemstruktur.

---

### MARKED ITEMS

Aktueller Arbeitsbereich.

---

### CODE LOCATION

Vollständiger Pfad.

Beispiel:

C:\Users\Leon\Desktop\QUANT\Dashboard\Main.py

oder

C:\Users\Leon\Desktop\QUANT\Dashboard\Building_Blocks\MyDashboard\code.py

---

### INPUTS

Alle Eingabedaten.

---

### OUTPUTS

Alle Ausgabedaten.

---

### PURPOSE

Was soll das Dashboard lösen?

---

### FEATURES

Gewünschte Funktionen.

---

### UI STANDARD

Aktiver UI Standard.

Beispiel:

QUANT TERMINAL UI DESIGN STANDARD

---

### BUILDER TEMPLATE

Aktiver Dashboard Builder.

Beispiel:

QUANT Dashboard Builder Template

---

# PHASE 02 — SYSTEM ANALYSIS

## CHATGPT

Antwortformat:

PHASE 02 — DASHBOARD ANALYSE

ARBEITSBEREICH

...

LAYER

...

DOMAIN

...

INPUTS

...

OUTPUTS

...

KPIs

...

VISUALISIERUNGEN

...

DASHBOARD TYP

...

---

# PHASE 03 — COMPONENT DESIGN

## CHATGPT

Definiert Dashboard Komponenten.

Antwortformat:

PHASE 03 — COMPONENT DESIGN

HEADER

...

KPI SECTION

...

SEARCH SECTION

...

NAVIGATION

...

MAIN CONTENT

...

DETAILS PANEL

...

VISUAL FLOW

...

ACTIONS

...

Pflicht:

UI Standard berücksichtigen.

---

# PHASE 04 — ASCII WIREFRAME

## CHATGPT

Erstellt ASCII Wireframe.

Antwortformat:

PHASE 04 — ASCII WIREFRAME

ASCII

...

PANEL BREITEN

...

RESPONSIVE VERHALTEN

...

Keine weiteren Inhalte.

---

# PHASE 05 — ASCII REVIEW

## USER

Antwort:

ASCII FREIGEGEBEN

oder

ÄNDERUNGEN

...

Keine Mockups vor ASCII Freigabe.

---

# PHASE 06 — VISUAL MOCKUP

## CHATGPT

Erstellt Dashboard Mockup.

Pflicht:

* Bloomberg Style
* OMS Style
* PMS Style
* Research Style

Antwortformat:

PHASE 06 — VISUAL MOCKUP

MOCKUP

...

DESIGN NOTIZEN

...

Keine Codevorschläge.

---

# PHASE 07 — MOCKUP REVIEW

## USER

Antwort:

MOCKUP FREIGEGEBEN

oder

ÄNDERUNGEN

...

---

# PHASE 08 — FINAL DESIGN

## CHATGPT

Antwortformat:

PHASE 08 — FINAL DESIGN

KOMPONENTEN

...

DATENQUELLEN

...

KPI MAPPING

...

BUTTONS

...

FILTER

...

INTERAKTIONEN

...

RESPONSIVE VERHALTEN

...

VISUAL FLOW

...

---

# PHASE 09 — FINAL APPROVAL

## USER

Antwort:

DESIGN FREIGEGEBEN

oder

ÄNDERUNGEN

...

Erst danach darf Builder Generation erfolgen.

---

# PHASE 10 — BUILDER GENERATION

## CHATGPT

Verwendet:

* Final Design
* UI Standard
* Dashboard Builder Template

Erstellt:

FINAL BUILDER PROMPT

Antwortformat:

PHASE 10 — BUILDER GENERATION

FINAL BUILDER PROMPT

...

---

# PHASE 10.5 — BUILDER REVIEW

## USER

Antwort:

BUILDER FREIGEGEBEN

oder

ÄNDERUNGEN

...

Erst danach darf Code Generation erfolgen.

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

Pflicht Inputs:

* Final Design
* UI Standard
* Builder Template

Fehlt eines davon:

CODE GENERATION GESPERRT

---

# PHASE 12 — ISSUE RESOLUTION

Optional.

Nur bei Fehlern.

## USER

Liefert:

* Screenshot

oder

* Fehlermeldung

## CHATGPT

Behebt:

* Fehler
* Layout Probleme
* Responsive Probleme
* Datenprobleme
* Registry Probleme
* Main.py Probleme

---

# PHASE 13 — PRODUCTION

## USER

Antwort:

CODE FUNKTIONIERT

Dashboard wird in QUANT integriert.

Dashboard gilt als abgeschlossen.

---

# WORKFLOW RESPONSE RULES

## RULE 01

Nur aktuelle Phase bearbeiten.

---

## RULE 02

Keine Einleitungen.

---

## RULE 03

Keine Workflow-Erklärungen.

---

## RULE 04

Keine zukünftigen Phasen vorwegnehmen.

---

## RULE 05

Kein Code vor PHASE 11.

---

## RULE 06

Keine Annahmen bei fehlenden Inputs.

Fehlende Inputs explizit auflisten.

---

## RULE 07

Antworten immer im Format der aktuellen Phase.

---

## RULE 08

UI Standard ist verpflichtend.

---

## RULE 09

Builder Template ist verpflichtend.

---

## RULE 10

Bloomberg Terminal Design hat Vorrang.

---

# GOLDEN RULE

KEIN CODE

OHNE

ASCII FREIGABE

UND

MOCKUP FREIGABE

UND

DESIGN FREIGABE

UND

BUILDER FREIGABE
