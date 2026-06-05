# SYSTEM_OVERVIEW.md

# PURPOSE

Dieses Dokument beschreibt die grundlegende Architektur des QUANT Systems.

Ziel ist es, ChatGPT vor Beginn einer Aufgabe einen vollständigen Überblick über die Systemstruktur zu geben.

Die eigentliche Arbeitslogik befindet sich nicht hier.

Diese wird über die jeweiligen Workflow-Dokumente gesteuert.

---

# SYSTEM STRUCTURE

```text
QUANT
│
├── Dashboard
│   ├── Building_Blocks
│   │   ├── Code_Registry
│   │   ├── Data_Catalog
│   │   └── Pipeline_Management
│   │
│   └── Main.py
│
├── Data_Center
│   │
│   ├── Backend_Management
│   │   ├── 1_Pipelines
│   │   ├── 2_Baseline
│   │   ├── 3_Research
│   │   ├── 4_Production
│   │   ├── 5_Data_Catalog
│   │   └── 6_Code_Registry
│   │
│   └── Data
│       ├── 1_Pipeline
│       ├── 2_Baseline
│       ├── 3_Research
│       ├── 4_Production
│       ├── 5_Catalog
│       └── 6_Code_Registry
│
├── System_Info
│
└── tools
```

---

# DASHBOARD

Enthält sämtliche Benutzeroberflächen.

Building_Blocks enthält einzelne Dashboard-Module.

Beispiele:

* Code Registry
* Data Catalog
* Pipeline Management

Main.py dient als zentraler Dashboard-Einstiegspunkt.

---

# DATA_CENTER

Zentrales Verzeichnis für Backend-Code und Daten.

Unterteilt in:

```text
Backend_Management
Data
```

---

# BACKEND_MANAGEMENT

Enthält ausführbare Backend-Codes.

## 1_Pipelines

Verarbeitet Rohdaten.

Aufgaben:

* Import
* Cleaning
* Transformation
* Aggregation
* Datenaufbereitung

Output:

```text
Data/1_Pipeline
```

---

## 2_Baseline

Standardisiert Pipeline-Daten.

Aufgaben:

* Vereinheitlichung
* Standardisierung
* Normalisierung
* Erstellung wiederverwendbarer Basis-Datasets

Input:

```text
Data/1_Pipeline
```

Output:

```text
Data/2_Baseline
```

---

## 3_Research

Analyse- und Forschungsbereich.

Aufgaben:

* Edge Research
* Strategietests
* Statistische Analysen
* Experimente
* Modellvalidierung

Input:

```text
Data/2_Baseline
```

Output:

```text
Data/3_Research
```

---

## 4_Production

Produktionsreife Systeme.

Aufgaben:

* Produktionslogik
* Automatisierung
* Monitoring
* Deployment
* Live-Betrieb

Input:

```text
Data/3_Research
```

Output:

```text
Data/4_Production
```

---

## 5_Data_Catalog

Verwaltet den Data Catalog.

Aufgaben:

* Scanner
* Asset Discovery
* Metadatenverwaltung
* catalog.db Pflege

Output:

```text
Data/5_Catalog
```

---

## 6_Code_Registry

Verwaltet die Code Registry.

Aufgaben:

* Code Discovery
* Registry Scanning
* Registry Quality Checks
* code_registry.db Pflege

Output:

```text
Data/6_Code_Registry
```

---

# DATA

Enthält sämtliche erzeugten Daten.

## 1_Pipeline

Pipeline Outputs

## 2_Baseline

Standardisierte Daten

## 3_Research

Research Outputs

## 4_Production

Produktionsdaten

## 5_Catalog

Catalog Datenbank und Artefakte

## 6_Code_Registry

Registry Datenbank und Artefakte

---

# SYSTEM_INFO

Enthält:

* Dokumentation
* Architektur
* Prompts
* Standards
* Workflows
* Projektinformationen

---

# TOOLS

Enthält Hilfsprogramme und Utilities.

Keine produktiven Daten.

Keine Dashboards.

---

# REGEL

ChatGPT soll dieses Dokument ausschließlich als Systemkontext verwenden.

Arbeitsanweisungen werden nicht aus dem System Overview abgeleitet.

Die Arbeitslogik wird ausschließlich durch die jeweiligen Workflow-Dokumente definiert.

---

# REIHENFOLGE

```text
SYSTEM OVERVIEW
↓
WORKFLOW
↓
AUSFÜHRUNG
```

ChatGPT muss zuerst das System verstehen.

Danach den Workflow lesen.

Erst danach darf mit der Ausführung begonnen werden.
