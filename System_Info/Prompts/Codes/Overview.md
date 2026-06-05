# QUANT SYSTEM OVERVIEW

## Mission

QUANT ist ein modulares Daten-, Research- und Analyseframework für systematisches Trading.

Ziel des Systems:

* Daten automatisiert erfassen
* Daten standardisieren
* Strategien analysieren
* Marktregime erforschen
* Portfolios bewerten
* Produktionsreife Modelle aufbauen
* Ergebnisse über Dashboards überwachen

---

# Hauptarchitektur

```text
Raw Data
    ↓
1_Pipeline
    ↓
2_Baseline
    ↓
3_Research
    ↓
4_Production
    ↓
Dashboard
```

Jede neue Komponente muss einer dieser Ebenen zugeordnet werden.

---

# Projektstruktur

```text
QUANT/
├── Dashboard/
├── Data_Center/
│   ├── Backend_Management/
│   └── Data/
├── System_Info/
└── tools/
```

---

# Backend Struktur

Verarbeitungscode:

```text
Data_Center/Backend_Management/
```

Layer:

```text
1_Pipelines
2_Baseline
3_Research
4_Production
5_Data_Catalog
6_Code_Registry
```

---

# Datenstruktur

Alle erzeugten Daten liegen unter:

```text
Data_Center/Data/
```

Layer:

```text
1_Pipeline
2_Baseline
3_Research
4_Production
5_Catalog
6_Code_Registry
```

---

# Dashboard

Dashboard-Code:

```text
Dashboard/
├── Main.py
└── Building_Blocks/
```

Das Dashboard dient ausschließlich:

* Visualisierung
* Monitoring
* Navigation
* Reporting

Geschäftslogik und Datenverarbeitung gehören nicht ins Dashboard.

---

# Aktuelle Kernmodule

## Pipeline Layer

Market:

* OHLC Logger
* Spread Logger

Trades:

* MT5 Trade Logger
* FTMO Demo Logger
* FTMO Live Logger

---

## Baseline Layer

Trades:

* Backtest Analytics
* Standardisierte Performance-Auswertungen

---

## Infrastruktur

### Data Catalog

Code:

```text
Data_Center/Backend_Management/5_Data_Catalog/
```

Datenbank:

```text
Data_Center/Data/5_Catalog/catalog.db
```

Zweck:

Zentrale Übersicht aller verfügbaren Datensätze.

---

### Code Registry

Code:

```text
Data_Center/Backend_Management/6_Code_Registry/
```

Datenbank:

```text
Data_Center/Data/6_Code_Registry/code_registry.db
```

Zweck:

Zentrale Übersicht aller Python-Komponenten.

---

# Wichtige Datenpfade

OHLC Daten:

```text
Data_Center/Data/1_Pipeline/Market/ohcl/
```

Trade Daten:

```text
Data_Center/Data/1_Pipeline/Trades/
```

Live Trades:

```text
Data_Center/Data/1_Pipeline/Trades/live/
```

Baseline Outputs:

```text
Data_Center/Data/2_Baseline/
```

Research Outputs:

```text
Data_Center/Data/3_Research/
```

Production Outputs:

```text
Data_Center/Data/4_Production/
```

---

# Entwicklungsprinzip

1_Pipeline

* Rohdaten erfassen
* APIs
* Broker
* Dateien
* Datenbanken

2_Baseline

* Validierung
* Standardisierung
* Bereinigung
* Basiskennzahlen

3_Research

* Regimeforschung
* Edge Analyse
* Portfolio Analyse
* Strategie Forschung

4_Production

* Produktionsmodelle
* Signalgenerierung
* Portfolio Entscheidungen
* Live Einsatz

Dashboard

* Monitoring
* Analyse
* Reporting

---

# Architekturregel

Neue Codes sollen immer:

1. In den passenden Layer einsortiert werden.
2. Daten nur in ihren vorgesehenen Data-Layer schreiben.
3. Die bestehende QUANT-Struktur nutzen.
4. Data Catalog und Code Registry berücksichtigen.
5. Das Dashboard nur für Visualisierung verwenden.

---

# Aktueller Entwicklungsstand

Vorhanden:

* Dashboard Framework
* Data Catalog
* Code Registry
* OHLC Pipeline
* Spread Pipeline
* MT5/FTMO Trade Pipelines
* Backtest Baseline

Research- und Production-Layer werden schrittweise aufgebaut.

Technische Implementierungsdetails einzelner Dashboards, Pipelines und Baselines werden über separate Builder-Prompts definiert und gehören nicht zu diesem Overview.
