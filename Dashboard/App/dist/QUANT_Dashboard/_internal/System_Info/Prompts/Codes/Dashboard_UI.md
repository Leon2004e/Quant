QUANT TERMINAL UI DESIGN STANDARD v1.0
Ziel

Das System soll aussehen wie:

70% Bloomberg Terminal
20% Hedge Fund OMS/PMS
10% Modern Quant Dashboard

Nicht:

Moderne SaaS App
Runde Ecken
Bunte Karten
Große Buttons

Sondern:

Institutionelles Trading Terminal
Hohe Informationsdichte
Dunkler Terminal-Look
Schnelle Lesbarkeit
GLOBAL THEME
Hintergrund
BG = "#000000"
Panels
PANEL_BG = "#0A0A0A"
PANEL_BG_DARK = "#050505"
PANEL_BG_LIGHT = "#151515"
Borders
BORDER = "#2A2A2A"

Immer:

highlightthickness=1
relief="solid"

Keine Schatten.

Keine modernen Cards.

FARBEN
Bloomberg Orange
ORANGE = "#FF9900"

Für:

Header
Navigation
Panel Titel
Aktive Auswahl
Bloomberg Gelb
YELLOW = "#FFD400"

Für:

Warnungen
Markierte Werte
Grün
GREEN = "#00FF66"

Für:

Positive PnL
Gewinner
Long Exposure
Rot
RED = "#FF4444"

Für:

Negative PnL
Drawdown
Verluste
Blau
BLUE = "#00AEEF"

Für:

Links
News
Information
SCHRIFTEN

Ausschließlich:

FONT_TITLE = ("Consolas", 13, "bold")
FONT_HEAD = ("Consolas", 10, "bold")
FONT_MAIN = ("Consolas", 9)
FONT_SMALL = ("Consolas", 8)
FONT_TINY = ("Consolas", 7)

Keine:

Segoe UI
Roboto
Open Sans
LAYOUT
Top Bar

Immer:

┌─────────────────────────────┐
│ QUANT TERMINAL             │
│ TIME                       │
└─────────────────────────────┘

Navigation:

OVRV
PORT
STRAT
TRDS
RSCH
REG
SET
KPI STRIP

Immer direkt unter Navigation.

Net PnL Today
Net PnL MTD
Net PnL YTD
Open Positions
Open Risk
Win Rate
PF
Max DD
Sharpe
WATCHLIST

Links oben.

Symbol
Last
Change %
Bid
Ask
Spread
NEWS PANEL

Rechts oben.

Time
Headline
Category

Filter:

Macro
Forex
Indices
Commodities
Crypto
EQUITY PANEL

Mitte.

Größtes Panel.

Enthält:

Equity Curve
Return
Sharpe
Sortino
Max DD
PERFORMANCE CALENDAR

Monatsansicht.

Farben:

Dunkelgrün
Hellgrün
Dunkelrot
Hellrot

Keine Farbverläufe.

ALERT CENTER

Links Mitte.

Red Dot
Yellow Dot
Green Dot

Alerts:

Risk
Strategy
System
Portfolio
Research
TRADE BLOTTER

Unten.

Spalten:

Time
Symbol
Strategy
Side
Volume
Entry
Exit
PnL
Status

Eigenschaften:

Monospace
Kompakt
20px Zeilenhöhe
STRATEGY TABLE

Rechts unten.

Strategy
PnL
Return
Sharpe
Win Rate
PF
Max DD
REGISTRY DESIGN

Code Registry bekommt denselben Stil.

Script ID
Layer
Domain
Inputs
Outputs
Dependencies
Version

Als Terminal-Tabelle.

RESEARCH DESIGN

Panels:

Strategy Research
Regime Research
Portfolio Research
Monte Carlo
Walk Forward

Alle Panels:

schwarzer Hintergrund
orange Header
weiße Schrift
TABELLEN STANDARD

Immer:

row_height = 20

Header:

bg="#111111"
fg="#FF9900"
font=("Consolas", 8, "bold")

Rows:

bg="#000000"
fg="#FFFFFF"

Selected:

bg="#332000"
fg="#FFD400"
VERBOTEN

Keine:

Runden Ecken
Gradienten
Glassmorphism
Neon Glow
Große Karten
Mobile Design
SaaS Design
DESIGN PHILOSOPHIE

Jeder neue Building Block muss aussehen wie:

Bloomberg Terminal
+
Institutionelles OMS
+
Hedge Fund Research System

und nicht wie:

Web Dashboard
Startup SaaS
TradingView Klon

Diesen Prompt kannst du künftig jedem neuen Dashboard-Block voranstellen, damit das gesamte QUANT-System optisch konsistent bleibt.