from __future__ import annotations

from dataclasses import dataclass
import html
import json
from pathlib import Path
import re
from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from scipy.optimize import least_squares, minimize


st.set_page_config(
    page_title="Eco Assignment Solver Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


WORKBOOK_PATH = Path("Assignment_2.xlsx")
PDF_PATH = Path("Eco_Assignment_2.pdf")


@dataclass(frozen=True)
class RowSpec:
    cell: str
    item: str
    lower: float
    upper: float
    default: float
    section: str
    formula: str


CAPITAL_SPECS = [
    RowSpec("D4", "Working Capital", 0.10, 0.20, 0.20, "Top level", "E4 = target TCI x D4"),
    RowSpec("D5", "Fixed Capital Investment", 0.00, 0.80, 0.80, "Top level", "E5 = target TCI x D5"),
    RowSpec("D9", "Purchased Equipment Cost", 0.15, 0.40, 0.23927228120613753, "Direct costs", "E9 = E5 x D9"),
    RowSpec("D10", "Installation Costs", 0.25, 0.55, 0.26671245019817313, "Direct costs", "E10 = E9 x D10"),
    RowSpec("D11", "Instrumentation Costs", 0.08, 0.50, 0.08271351374768844, "Direct costs", "E11 = E9 x D11"),
    RowSpec("D12", "Piping Cost", 0.10, 0.80, 0.10387263597160327, "Direct costs", "E12 = E9 x D12"),
    RowSpec("D13", "Electricity Costs", 0.10, 0.40, 0.10387263597160329, "Direct costs", "E13 = E9 x D13"),
    RowSpec("D14", "Buildings, process, and auxiliary", 0.10, 0.70, 0.10797980798584342, "Direct costs", "E14 = E9 x D14"),
    RowSpec("D15", "Service facilities and yard improvements", 0.40, 1.00, 0.43277156283783685, "Direct costs", "E15 = E9 x D15"),
    RowSpec("D16", "Land", 0.04, 0.08, 0.04211068150794063, "Direct costs", "E16 = E9 x D16"),
    RowSpec("D23", "Engineering and Supervision", 0.05, 0.30, 0.19561835278220646, "Indirect costs", "E23 = E18 x D23"),
    RowSpec("D24", "Legal Expenses", 0.01, 0.03, 0.024929643456153963, "Indirect costs", "E24 = target TCI x D24"),
    RowSpec("D25", "Construction Expense and Contractor fee", 0.10, 0.20, 0.16377462912934274, "Indirect costs", "E25 = target TCI x D25"),
    RowSpec("D26", "Contingency", 0.05, 0.15, 0.12152199833527207, "Indirect costs", "E26 = target TCI x D26"),
]

PRODUCT_SPECS = [
    RowSpec("D5", "Direct Production Costs", 0.00, 1.00, 0.6766642181356277, "Manufacturing summary", "E5 = target TCP x D5"),
    RowSpec("D6", "Raw Materials", 0.10, 0.80, 0.3111871645942623, "Direct production details", "E6 = target TCP x D6"),
    RowSpec("D7", "Operating Labour", 0.10, 0.20, 0.13022066737799068, "Direct production details", "E7 = target TCP x D7"),
    RowSpec("D8", "Direct Supervisory and clerical labour", 0.10, 0.20, 0.1252333206685434, "Direct production details", "E8 = E7 x D8"),
    RowSpec("D9", "Utilities", 0.10, 0.20, 0.12561694329816703, "Direct production details", "E9 = target TCP x D9"),
    RowSpec("D10", "Maintenance and Repairs", 0.02, 0.10, 0.06651962735758735, "Direct production details", "E10 = E5 x D10"),
    RowSpec("D11", "Operating Supplies", 0.005, 0.010, 0.007605513982554804, "Direct production details", "E11 = E5 x D11"),
    RowSpec("D12", "Laboratory Charges", 0.10, 0.20, 0.12499512219842353, "Direct production details", "E12 = E8 x D12"),
    RowSpec("D13", "Patents and Royalties", 0.00, 0.06, 0.015328728775059322, "Direct production details", "E13 = target TCP x D13"),
    RowSpec("D16", "Fixed Charges", 0.10, 0.20, 0.10, "Fixed charges summary", "E16 = target TCP x D16"),
    RowSpec("D19", "Local Taxes", 0.01, 0.04, 0.02500431109929442, "Fixed charges details", "E19 = E5 x D19"),
    RowSpec("D20", "Insurance", 0.004, 0.010, 0.007000761877550604, "Fixed charges details", "E20 = E5 x D20"),
    RowSpec("D21", "Rent", 0.08, 0.12, 0.09, "Fixed charges details", "E21 = rent base x D21"),
    RowSpec("D22", "Financing (interest)", 0.00, 0.10, 0.07439594161852621, "Fixed charges details", "E22 = target TCP x D22"),
    RowSpec("D26", "Plant Overhead Costs", 0.05, 0.15, 0.05, "Plant overhead", "E26 = target TCP x D26"),
    RowSpec("D32", "General Expenses", 0.15, 0.25, 0.17333586218576122, "General expenses summary", "E32 = target TCP x D32"),
    RowSpec("D33", "Administrative Costs", 0.02, 0.05, 0.02, "General expenses details", "E33 = target TCP x D33"),
    RowSpec("D34", "Distribution and Marketing Costs", 0.02, 0.20, 0.02, "General expenses details", "E34 = target TCP x D34"),
    RowSpec("D35", "Research and Development Costs", 0.04, 0.06, 0.04, "General expenses details", "E35 = target TCP x D35"),
]

CAPITAL_DEFAULTS = {spec.cell: spec.default for spec in CAPITAL_SPECS}
PRODUCT_DEFAULTS = {spec.cell: spec.default for spec in PRODUCT_SPECS}
ASSIGNMENT_TARGET = 100000.0
FIXED_SOLVER_EVALS = 2000
PREVIEW_STEPS = 2000


def inject_css() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"] {
            display: none !important;
        }
        .main .block-container {
            padding-top: 1rem;
            padding-left: 1.25rem;
            padding-right: 1.25rem;
            padding-bottom: 3rem;
            max-width: 98vw;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 0.7rem 0.8rem;
            border-radius: 10px;
        }
        .hero {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 1rem 1.1rem;
            background: linear-gradient(180deg, rgba(19,25,39,0.95), rgba(14,18,28,0.92));
            margin-bottom: 1rem;
        }
        .hero h1 {
            font-size: 1.8rem;
            margin: 0 0 0.3rem 0;
        }
        .hero p {
            margin: 0;
            color: rgba(250,250,250,0.75);
            line-height: 1.5;
        }
        .professor-line {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            margin: 0.15rem 0 0.7rem 0;
            padding: 0.28rem 0.65rem;
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 999px;
            color: rgba(250,250,250,0.82);
            background: rgba(255,255,255,0.04);
            font-size: 0.9rem;
        }
        .friendly-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 0.75rem 0 1rem 0;
        }
        .friendly-card {
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 8px;
            padding: 1rem;
            background: rgba(255,255,255,0.025);
            min-height: 155px;
        }
        .friendly-card h3 {
            margin: 0 0 0.55rem 0;
            font-size: 1.05rem;
        }
        .friendly-card p {
            margin: 0;
            color: rgba(250,250,250,0.72);
            line-height: 1.55;
            font-size: 0.93rem;
        }
        .plain-equation {
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            background: rgba(18, 23, 35, 0.92);
            margin-bottom: 0.75rem;
        }
        .plain-equation .eq-title {
            color: rgba(250,250,250,0.55);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.25rem;
        }
        .plain-equation .eq-body {
            color: rgba(250,250,250,0.92);
            font-size: 1.05rem;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        }
        .walkthrough {
            counter-reset: step;
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 0.75rem;
        }
        .walkthrough-card {
            counter-increment: step;
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 8px;
            padding: 0.9rem;
            background: rgba(255,255,255,0.025);
        }
        .walkthrough-card::before {
            content: counter(step);
            display: inline-grid;
            place-items: center;
            width: 1.45rem;
            height: 1.45rem;
            border-radius: 999px;
            background: #ef5350;
            color: white;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }
        .walkthrough-card h4 {
            margin: 0 0 0.35rem 0;
            font-size: 0.98rem;
        }
        .walkthrough-card p {
            margin: 0;
            color: rgba(250,250,250,0.7);
            font-size: 0.88rem;
            line-height: 1.45;
        }
        @media (max-width: 900px) {
            .friendly-grid,
            .walkthrough {
                grid-template-columns: 1fr;
            }
        }
        .solver-card {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 0.95rem 1rem;
            background: rgba(255,255,255,0.02);
            height: 100%;
        }
        .solver-card h3 {
            margin-top: 0;
            margin-bottom: 0.65rem;
            font-size: 1.05rem;
        }
        .chip-row {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin: 0.5rem 0 0.2rem 0;
        }
        .chip {
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 999px;
            padding: 0.2rem 0.6rem;
            font-size: 0.8rem;
            color: rgba(250,250,250,0.82);
            background: rgba(255,255,255,0.03);
        }
        .mini-note {
            color: rgba(250,250,250,0.68);
            font-size: 0.9rem;
            line-height: 1.45;
        }
        .toolbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            margin-bottom: 0.8rem;
        }
        .toolbar-title {
            font-size: 0.95rem;
            color: rgba(250,250,250,0.7);
        }
        .notebook-cell {
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 8px;
            padding: 0.95rem 1rem;
            background: rgba(255,255,255,0.02);
            margin-bottom: 0.9rem;
        }
        .notebook-cell h3 {
            margin: 0 0 0.55rem 0;
            font-size: 1.02rem;
        }
        .excel-frame {
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 8px;
            padding: 0.55rem;
            background: rgba(255,255,255,0.02);
            margin-bottom: 0.9rem;
        }
        .drawer-hint {
            color: rgba(250,250,250,0.6);
            font-size: 0.88rem;
            margin-top: 0.15rem;
        }
        div[data-testid="stDataEditor"] {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 8px;
            overflow: hidden;
        }
        div[data-testid="stDataEditor"] [role="grid"] {
            border: none !important;
        }
        .sheet-shell {
            background-color: #ffffff;
            background-image:
                linear-gradient(135deg, transparent 0 58%, #a9b6ca 59% 64%, transparent 65%),
                linear-gradient(135deg, transparent 0 70%, #a9b6ca 71% 76%, transparent 77%);
            background-repeat: no-repeat;
            background-position: right 8px bottom 8px, right 14px bottom 8px;
            background-size: 18px 18px, 18px 18px;
            border: 1px solid #cfd6e0;
            border-radius: 8px;
            padding: 10px;
            overflow: auto;
            resize: both;
            width: 100%;
            height: 66vh;
            min-width: 420px;
            min-height: 300px;
            max-width: 96vw;
            max-height: 86vh;
            scrollbar-gutter: stable both-edges;
            margin-bottom: 1rem;
            box-shadow: 0 0 0 1px rgba(255,255,255,0.45);
            cursor: default;
        }
        .sheet-shell:hover {
            border-color: #8fb4ff;
        }
        .sheet-shell::-webkit-scrollbar {
            width: 14px;
            height: 14px;
        }
        .sheet-shell::-webkit-scrollbar-track {
            background: #eef2f7;
            border-radius: 999px;
        }
        .sheet-shell::-webkit-scrollbar-thumb {
            background: #c0cada;
            border-radius: 999px;
            border: 2px solid #eef2f7;
        }
        .sheet-shell::-webkit-scrollbar-thumb:hover {
            background: #a9b6ca;
        }
        table.excel-sheet {
            border-collapse: collapse;
            width: max-content;
            min-width: 100%;
            background: #ffffff;
        }
        table.excel-sheet td {
            border: 1px solid #cfd6e0;
            min-width: 84px;
            height: 30px;
            padding: 3px 8px;
            font-size: 13px;
            line-height: 1.15;
            color: #111827;
            vertical-align: middle;
            background: #ffffff;
            white-space: nowrap;
        }
        table.excel-sheet td.sheet-header {
            background: #f1f5f9;
            color: #475467;
            font-weight: 600;
            text-align: center;
            min-width: 52px;
        }
        table.excel-sheet td.sheet-header-active {
            background: #dbe7ff;
            color: #173a8a;
            font-weight: 700;
        }
        table.excel-sheet td.sheet-corner {
            background: #e8edf5;
            min-width: 52px;
        }
        table.excel-sheet td.cell-active {
            position: relative;
            background: #e8f0fe !important;
            box-shadow: inset 0 0 0 2px #3b82f6;
        }
        table.excel-sheet td .sheet-link {
            display: block;
            width: calc(100% + 16px);
            margin: -3px -8px;
            padding: 3px 8px;
            color: inherit;
            text-decoration: none;
        }
        table.excel-sheet td .sheet-link:hover {
            background: rgba(59, 130, 246, 0.08);
        }
        table.excel-sheet td.text {
            text-align: left;
        }
        table.excel-sheet td.num {
            text-align: right;
            font-variant-numeric: tabular-nums;
        }
        table.excel-sheet td.center {
            text-align: center;
        }
        table.excel-sheet td.title-yellow {
            background: #fff45c;
            font-weight: 700;
        }
        table.excel-sheet td.section-yellow {
            background: #fff45c;
            font-weight: 700;
        }
        table.excel-sheet td.head-green {
            background: #afc3ae;
            font-weight: 600;
        }
        table.excel-sheet td.head-peach {
            background: #e3c6ad;
            font-weight: 600;
        }
        table.excel-sheet td.total-green {
            background: #bfd3bf;
            font-weight: 600;
        }
        table.excel-sheet td.soft {
            color: #4b5563;
        }
        .gs-toolbar {
            background: #eef2f7;
            border: 1px solid #d9e0ea;
            border-radius: 14px 14px 0 0;
            padding: 10px 14px;
            margin-bottom: 0;
        }
        .gs-toolbar .row {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        .gs-chip {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 28px;
            height: 28px;
            border: 1px solid #d1d9e6;
            border-radius: 7px;
            background: #ffffff;
            color: #475467;
            font-size: 13px;
            padding: 0 8px;
        }
        .gs-chip.wide {
            min-width: 76px;
        }
        .gs-divider {
            width: 1px;
            height: 22px;
            background: #d1d9e6;
            margin: 0 2px;
        }
        .formula-shell {
            display: grid;
            grid-template-columns: 120px 1fr 140px;
            gap: 10px;
            align-items: center;
            background: #f8fafc;
            border-left: 1px solid #d9e0ea;
            border-right: 1px solid #d9e0ea;
            border-bottom: 1px solid #d9e0ea;
            padding: 10px 14px;
            margin-bottom: 0.9rem;
        }
        .formula-label {
            color: #4b5563;
            font-size: 12px;
            margin-bottom: 3px;
        }
        .formula-pill {
            border: 1px solid #d1d9e6;
            background: #ffffff;
            color: #111827;
            padding: 7px 10px;
            border-radius: 8px;
            font-size: 13px;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        }
        .iteration-panel {
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 12px;
            padding: 0.95rem 1rem 0.8rem 1rem;
            background: linear-gradient(180deg, rgba(22,28,42,0.98), rgba(16,21,32,0.96));
            margin-top: 0.1rem;
        }
        .iteration-panel.compact {
            padding: 0.55rem 0.65rem 0.35rem 0.65rem;
            border-radius: 8px;
        }
        .iteration-panel h4 {
            margin: 0 0 0.25rem 0;
            font-size: 1rem;
        }
        .iteration-panel.compact h4 {
            font-size: 0.88rem;
            margin-bottom: 0.15rem;
        }
        .iteration-panel p {
            margin: 0;
            color: rgba(250,250,250,0.68);
            font-size: 0.88rem;
            line-height: 1.4;
        }
        .iteration-chip-row {
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin: 0.75rem 0 0.25rem 0;
        }
        .iteration-panel.compact .iteration-chip-row {
            gap: 0.35rem;
            margin: 0.3rem 0 0 0;
        }
        .iteration-chip {
            border: 1px solid rgba(96, 165, 250, 0.22);
            background: rgba(96, 165, 250, 0.08);
            color: #dbeafe;
            border-radius: 999px;
            padding: 0.22rem 0.65rem;
            font-size: 0.8rem;
        }
        .iteration-panel.compact .iteration-chip {
            padding: 0.14rem 0.45rem;
            font-size: 0.72rem;
        }
        div[data-testid="stSlider"] {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 12px;
            padding: 0.55rem 0.85rem 0.15rem 0.85rem;
            margin-bottom: 0.55rem;
        }
        div[data-testid="stProgress"] {
            margin-top: 0.25rem;
            margin-bottom: 0.15rem;
        }
        div[data-testid="stSlider"] label p {
            font-weight: 600;
            color: rgba(250,250,250,0.92) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_money(value: float) -> str:
    return f"{value:,.2f}"


def build_bounds(specs: list[RowSpec]) -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([spec.lower for spec in specs], dtype=float)
    upper = np.array([spec.upper for spec in specs], dtype=float)
    return lower, upper


def defaults_vector(specs: list[RowSpec], values: dict[str, float]) -> np.ndarray:
    return np.array([values[spec.cell] for spec in specs], dtype=float)


def vector_to_values(specs: list[RowSpec], x: np.ndarray) -> dict[str, float]:
    return {spec.cell: float(value) for spec, value in zip(specs, x)}


def randomize_values(specs: list[RowSpec]) -> dict[str, float]:
    rng = np.random.default_rng()
    return {spec.cell: float(rng.uniform(spec.lower, spec.upper)) for spec in specs}


def append_path_snapshot(
    path: list[dict[str, float]],
    specs: list[RowSpec],
    x: np.ndarray,
) -> None:
    snapshot = vector_to_values(specs, x)
    if not path:
        path.append(snapshot)
        return
    keys = snapshot.keys()
    if any(abs(snapshot[key] - path[-1][key]) > 1e-12 for key in keys):
        path.append(snapshot)


def capital_calculator(values: dict[str, float], target_tci: float) -> dict[str, float]:
    d = values
    e4 = target_tci * d["D4"]
    e5 = target_tci * d["D5"]
    e9 = e5 * d["D9"]
    e10 = e9 * d["D10"]
    e11 = e9 * d["D11"]
    e12 = e9 * d["D12"]
    e13 = e9 * d["D13"]
    e14 = e9 * d["D14"]
    e15 = e9 * d["D15"]
    e16 = e9 * d["D16"]
    e18 = e9 + e10 + e11 + e12 + e13 + e14 + e15 + e16
    e23 = e18 * d["D23"]
    e24 = target_tci * d["D24"]
    e25 = target_tci * d["D25"]
    e26 = target_tci * d["D26"]
    e28 = e23 + e24 + e25 + e26
    e30 = e18 + e28
    e31 = e30 + e4
    return {
        "D4_cost": e4,
        "D5_cost": e5,
        "D9_cost": e9,
        "D10_cost": e10,
        "D11_cost": e11,
        "D12_cost": e12,
        "D13_cost": e13,
        "D14_cost": e14,
        "D15_cost": e15,
        "D16_cost": e16,
        "E18": e18,
        "D23_cost": e23,
        "D24_cost": e24,
        "D25_cost": e25,
        "D26_cost": e26,
        "E28": e28,
        "E30": e30,
        "E31": e31,
        "fci_balance": e5 - e30,
        "target_residual": e31 - target_tci,
    }


def product_calculator(
    values: dict[str, float],
    target_tcp: float,
    rent_base: float,
    depreciation_cost: float,
) -> dict[str, float]:
    d = values
    e5 = target_tcp * d["D5"]
    e6 = target_tcp * d["D6"]
    e7 = target_tcp * d["D7"]
    e8 = d["D8"] * e7
    e9 = target_tcp * d["D9"]
    e10 = d["D10"] * e5
    e11 = d["D11"] * e5
    e12 = d["D12"] * e8
    e13 = d["D13"] * target_tcp
    e14 = e6 + e7 + e8 + e9 + e10 + e11 + e12 + e13

    e16 = target_tcp * d["D16"]
    e18 = depreciation_cost
    e19 = d["D19"] * e5
    e20 = d["D20"] * e5
    e21 = d["D21"] * rent_base
    e22 = d["D22"] * target_tcp
    e23 = e18 + e19 + e20 + e21 + e22

    e26 = d["D26"] * target_tcp
    e28 = e14 + e23 + e26

    e32 = target_tcp * d["D32"]
    e33 = d["D33"] * target_tcp
    e34 = d["D34"] * target_tcp
    e35 = d["D35"] * target_tcp
    e37 = e33 + e34 + e35

    e39 = e28 + e37
    return {
        "D5_cost": e5,
        "D6_cost": e6,
        "D7_cost": e7,
        "D8_cost": e8,
        "D9_cost": e9,
        "D10_cost": e10,
        "D11_cost": e11,
        "D12_cost": e12,
        "D13_cost": e13,
        "E14": e14,
        "D16_cost": e16,
        "E18": e18,
        "D19_cost": e19,
        "D20_cost": e20,
        "D21_cost": e21,
        "D22_cost": e22,
        "E23": e23,
        "D26_cost": e26,
        "E28": e28,
        "D32_cost": e32,
        "D33_cost": e33,
        "D34_cost": e34,
        "D35_cost": e35,
        "E37": e37,
        "E39": e39,
        "direct_balance": e5 - e14,
        "fixed_balance": e16 - e23,
        "general_balance": e32 - e37,
        "target_residual": e39 - target_tcp,
    }


def solve_with_trace(
    specs: list[RowSpec],
    initial_values: dict[str, float],
    residual_fn: Callable[[np.ndarray], np.ndarray],
    max_evals: int,
) -> tuple[dict[str, float], list[float], list[dict[str, float]], dict[str, object]]:
    x0 = defaults_vector(specs, initial_values)
    lower, upper = build_bounds(specs)
    initial_residual = residual_fn(x0)
    trace: list[float] = [float(np.linalg.norm(initial_residual[: min(4, initial_residual.shape[0])]))]
    path: list[dict[str, float]] = [vector_to_values(specs, x0)]

    def wrapped_residual(x: np.ndarray) -> np.ndarray:
        append_path_snapshot(path, specs, x)
        residual = residual_fn(x)
        norm = float(np.linalg.norm(residual[: min(4, residual.shape[0])]))
        trace.append(norm)
        return residual

    result = least_squares(
        wrapped_residual,
        x0,
        bounds=(lower, upper),
        max_nfev=max_evals,
    )
    append_path_snapshot(path, specs, result.x)
    solved = vector_to_values(specs, result.x)
    meta = {
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "cost": float(result.cost),
    }
    return solved, trace, path, meta


def solve_with_penalty_trace(
    specs: list[RowSpec],
    initial_values: dict[str, float],
    residual_fn: Callable[[np.ndarray], np.ndarray],
    max_iters: int,
) -> tuple[dict[str, float], list[float], list[dict[str, float]], dict[str, object]]:
    x0 = defaults_vector(specs, initial_values)
    lower, upper = build_bounds(specs)
    bounds = list(zip(lower.tolist(), upper.tolist()))
    initial_residual = residual_fn(x0)
    trace: list[float] = [float(np.linalg.norm(initial_residual[: min(4, initial_residual.shape[0])]))]
    path: list[dict[str, float]] = [vector_to_values(specs, x0)]

    def objective(x: np.ndarray) -> float:
        append_path_snapshot(path, specs, x)
        residual = residual_fn(x)
        norm = float(np.linalg.norm(residual[: min(4, residual.shape[0])]))
        trace.append(norm)
        return float(np.dot(residual, residual))

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        options={"maxiter": max_iters, "ftol": 1e-12},
    )
    append_path_snapshot(path, specs, result.x)
    solved = vector_to_values(specs, result.x)
    meta = {
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(getattr(result, "nfev", 0)),
        "cost": float(result.fun),
    }
    return solved, trace, path, meta


def solve_capital(
    current_values: dict[str, float],
    target_tci: float,
    max_evals: int,
    method: str,
) -> tuple[dict[str, float], list[float], list[dict[str, float]], dict[str, object]]:
    specs = CAPITAL_SPECS
    anchor = defaults_vector(specs, CAPITAL_DEFAULTS)
    lower, upper = build_bounds(specs)
    scale = np.maximum(upper - lower, 1e-6)

    def residual_fn(x: np.ndarray) -> np.ndarray:
        values = {spec.cell: float(value) for spec, value in zip(specs, x)}
        calc = capital_calculator(values, target_tci)
        residuals = [
            calc["target_residual"] / max(target_tci, 1.0),
            calc["fci_balance"] / max(target_tci, 1.0),
        ]
        regularization = 0.001 * (x - anchor) / scale
        return np.concatenate([np.array(residuals, dtype=float), regularization])

    if method == "SLSQP Penalty":
        return solve_with_penalty_trace(specs, current_values, residual_fn, max_evals)
    return solve_with_trace(specs, current_values, residual_fn, max_evals)


def solve_product(
    current_values: dict[str, float],
    target_tcp: float,
    rent_base: float,
    depreciation_cost: float,
    max_evals: int,
    method: str,
) -> tuple[dict[str, float], list[float], list[dict[str, float]], dict[str, object]]:
    specs = PRODUCT_SPECS
    anchor = defaults_vector(specs, PRODUCT_DEFAULTS)
    lower, upper = build_bounds(specs)
    scale = np.maximum(upper - lower, 1e-6)

    def residual_fn(x: np.ndarray) -> np.ndarray:
        values = {spec.cell: float(value) for spec, value in zip(specs, x)}
        calc = product_calculator(values, target_tcp, rent_base, depreciation_cost)
        residuals = [
            calc["target_residual"] / max(target_tcp, 1.0),
            calc["direct_balance"] / max(target_tcp, 1.0),
            calc["fixed_balance"] / max(target_tcp, 1.0),
            calc["general_balance"] / max(target_tcp, 1.0),
        ]
        regularization = 0.001 * (x - anchor) / scale
        return np.concatenate([np.array(residuals, dtype=float), regularization])

    if method == "SLSQP Penalty":
        return solve_with_penalty_trace(specs, current_values, residual_fn, max_evals)
    return solve_with_trace(specs, current_values, residual_fn, max_evals)


def init_state() -> None:
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Chemical Plant Design Economics"
    if "solver_method" not in st.session_state:
        st.session_state["solver_method"] = "Least Squares"
    reset_capital_path = "capital_target" in st.session_state and float(st.session_state["capital_target"]) < 50000.0
    reset_product_path = "product_target" in st.session_state and float(st.session_state["product_target"]) < 50000.0
    if "capital_target" not in st.session_state or reset_capital_path:
        st.session_state["capital_target"] = ASSIGNMENT_TARGET
    if "product_target" not in st.session_state or reset_product_path:
        st.session_state["product_target"] = ASSIGNMENT_TARGET
    if reset_capital_path:
        for key in ("capital_values", "capital_trace", "capital_path", "capital_meta"):
            st.session_state.pop(key, None)
    if reset_product_path:
        for key in ("product_values", "product_trace", "product_path", "product_meta"):
            st.session_state.pop(key, None)
    if "product_rent_base" not in st.session_state:
        st.session_state["product_rent_base"] = 0.0
    if "product_depreciation" not in st.session_state:
        st.session_state["product_depreciation"] = 0.0
    if "capital_values" not in st.session_state:
        capital_random = randomize_values(CAPITAL_SPECS)
        _, capital_trace, capital_path, capital_meta = solve_capital(
            capital_random,
            st.session_state["capital_target"],
            FIXED_SOLVER_EVALS,
            st.session_state["solver_method"],
        )
        st.session_state["capital_values"] = capital_random
        st.session_state["capital_trace"] = capital_trace
        st.session_state["capital_path"] = capital_path
        st.session_state["capital_meta"] = {
            **capital_meta,
            "message": "Randomized assumptions | preview path ready",
        }
    if "product_values" not in st.session_state:
        product_random = randomize_values(PRODUCT_SPECS)
        _, product_trace, product_path, product_meta = solve_product(
            product_random,
            st.session_state["product_target"],
            st.session_state["product_rent_base"],
            st.session_state["product_depreciation"],
            FIXED_SOLVER_EVALS,
            st.session_state["solver_method"],
        )
        st.session_state["product_values"] = product_random
        st.session_state["product_trace"] = product_trace
        st.session_state["product_path"] = product_path
        st.session_state["product_meta"] = {
            **product_meta,
            "message": "Randomized assumptions | preview path ready",
        }
    if "capital_nonce" not in st.session_state:
        st.session_state["capital_nonce"] = 0
    if "product_nonce" not in st.session_state:
        st.session_state["product_nonce"] = 0
    if "capital_iteration_view" not in st.session_state:
        st.session_state["capital_iteration_view"] = 0
    if "product_iteration_view" not in st.session_state:
        st.session_state["product_iteration_view"] = 0


def render_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_notebook_cell(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="notebook-cell">
            <h3>{title}</h3>
            <div class="mini-note">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_editor_df(
    specs: list[RowSpec],
    values: dict[str, float],
    cost_lookup: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for spec in specs:
        rows.append(
            {
                "Cell": spec.cell,
                "Section": spec.section,
                "Item": spec.item,
                "Lower": spec.lower,
                "Selected": values[spec.cell],
                "Upper": spec.upper,
                "Cost": cost_lookup.get(f"{spec.cell}_cost", np.nan),
                "Formula": spec.formula,
            }
        )
    return pd.DataFrame(rows)


def render_editor(
    title: str,
    specs: list[RowSpec],
    values: dict[str, float],
    cost_lookup: dict[str, float],
    editor_key: str,
    sheet_height: int = 520,
) -> dict[str, float]:
    st.markdown(f"#### {title}")
    df = make_editor_df(specs, values, cost_lookup)
    st.markdown('<div class="excel-frame">', unsafe_allow_html=True)
    edited = st.data_editor(
        df,
        hide_index=True,
        key=editor_key,
        disabled=["Section", "Cell", "Item", "Lower", "Upper", "Cost", "Formula"],
        column_config={
            "Section": st.column_config.TextColumn(width="medium"),
            "Cell": st.column_config.TextColumn(width="small"),
            "Item": st.column_config.TextColumn(width="large"),
            "Lower": st.column_config.NumberColumn(format="%.4f"),
            "Selected": st.column_config.NumberColumn(format="%.6f"),
            "Upper": st.column_config.NumberColumn(format="%.4f"),
            "Cost": st.column_config.NumberColumn(format="%.2f"),
            "Formula": st.column_config.TextColumn(width="large"),
        },
        height=sheet_height,
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    updated = values.copy()
    for row in edited.to_dict("records"):
        updated[str(row["Cell"])] = float(row["Selected"])
    return updated


def trace_chart(trace: list[float], title: str) -> go.Figure:
    if not trace:
        trace = [0.0]
    frame = pd.DataFrame(
        {
            "Evaluation": np.arange(1, len(trace) + 1, dtype=int),
            "Residual norm": trace,
        }
    )
    fig = px.line(frame, x="Evaluation", y="Residual norm", markers=False, title=title)
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Function evaluation",
        yaxis_title="Constraint residual",
    )
    return fig


def contribution_chart(contrib: dict[str, float], title: str) -> go.Figure:
    frame = pd.DataFrame({"Component": list(contrib.keys()), "Cost": list(contrib.values())})
    fig = px.bar(
        frame,
        x="Cost",
        y="Component",
        orientation="h",
        title=title,
        text="Cost",
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def render_iteration_panel(
    title: str,
    subtitle: str,
    display_index: int,
    has_unsaved_edits: bool,
    iteration_key: str,
    compact: bool = False,
) -> None:
    progress = display_index / PREVIEW_STEPS
    panel_class = "iteration-panel compact" if compact else "iteration-panel"
    body = "" if compact else f"<p>{html.escape(subtitle)}</p>"
    mode_label = "Manual edit mode" if has_unsaved_edits else "Path preview mode"
    panel_html = (
        f'<div class="{panel_class}">'
        f"<h4>{html.escape(title)}</h4>"
        f"{body}"
        '<div class="iteration-chip-row">'
        f'<span class="iteration-chip">Step {display_index} / {PREVIEW_STEPS}</span>'
        f'<span class="iteration-chip">{progress * 100:.1f}% along path</span>'
        f'<span class="iteration-chip">{html.escape(mode_label)}</span>'
        "</div>"
        "</div>"
    )
    st.markdown(panel_html, unsafe_allow_html=True)
    if compact:
        st.progress(progress)
    else:
        st.progress(progress, text=f"Iteration progress: step {display_index} of {PREVIEW_STEPS}")
    st.slider(
        "Preview iteration",
        min_value=0,
        max_value=PREVIEW_STEPS,
        step=1,
        key=iteration_key,
        disabled=has_unsaved_edits,
        label_visibility="collapsed" if compact else "visible",
    )


def path_index_from_preview_step(preview_step: int, path_length: int) -> int:
    if path_length <= 1:
        return 0
    bounded_step = min(PREVIEW_STEPS, max(0, int(preview_step)))
    return int(round((bounded_step / PREVIEW_STEPS) * (path_length - 1)))


def fmt_rate_bound(value: float) -> str:
    if abs(value) < 1e-12:
        return "0"
    pct = value * 100.0
    if abs(pct - round(pct)) < 1e-9:
        return f"{int(round(pct))}%"
    return f"{pct:g}%"


def fmt_selected(value: float) -> str:
    return f"{value:.9f}".rstrip("0").rstrip(".")


def fmt_cost(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return str(int(rounded))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def column_label(index: int) -> str:
    label = ""
    current = index
    while current > 0:
        current, rem = divmod(current - 1, 26)
        label = chr(65 + rem) + label
    return label


def split_cell_ref(cell_ref: str) -> tuple[str, int]:
    match = re.fullmatch(r"([A-Z]+)(\d+)", cell_ref)
    if not match:
        return "", 0
    return match.group(1), int(match.group(2))


def format_formula_display(value: object) -> str:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return fmt_cost(float(value))
    return str(value)


def sheet_cell(
    text: str = "",
    cls: str = "text",
    colspan: int = 1,
) -> dict[str, object]:
    return {"text": text, "class": cls, "colspan": colspan}


def blank_row(count: int) -> list[dict[str, object]]:
    return [sheet_cell("", "text") for _ in range(count)]


def render_excel_sheet(
    rows: list[list[dict[str, object]]],
    widths: list[str],
    page_key: str,
    active_cell: str,
    clickable_map: dict[str, tuple[object, object]],
    zoom_pct: int = 100,
    font_family: str = "Arial",
    font_size: int = 10,
) -> None:
    scale = zoom_pct / 100.0
    scaled_widths: list[str] = []
    for w in widths:
        if w.endswith("px"):
            scaled_widths.append(f"{int(round(float(w[:-2]) * scale))}px")
        else:
            scaled_widths.append(w)
    default_blank_width = f"{int(round(120 * scale))}px"
    min_total_cols = max(len(scaled_widths), 14)
    while len(scaled_widths) < min_total_cols:
        scaled_widths.append(default_blank_width)
    total_cols = len(scaled_widths)
    min_total_rows = max(len(rows), 36)
    row_height = max(24, int(round(30 * scale)))
    header_width = f"{max(48, int(round(56 * scale)))}px"
    full_widths = [header_width] + scaled_widths
    colgroup = "".join([f'<col style="width:{w};">' for w in full_widths])
    body = []
    active_col, active_row = split_cell_ref(active_cell)
    header_cells = ['<tr><td class="sheet-corner" style="height:{0}px;"></td>'.format(row_height)]
    for idx in range(1, total_cols + 1):
        col_name = column_label(idx)
        header_cls = "sheet-header"
        if col_name == active_col:
            header_cls += " sheet-header-active"
        header_cells.append(
            f'<td id="{page_key}_col_{col_name}" class="{header_cls}" style="height:{row_height}px;font-family:{html.escape(font_family)};font-size:{font_size}px;">{col_name}</td>'
        )
    header_cells.append("</tr>")
    body.append("".join(header_cells))
    for row_index, row in enumerate(rows, start=1):
        row_header_cls = "sheet-header"
        if row_index == active_row:
            row_header_cls += " sheet-header-active"
        parts = [
            "<tr>",
            f'<td id="{page_key}_row_{row_index}" class="{row_header_cls}" style="height:{row_height}px;font-family:{html.escape(font_family)};font-size:{font_size}px;">{row_index}</td>',
        ]
        col_index = 1
        for cell in row:
            raw_text = str(cell.get("text", ""))
            text = html.escape(raw_text)
            cls = cell.get("class", "text")
            colspan = int(cell.get("colspan", 1))
            address = f"{column_label(col_index)}{row_index}"
            cell_cls = str(cls)
            if address == active_cell:
                cell_cls += " cell-active"
            display_text = text if raw_text.strip() else "&nbsp;"
            parts.append(
                f'<td id="{page_key}_cell_{address}" class="{cell_cls}" data-cell="{address}" colspan="{colspan}" style="height:{row_height}px;font-family:{html.escape(font_family)};font-size:{font_size}px;">{display_text}</td>'
            )
            col_index += colspan
        while col_index <= total_cols:
            address = f"{column_label(col_index)}{row_index}"
            cell_cls = "text"
            if address == active_cell:
                cell_cls += " cell-active"
            parts.append(
                f'<td id="{page_key}_cell_{address}" class="{cell_cls}" data-cell="{address}" style="height:{row_height}px;font-family:{html.escape(font_family)};font-size:{font_size}px;">&nbsp;</td>'
            )
            col_index += 1
        parts.append("</tr>")
        body.append("".join(parts))
    for row_index in range(len(rows) + 1, min_total_rows + 1):
        row_header_cls = "sheet-header"
        if row_index == active_row:
            row_header_cls += " sheet-header-active"
        parts = [
            "<tr>",
            f'<td id="{page_key}_row_{row_index}" class="{row_header_cls}" style="height:{row_height}px;font-family:{html.escape(font_family)};font-size:{font_size}px;">{row_index}</td>',
        ]
        for col_index in range(1, total_cols + 1):
            address = f"{column_label(col_index)}{row_index}"
            cell_cls = "text"
            if address == active_cell:
                cell_cls += " cell-active"
            parts.append(
                f'<td id="{page_key}_cell_{address}" class="{cell_cls}" data-cell="{address}" style="height:{row_height}px;font-family:{html.escape(font_family)};font-size:{font_size}px;">&nbsp;</td>'
            )
        parts.append("</tr>")
        body.append("".join(parts))
    formula_payload = {
        key: [format_formula_display(value[0]), format_formula_display(value[1])]
        for key, value in clickable_map.items()
    }
    initial_formula, initial_value = formula_payload.get(active_cell, ["", ""])
    payload_json = json.dumps(formula_payload, separators=(",", ":"))
    table_html = f"""
    <!doctype html>
    <html>
    <head>
    <meta charset="utf-8" />
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            background: transparent;
            font-family: Arial, sans-serif;
            overflow: auto;
        }}
        .gs-toolbar {{
            background: #eef2f7;
            border: 1px solid #d9e0ea;
            border-radius: 14px 14px 0 0;
            padding: 10px 14px;
            color: #475467;
        }}
        .toolbar-row {{
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }}
        .gs-chip {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 25px;
            height: 26px;
            border: 1px solid #d1d9e6;
            border-radius: 7px;
            background: #ffffff;
            color: #475467;
            font-size: 12px;
            padding: 0 7px;
        }}
        .gs-chip.wide {{
            min-width: 65px;
        }}
        .gs-divider {{
            width: 1px;
            height: 20px;
            background: #d1d9e6;
            margin: 0 1px;
        }}
        .formula-shell {{
            display: grid;
            grid-template-columns: 105px 1fr 128px;
            gap: 10px;
            align-items: center;
            background: #f8fafc;
            border-left: 1px solid #d9e0ea;
            border-right: 1px solid #d9e0ea;
            border-bottom: 1px solid #d9e0ea;
            padding: 8px 12px;
        }}
        .formula-label {{
            color: #4b5563;
            font-size: 11px;
            margin-bottom: 3px;
        }}
        .formula-pill {{
            border: 1px solid #d1d9e6;
            background: #ffffff;
            color: #111827;
            padding: 6px 9px;
            border-radius: 8px;
            font-size: 12px;
            min-height: 18px;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .sheet-shell {{
            background-color: #ffffff;
            background-image:
                linear-gradient(135deg, transparent 0 58%, #a9b6ca 59% 64%, transparent 65%),
                linear-gradient(135deg, transparent 0 70%, #a9b6ca 71% 76%, transparent 77%);
            background-repeat: no-repeat;
            background-position: right 8px bottom 8px, right 14px bottom 8px;
            background-size: 18px 18px, 18px 18px;
            border: 1px solid #cfd6e0;
            border-top: 0;
            border-radius: 0 0 8px 8px;
            padding: 0;
            overflow: auto;
            resize: both;
            width: 96%;
            height: 575px;
            min-width: 420px;
            min-height: 300px;
            max-width: 1600px;
            max-height: 900px;
            scrollbar-gutter: stable both-edges;
            box-shadow: 0 0 0 1px rgba(255,255,255,0.45);
        }}
        .sheet-shell:hover {{
            border-color: #8fb4ff;
        }}
        .sheet-shell::-webkit-scrollbar {{
            width: 14px;
            height: 14px;
        }}
        .sheet-shell::-webkit-scrollbar-track {{
            background: #eef2f7;
            border-radius: 999px;
        }}
        .sheet-shell::-webkit-scrollbar-thumb {{
            background: #c0cada;
            border-radius: 999px;
            border: 2px solid #eef2f7;
        }}
        .sheet-shell::-webkit-scrollbar-thumb:hover {{
            background: #a9b6ca;
        }}
        table.excel-sheet {{
            border-collapse: collapse;
            width: max-content;
            min-width: 100%;
            background: #ffffff;
        }}
        table.excel-sheet td {{
            border: 1px solid #cfd6e0;
            min-width: 84px;
            height: 30px;
            padding: 3px 8px;
            font-size: 13px;
            line-height: 1.15;
            color: #111827;
            vertical-align: middle;
            background: #ffffff;
            white-space: nowrap;
            cursor: pointer;
            user-select: none;
        }}
        table.excel-sheet td.sheet-header,
        table.excel-sheet td.sheet-corner {{
            cursor: default;
            user-select: none;
        }}
        table.excel-sheet td.sheet-header {{
            background: #f1f5f9;
            color: #475467;
            font-weight: 600;
            text-align: center;
            min-width: 52px;
        }}
        table.excel-sheet td.sheet-header-active {{
            background: #dbe7ff;
            color: #173a8a;
            font-weight: 700;
        }}
        table.excel-sheet td.sheet-corner {{
            background: #e8edf5;
            min-width: 52px;
        }}
        table.excel-sheet td.cell-active {{
            position: relative;
            background: #e8f0fe !important;
            box-shadow: inset 0 0 0 2px #3b82f6;
        }}
        table.excel-sheet td.text {{
            text-align: left;
        }}
        table.excel-sheet td.num {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
        table.excel-sheet td.center {{
            text-align: center;
        }}
        table.excel-sheet td.title-yellow,
        table.excel-sheet td.section-yellow {{
            background: #fff45c;
            font-weight: 700;
        }}
        table.excel-sheet td.head-green {{
            background: #afc3ae;
            font-weight: 600;
        }}
        table.excel-sheet td.head-peach {{
            background: #e3c6ad;
            font-weight: 600;
        }}
        table.excel-sheet td.total-green {{
            background: #bfd3bf;
            font-weight: 600;
        }}
        table.excel-sheet td.soft {{
            color: #4b5563;
        }}
    </style>
    </head>
    <body>
        <div class="gs-toolbar">
            <div class="toolbar-row">
                <div class="gs-chip">Search</div><div class="gs-chip">Undo</div><div class="gs-chip">Redo</div>
                <div class="gs-chip">Print</div><div class="gs-divider"></div><div class="gs-chip wide">{zoom_pct}%</div>
                <div class="gs-divider"></div><div class="gs-chip">$</div><div class="gs-chip">%</div>
                <div class="gs-chip">.0</div><div class="gs-chip">.00</div><div class="gs-chip">123</div>
                <div class="gs-divider"></div><div class="gs-chip wide">{html.escape(font_family)}</div>
                <div class="gs-chip">{font_size}</div><div class="gs-chip">B</div><div class="gs-chip">I</div>
                <div class="gs-chip">A</div><div class="gs-chip">Fill</div><div class="gs-chip">Grid</div>
                <div class="gs-chip">Align</div><div class="gs-chip">Σ</div>
            </div>
        </div>
        <div class="formula-shell">
            <div><div class="formula-label">Name box</div><div id="name-box" class="formula-pill">{html.escape(active_cell)}</div></div>
            <div><div class="formula-label">Formula bar</div><div id="formula-box" class="formula-pill">fx&nbsp;&nbsp;{html.escape(initial_formula)}</div></div>
            <div><div class="formula-label">Displayed value</div><div id="value-box" class="formula-pill">{html.escape(initial_value)}</div></div>
        </div>
        <div id="{page_key}_sheet" class="sheet-shell">
            <table class="excel-sheet">{colgroup}<tbody>{"".join(body)}</tbody></table>
        </div>
        <script>
            const formulaMap = {payload_json};
            const pageKey = {json.dumps(page_key)};
            const shell = document.getElementById(pageKey + "_sheet");
            let activeCell = localStorage.getItem(pageKey + "_activeCell") || {json.dumps(active_cell)};
            function storageKey(name) {{
                return pageKey + "_" + name;
            }}
            function restoreShellState() {{
                const savedWidth = Number(localStorage.getItem(storageKey("sheetWidth")));
                const savedHeight = Number(localStorage.getItem(storageKey("sheetHeight")));
                const savedLeft = Number(localStorage.getItem(storageKey("scrollLeft")));
                const savedTop = Number(localStorage.getItem(storageKey("scrollTop")));
                if (Number.isFinite(savedWidth) && savedWidth >= 420) {{
                    shell.style.width = savedWidth + "px";
                }}
                if (Number.isFinite(savedHeight) && savedHeight >= 300) {{
                    shell.style.height = savedHeight + "px";
                }}
                requestAnimationFrame(() => {{
                    if (Number.isFinite(savedLeft)) shell.scrollLeft = savedLeft;
                    if (Number.isFinite(savedTop)) shell.scrollTop = savedTop;
                }});
            }}
            function persistShellSize() {{
                const rect = shell.getBoundingClientRect();
                if (rect.width >= 420) localStorage.setItem(storageKey("sheetWidth"), Math.round(rect.width));
                if (rect.height >= 300) localStorage.setItem(storageKey("sheetHeight"), Math.round(rect.height));
            }}
            function splitCell(ref) {{
                const match = /^([A-Z]+)(\\d+)$/.exec(ref || "");
                return match ? [match[1], match[2]] : ["", ""];
            }}
            function clearActive() {{
                document.querySelectorAll(".cell-active").forEach(el => el.classList.remove("cell-active"));
                document.querySelectorAll(".sheet-header-active").forEach(el => el.classList.remove("sheet-header-active"));
            }}
            function selectCell(ref) {{
                activeCell = ref;
                localStorage.setItem(pageKey + "_activeCell", ref);
                const pair = formulaMap[ref] || ["", ""];
                document.getElementById("name-box").textContent = ref;
                document.getElementById("formula-box").textContent = "fx  " + pair[0];
                document.getElementById("value-box").textContent = pair[1];
                clearActive();
                const cell = document.getElementById(pageKey + "_cell_" + ref);
                if (cell) cell.classList.add("cell-active");
                const [col, row] = splitCell(ref);
                const colHeader = document.getElementById(pageKey + "_col_" + col);
                const rowHeader = document.getElementById(pageKey + "_row_" + row);
                if (colHeader) colHeader.classList.add("sheet-header-active");
                if (rowHeader) rowHeader.classList.add("sheet-header-active");
            }}
            document.querySelectorAll("td[data-cell]").forEach(cell => {{
                cell.addEventListener("click", () => selectCell(cell.dataset.cell));
            }});
            restoreShellState();
            const resizeObserver = new ResizeObserver(() => persistShellSize());
            resizeObserver.observe(shell);
            shell.addEventListener("scroll", () => {{
                localStorage.setItem(storageKey("scrollLeft"), Math.round(shell.scrollLeft));
                localStorage.setItem(storageKey("scrollTop"), Math.round(shell.scrollTop));
            }}, {{ passive: true }});
            selectCell(activeCell);
        </script>
    </body>
    </html>
    """
    components.html(table_html, height=780, scrolling=True)


def render_fake_toolbar() -> None:
    st.markdown(
        """
        <div class="gs-toolbar">
            <div class="row">
                <div class="gs-chip">Search</div>
                <div class="gs-chip">Undo</div>
                <div class="gs-chip">Redo</div>
                <div class="gs-chip">Print</div>
                <div class="gs-divider"></div>
                <div class="gs-chip wide">100%</div>
                <div class="gs-divider"></div>
                <div class="gs-chip">$</div>
                <div class="gs-chip">%</div>
                <div class="gs-chip">.0</div>
                <div class="gs-chip">.00</div>
                <div class="gs-chip">123</div>
                <div class="gs-divider"></div>
                <div class="gs-chip wide">Default</div>
                <div class="gs-chip">10</div>
                <div class="gs-chip">B</div>
                <div class="gs-chip">I</div>
                <div class="gs-chip">A</div>
                <div class="gs-chip">Fill</div>
                <div class="gs-chip">Grid</div>
                <div class="gs-chip">Merge</div>
                <div class="gs-chip">Align</div>
                <div class="gs-chip">Filter</div>
                <div class="gs-chip">Σ</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_capital_formula_overrides(
    values: dict[str, float],
    target: float,
    calc: dict[str, float],
) -> dict[str, tuple[str, float]]:
    overrides: dict[str, tuple[str, float]] = {
        "I1": ("Total Capital Investment [TCI]", target),
        "D5": ("Selected working capital percentage", values["D4"]),
        "E5": ("=I1*D5", calc["D4_cost"]),
        "D6": ("Selected fixed-capital-investment percentage", values["D5"]),
        "E6": ("=I1*D6", calc["D5_cost"]),
        "E10": ("=E6*D10", calc["D9_cost"]),
        "E11": ("=E10*D11", calc["D10_cost"]),
        "E12": ("=E10*D12", calc["D11_cost"]),
        "E13": ("=E10*D13", calc["D12_cost"]),
        "E14": ("=E10*D14", calc["D13_cost"]),
        "E15": ("=E10*D15", calc["D14_cost"]),
        "E16": ("=E10*D16", calc["D15_cost"]),
        "E17": ("=E10*D17", calc["D16_cost"]),
        "E19": ("=SUM(E10:E17)", calc["E18"]),
        "E23": ("=E19*D23", calc["D23_cost"]),
        "E24": ("=I1*D24", calc["D24_cost"]),
        "E25": ("=I1*D25", calc["D25_cost"]),
        "E26": ("=I1*D26", calc["D26_cost"]),
        "E28": ("=SUM(E23:E26)", calc["E28"]),
        "E29": ("=E19+E28", calc["E30"]),
        "E30": ("=E29+E5", calc["E31"]),
    }
    direct_keys = ["D9", "D10", "D11", "D12", "D13", "D14", "D15", "D16"]
    for row, key in zip(range(10, 18), direct_keys):
        overrides[f"D{row}"] = (f"Selected percentage for {key}", values[key])
    indirect_keys = ["D23", "D24", "D25", "D26"]
    for row, key in zip(range(23, 27), indirect_keys):
        overrides[f"D{row}"] = (f"Selected percentage for {key}", values[key])
    return overrides


def build_product_formula_overrides(
    values: dict[str, float],
    target: float,
    calc: dict[str, float],
    rent_base: float,
    depreciation_cost: float,
) -> dict[str, tuple[str, float]]:
    overrides: dict[str, tuple[str, float]] = {
        "J1": ("Total Product Costs [TCP]", target),
        "E5": ("=J1*D5", calc["D6_cost"]),
        "E6": ("=J1*D6", calc["D7_cost"]),
        "E7": ("=E6*D7", calc["D8_cost"]),
        "E8": ("=J1*D8", calc["D9_cost"]),
        "E9": ("=E13*D9", calc["D10_cost"]),
        "E10": ("=E13*D10", calc["D11_cost"]),
        "E11": ("=E6*D11", calc["D12_cost"]),
        "E12": ("=J1*D12", calc["D13_cost"]),
        "E13": ("=SUM(E5:E12)", calc["E14"]),
        "E15": ("=J1*D15", calc["D16_cost"]),
        "E16": ("Depreciation cost assumption", depreciation_cost),
        "E17": ("=E13*D17", calc["D19_cost"]),
        "E18": ("=E13*D18", calc["D20_cost"]),
        "E19": (f"={fmt_cost(rent_base)}*D19", calc["D21_cost"]),
        "E20": ("=J1*D20", calc["D22_cost"]),
        "E21": ("=SUM(E16:E20)", calc["E23"]),
        "E23": ("=J1*D23", calc["D26_cost"]),
        "E25": ("=E13+E21+E23", calc["E28"]),
        "E29": ("=J1*D29", calc["D33_cost"]),
        "E30": ("=J1*D30", calc["D34_cost"]),
        "E31": ("=J1*D31", calc["D35_cost"]),
        "E33": ("=SUM(E29:E31)", calc["E37"]),
        "E35": ("=E25+E33", calc["E39"]),
    }
    direct_keys = ["D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13"]
    for row, key in zip(range(5, 13), direct_keys):
        overrides[f"D{row}"] = (f"Selected percentage for {key}", values[key])
    for address, key in {
        "D13": "D5",
        "D15": "D16",
        "D17": "D19",
        "D18": "D20",
        "D19": "D21",
        "D20": "D22",
        "D23": "D26",
        "D29": "D33",
        "D30": "D34",
        "D31": "D35",
    }.items():
        overrides[address] = (f"Selected percentage for {key}", values[key])
    return overrides


def render_formula_shell(
    active_cell: str,
    formula_text: object,
    formula_value: object,
) -> None:
    st.markdown(
        f"""
        <div class="formula-shell">
            <div>
                <div class="formula-label">Name box</div>
                <div class="formula-pill">{html.escape(active_cell)}</div>
            </div>
            <div>
                <div class="formula-label">Formula bar</div>
                <div class="formula-pill">fx&nbsp;&nbsp;{html.escape(format_formula_display(formula_text))}</div>
            </div>
            <div>
                <div class="formula-label">Displayed value</div>
                <div class="formula-pill">{html.escape(format_formula_display(formula_value))}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_grid_formula_map(
    rows: list[list[dict[str, object]]],
    overrides: dict[str, tuple[object, object]] | None = None,
    min_total_cols: int = 14,
    min_total_rows: int = 36,
) -> dict[str, tuple[object, object]]:
    formula_map: dict[str, tuple[object, object]] = {}
    override_map = overrides or {}
    total_rows = max(len(rows), min_total_rows)
    for row_index in range(1, total_rows + 1):
        row = rows[row_index - 1] if row_index <= len(rows) else []
        col_index = 1
        for cell in row:
            colspan = int(cell.get("colspan", 1))
            address = f"{column_label(col_index)}{row_index}"
            text = str(cell.get("text", ""))
            if address in override_map:
                formula_map[address] = override_map[address]
            elif text.strip():
                formula_map[address] = (text, text)
            else:
                formula_map[address] = ("", "")
            col_index += colspan
        while col_index <= min_total_cols:
            address = f"{column_label(col_index)}{row_index}"
            formula_map[address] = override_map.get(address, ("", ""))
            col_index += 1
    for address, formula_pair in override_map.items():
        formula_map[address] = formula_pair
    return formula_map


def render_sheet_chrome(
    page_key: str,
    formula_map: dict[str, tuple[object, object]],
) -> tuple[int, str, int, str]:
    active_key = f"{page_key}_active_cell"
    if active_key not in st.session_state or st.session_state[active_key] not in formula_map:
        st.session_state[active_key] = "A1"
        if "A1" not in formula_map:
            formula_map["A1"] = ("", "")
    chrome_cols = st.columns([1.1, 1.1, 0.8, 2.8], gap="small")
    with chrome_cols[0]:
        zoom = st.selectbox(
            "Zoom",
            [35, 40, 50, 60, 70, 80, 90, 100, 110, 125, 150],
            index=7,
            key=f"{page_key}_zoom",
        )
    with chrome_cols[1]:
        font_family = st.selectbox(
            "Font",
            ["Arial", "Calibri", "Times New Roman", "Georgia", "Verdana"],
            index=0,
            key=f"{page_key}_font_family",
        )
    with chrome_cols[2]:
        font_size = st.selectbox(
            "Font Size",
            [9, 10, 11, 12, 13, 14],
            index=1,
            key=f"{page_key}_font_size",
        )
    with chrome_cols[3]:
        st.caption("Click any cell inside the sheet to update the name box, formula bar, and highlighted row/column instantly.")
    active_cell = str(st.session_state[active_key])
    return int(zoom), str(font_family), int(font_size), active_cell


def build_capital_sheet_rows(values: dict[str, float], target: float, calc: dict[str, float]) -> list[list[dict[str, object]]]:
    rows: list[list[dict[str, object]]] = []
    rows.append(
        [
            sheet_cell("", "text", 6),
            sheet_cell("Total Capital Investment [TCI]", "title-yellow text", 2),
            sheet_cell(fmt_cost(target), "num"),
            sheet_cell("(supposed value)", "text soft"),
        ]
    )
    rows.append(blank_row(10))
    rows.append(
        [
            sheet_cell("", "text"),
            sheet_cell("Rate(%)", "head-green center", 3),
            sheet_cell("Cost", "head-green center"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
        ]
    )
    rows.append(
        [
            sheet_cell("", "text"),
            sheet_cell("Lower Limit", "head-peach center"),
            sheet_cell("Upper Limit", "head-peach center"),
            sheet_cell("Selected %", "head-peach center"),
            sheet_cell("", "head-peach center"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
        ]
    )
    rows.append(
        [
            sheet_cell("Working Capital", "section-yellow text"),
            sheet_cell(fmt_rate_bound(0.10), "num"),
            sheet_cell(fmt_rate_bound(0.20), "num"),
            sheet_cell(fmt_selected(values["D4"]), "num"),
            sheet_cell(fmt_cost(calc["D4_cost"]), "num"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
        ]
    )
    rows.append(
        [
            sheet_cell("Fixed Capital Investment", "section-yellow text"),
            sheet_cell("0", "num"),
            sheet_cell(fmt_rate_bound(0.80), "num"),
            sheet_cell(fmt_selected(values["D5"]), "num"),
            sheet_cell(fmt_cost(calc["D5_cost"]), "num"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
        ]
    )
    rows.append(blank_row(10))
    rows.append(
        [
            sheet_cell("Direct Costs", "section-yellow text"),
            sheet_cell("Rate(%)", "head-green center", 3),
            sheet_cell("Cost", "head-green center"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
        ]
    )
    rows.append(
        [
            sheet_cell("", "text"),
            sheet_cell("Lower Limit", "head-peach center"),
            sheet_cell("Upper Limit", "head-peach center"),
            sheet_cell("Selected %", "head-peach center"),
            sheet_cell("", "head-peach center"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
        ]
    )
    direct_items = [
        ("Purchased Equipment Cost [PEC] (15-40% of FCI)", "D9", "D9_cost"),
        ("Installation Costs (25-55% of PEC)", "D10", "D10_cost"),
        ("Instrumentation Costs (8-50% of PEC)", "D11", "D11_cost"),
        ("Piping Cost (10-80% of PEC)", "D12", "D12_cost"),
        ("Electricity Costs (10-40% of PEC)", "D13", "D13_cost"),
        ("Buildings, process, and auxiliary (10-70% of PEC)", "D14", "D14_cost"),
        ("Service facilities and yard improvements (40-100% of PEC)", "D15", "D15_cost"),
        ("Land (4-8% of PEC)", "D16", "D16_cost"),
    ]
    for label, key, cost_key in direct_items:
        spec = next(spec for spec in CAPITAL_SPECS if spec.cell == key)
        rows.append(
            [
                sheet_cell(label, "text"),
                sheet_cell(fmt_rate_bound(spec.lower), "num"),
                sheet_cell(fmt_rate_bound(spec.upper), "num"),
                sheet_cell(fmt_selected(values[key]), "num"),
                sheet_cell(fmt_cost(calc[cost_key]), "num"),
                sheet_cell("", "text"),
                sheet_cell("", "text"),
                sheet_cell("", "text"),
                sheet_cell("", "text"),
                sheet_cell("", "text"),
            ]
        )
    rows.append(blank_row(10))
    rows.append(
        [
            sheet_cell("Total Direct Costs", "total-green text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell(fmt_cost(calc["E18"]), "total-green num"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
        ]
    )
    rows.append(blank_row(10))
    rows.append(
        [
            sheet_cell("Indirect Costs", "section-yellow text"),
            sheet_cell("Rate(%)", "head-green center", 3),
            sheet_cell("Cost", "head-green center"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
        ]
    )
    rows.append(
        [
            sheet_cell("", "text"),
            sheet_cell("Lower Limit", "head-peach center"),
            sheet_cell("Upper Limit", "head-peach center"),
            sheet_cell("Selected %", "head-peach center"),
            sheet_cell("", "head-peach center"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
        ]
    )
    indirect_items = [
        ("Engineering and Supervision (5-30% of Direct Costs)", "D23", "D23_cost"),
        ("Legal Expenses (1-3% of FCI)", "D24", "D24_cost"),
        ("Construction Expense and Contractor's fee (10-20% of FCI)", "D25", "D25_cost"),
        ("Contingency (5-15% of FCI)", "D26", "D26_cost"),
    ]
    for label, key, cost_key in indirect_items:
        spec = next(spec for spec in CAPITAL_SPECS if spec.cell == key)
        rows.append(
            [
                sheet_cell(label, "text"),
                sheet_cell(fmt_rate_bound(spec.lower), "num"),
                sheet_cell(fmt_rate_bound(spec.upper), "num"),
                sheet_cell(fmt_selected(values[key]), "num"),
                sheet_cell(fmt_cost(calc[cost_key]), "num"),
                sheet_cell("", "text"),
                sheet_cell("", "text"),
                sheet_cell("", "text"),
                sheet_cell("", "text"),
                sheet_cell("", "text"),
            ]
        )
    rows.append(blank_row(10))
    rows.append(
        [
            sheet_cell("Total Indirect Costs", "total-green text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell(fmt_cost(calc["E28"]), "total-green num"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
        ]
    )
    rows.append(
        [
            sheet_cell("Total FCI", "total-green text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell(fmt_cost(calc["E30"]), "total-green num"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
        ]
    )
    rows.append(
        [
            sheet_cell("Total Capital Investment", "total-green text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell(fmt_cost(calc["E31"]), "total-green num"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
        ]
    )
    for _ in range(5):
        rows.append(blank_row(10))
    return rows


def build_product_sheet_rows(
    values: dict[str, float],
    target: float,
    calc: dict[str, float],
) -> list[list[dict[str, object]]]:
    rows: list[list[dict[str, object]]] = []
    rows.append(
        [
            sheet_cell("", "text", 7),
            sheet_cell("Total Product Costs [TCP]", "title-yellow text", 2),
            sheet_cell(fmt_cost(target), "num"),
            sheet_cell("(supposed value)", "text soft"),
        ]
    )
    rows.append(blank_row(11))
    rows.append(
        [
            sheet_cell("I. Manufacturing Costs", "section-yellow text"),
            sheet_cell("Rate(%)", "head-green center", 3),
            sheet_cell("Cost", "head-green center"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
        ]
    )
    rows.append(
        [
            sheet_cell("A. Direct Production Costs (about 66% of TCP)", "total-green text"),
            sheet_cell("Lower Limit", "head-peach center"),
            sheet_cell("Upper Limit", "head-peach center"),
            sheet_cell("Selected %", "head-peach center"),
            sheet_cell("", "head-peach center"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
        ]
    )
    rows.append(
        [
            sheet_cell("Raw Materials (10 - 80% of TCP)", "text"),
            sheet_cell(fmt_rate_bound(0.10), "num"),
            sheet_cell(fmt_rate_bound(0.80), "num"),
            sheet_cell(fmt_selected(values["D6"]), "num"),
            sheet_cell(fmt_cost(calc["D6_cost"]), "num"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
        ]
    )
    prod_detail_items = [
        ("Operating Labour (10-20% of TCP)", "D7", "D7_cost"),
        ("Direct Supervisory and clerical labour (10-20% of Operating Labour)", "D8", "D8_cost"),
        ("Utilities (10-20% of TCP)", "D9", "D9_cost"),
        ("Maintenance and Repairs (2-10% of FCI)", "D10", "D10_cost"),
        ("Operating Supplies (0.5-1% of FCI)", "D11", "D11_cost"),
        ("Laboratory Charges (10-20% of Operating Labour)", "D12", "D12_cost"),
        ("Patents and Royalties (0-6% of TCP)", "D13", "D13_cost"),
    ]
    for label, key, cost_key in prod_detail_items:
        spec = next(spec for spec in PRODUCT_SPECS if spec.cell == key)
        rows.append(
            [
                sheet_cell(label, "text"),
                sheet_cell(fmt_rate_bound(spec.lower), "num"),
                sheet_cell(fmt_rate_bound(spec.upper), "num"),
                sheet_cell(fmt_selected(values[key]), "num"),
                sheet_cell(fmt_cost(calc[cost_key]), "num"),
                sheet_cell("", "text"),
                sheet_cell("", "text"),
                sheet_cell("", "text"),
                sheet_cell("", "text"),
                sheet_cell("", "text"),
                sheet_cell("", "text"),
            ]
        )
    rows.append(
        [
            sheet_cell("Total Direct Production Costs", "total-green text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell(fmt_selected(values["D5"]), "num"),
            sheet_cell(fmt_cost(calc["E14"]), "total-green num"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
        ]
    )
    rows.append(blank_row(11))
    rows.append(
        [
            sheet_cell("B. Fixed Charges (10-20% of TCP)", "total-green text"),
            sheet_cell(fmt_rate_bound(0.10), "num"),
            sheet_cell(fmt_rate_bound(0.20), "num"),
            sheet_cell(fmt_selected(values["D16"]), "num"),
            sheet_cell(fmt_cost(calc["D16_cost"]), "total-green num"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
        ]
    )
    fixed_items = [
        ("Depreciation", None, "E18"),
        ("Local Taxes (1-4% of FCI)", "D19", "D19_cost"),
        ("Insurance (0.4-1% of FCI)", "D20", "D20_cost"),
        ("Rent (8-12% of value of rented land and buildings)", "D21", "D21_cost"),
        ("Financing (interest) (0-10% of TCP)", "D22", "D22_cost"),
    ]
    for label, key, cost_key in fixed_items:
        if key is None:
            rows.append(
                [
                    sheet_cell(label, "text"),
                    sheet_cell("0", "num"),
                    sheet_cell("0", "num"),
                    sheet_cell("0", "num"),
                    sheet_cell(fmt_cost(calc["E18"]), "num"),
                    sheet_cell("", "text"),
                    sheet_cell("", "text"),
                    sheet_cell("", "text"),
                    sheet_cell("", "text"),
                    sheet_cell("", "text"),
                    sheet_cell("", "text"),
                ]
            )
        else:
            spec = next(spec for spec in PRODUCT_SPECS if spec.cell == key)
            rows.append(
                [
                    sheet_cell(label, "text"),
                    sheet_cell(fmt_rate_bound(spec.lower), "num"),
                    sheet_cell(fmt_rate_bound(spec.upper), "num"),
                    sheet_cell(fmt_selected(values[key]), "num"),
                    sheet_cell(fmt_cost(calc[cost_key]), "num"),
                    sheet_cell("", "text"),
                    sheet_cell("", "text"),
                    sheet_cell("", "text"),
                    sheet_cell("", "text"),
                    sheet_cell("", "text"),
                    sheet_cell("", "text"),
                ]
            )
    rows.append(
        [
            sheet_cell("Total Fixed Charges", "total-green text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell(fmt_cost(calc["E23"]), "total-green num"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
        ]
    )
    rows.append(blank_row(11))
    rows.append(
        [
            sheet_cell("C. Plant Overhead Costs (5-15% of TCP)", "total-green text"),
            sheet_cell(fmt_rate_bound(0.05), "num"),
            sheet_cell(fmt_rate_bound(0.15), "num"),
            sheet_cell(fmt_selected(values["D26"]), "num"),
            sheet_cell(fmt_cost(calc["D26_cost"]), "total-green num"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
        ]
    )
    rows.append(blank_row(11))
    rows.append(
        [
            sheet_cell("Total Manufacturing Cost", "section-yellow text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell(fmt_cost(calc["E28"]), "section-yellow num"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
        ]
    )
    rows.append(blank_row(11))
    rows.append(
        [
            sheet_cell("", "text"),
            sheet_cell("Rate(%)", "head-green center", 3),
            sheet_cell("Cost", "head-green center"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
        ]
    )
    rows.append(
        [
            sheet_cell("II. General Expenses (15-25% of TCP)", "section-yellow text"),
            sheet_cell("Lower Limit", "head-peach center"),
            sheet_cell("Upper Limit", "head-peach center"),
            sheet_cell("Selected %", "head-peach center"),
            sheet_cell("", "head-peach center"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
        ]
    )
    rows.append(
        [
            sheet_cell("A. Administrative Costs (2-5% of TCP)", "total-green text"),
            sheet_cell(fmt_rate_bound(0.02), "num"),
            sheet_cell(fmt_rate_bound(0.05), "num"),
            sheet_cell(fmt_selected(values["D33"]), "num"),
            sheet_cell(fmt_cost(calc["D33_cost"]), "num"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
        ]
    )
    rows.append(
        [
            sheet_cell("B. Distribution and Marketing Costs (2-20% of TCP)", "total-green text"),
            sheet_cell(fmt_rate_bound(0.02), "num"),
            sheet_cell(fmt_rate_bound(0.20), "num"),
            sheet_cell(fmt_selected(values["D34"]), "num"),
            sheet_cell(fmt_cost(calc["D34_cost"]), "num"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
        ]
    )
    rows.append(
        [
            sheet_cell("C. Research and Development Costs ( About 5% of TCP)", "total-green text"),
            sheet_cell(fmt_rate_bound(0.04), "num"),
            sheet_cell(fmt_rate_bound(0.06), "num"),
            sheet_cell(fmt_selected(values["D35"]), "num"),
            sheet_cell(fmt_cost(calc["D35_cost"]), "num"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
        ]
    )
    rows.append(blank_row(11))
    rows.append(
        [
            sheet_cell("Total General Expenses", "section-yellow text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell(fmt_cost(calc["E37"]), "section-yellow num"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
        ]
    )
    rows.append(blank_row(11))
    rows.append(
        [
            sheet_cell("Total Product Cost", "total-green text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell(fmt_cost(calc["E39"]), "total-green num"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
            sheet_cell("", "text"),
        ]
    )
    for _ in range(4):
        rows.append(blank_row(11))
    return rows


def render_control_drawer() -> None:
    left, right = st.columns([4.5, 1.5])
    with left:
        st.markdown(
            """
            <div class="toolbar">
                <div>
                    <div class="toolbar-title">Chemical plant design economics dashboard</div>
                    <div class="drawer-hint">Use the control drawer to switch pages, choose the solver, and tune targets without keeping a bulky sidebar open.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        popover_fn = getattr(st, "popover", None)
        if popover_fn is None:
            drawer = st.expander("Open Controls", expanded=False)
        else:
            drawer = popover_fn("Open Controls", use_container_width=True)
        with drawer:
            st.markdown("### Navigation")
            st.radio(
                "Page",
                ["Chemical Plant Design Economics", "Sheet One", "Sheet Two"],
                key="active_page",
                label_visibility="collapsed",
            )
            st.markdown("---")
            st.markdown("### Solver Settings")
            st.selectbox(
                "Solver function",
                ["Least Squares", "SLSQP Penalty"],
                key="solver_method",
            )
            st.caption("Least Squares is steadier. SLSQP Penalty behaves more like a generic constrained optimizer. The preview path is fixed from 0 to 2000.")
            st.markdown("---")
            st.markdown("### Sheet Inputs")
            if st.session_state["active_page"] == "Sheet One":
                st.number_input("Target TCI", min_value=ASSIGNMENT_TARGET, step=1000.0, key="capital_target")
            elif st.session_state["active_page"] == "Sheet Two":
                st.number_input("Target TCP", min_value=ASSIGNMENT_TARGET, step=1000.0, key="product_target")
                st.number_input("Rent base", min_value=0.0, step=100.0, key="product_rent_base")
                st.number_input("Depreciation cost", min_value=0.0, step=100.0, key="product_depreciation")
            else:
                st.caption("Open Sheet One or Sheet Two to expose the sheet-specific numerical inputs here.")


def render_theory_page() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>Chemical Plant Design Economics</h1>
            <div class="professor-line">Prepared for Dr. Ejaz Ahmed</div>
            <p>
                This dashboard explains and solves the assignment in the same spirit as the spreadsheet:
                estimate plant investment and product cost from standard percentage relationships, then
                adjust the selected percentages until the totals match the required target.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="friendly-grid">
            <div class="friendly-card">
                <h3>The big idea</h3>
                <p>
                    A chemical plant has many cost items. Instead of estimating every nut, pipe, and service separately,
                    the assignment groups costs into major blocks and estimates each block as a percentage of a known base.
                </p>
            </div>
            <div class="friendly-card">
                <h3>Why percentages?</h3>
                <p>
                    Early design estimates usually do not have final vendor quotes. Percentage ranges give a practical
                    first estimate while keeping every assumption inside an acceptable engineering range.
                </p>
            </div>
            <div class="friendly-card">
                <h3>What the solver does</h3>
                <p>
                    The solver tries different selected percentages, recalculates the sheet, and keeps improving the
                    choices until the final total becomes very close to the required value.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    equation_left, equation_right = st.columns([1, 1], gap="large")
    with equation_left:
        st.markdown(
            """
            <div class="plain-equation">
                <div class="eq-title">Sheet One: total capital investment</div>
                <div class="eq-body">TCI = FCI + Working Capital</div>
            </div>
            <div class="plain-equation">
                <div class="eq-title">Fixed capital investment</div>
                <div class="eq-body">FCI = Direct Costs + Indirect Costs</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_notebook_cell(
            "Sheet One in plain language",
            """
            Sheet One answers: <strong>How much capital is needed to build the plant?</strong>
            It starts with fixed capital investment, adds working capital, and checks whether the selected
            percentages produce the required total capital investment.
            """,
        )
    with equation_right:
        st.markdown(
            """
            <div class="plain-equation">
                <div class="eq-title">Sheet Two: total product cost</div>
                <div class="eq-body">TCP = Manufacturing Cost + General Expenses</div>
            </div>
            <div class="plain-equation">
                <div class="eq-title">Manufacturing cost</div>
                <div class="eq-body">Manufacturing Cost = Direct Production + Fixed Charges + Overhead</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_notebook_cell(
            "Sheet Two in plain language",
            """
            Sheet Two answers: <strong>How much does it cost to produce the product?</strong>
            It combines raw materials, labour, utilities, fixed charges, overhead, and general expenses until
            the total product cost reaches the required target.
            """,
        )

    st.markdown("### How To Read The Sheets")
    st.markdown(
        """
        <div class="walkthrough">
            <div class="walkthrough-card">
                <h4>Start with the target</h4>
                <p>The yellow target cell is the value the assignment asks the sheet to reach.</p>
            </div>
            <div class="walkthrough-card">
                <h4>Choose percentages</h4>
                <p>Each selected percentage must remain between its lower and upper limit.</p>
            </div>
            <div class="walkthrough-card">
                <h4>Calculate costs</h4>
                <p>The sheet multiplies each selected percentage by its base cost to get each row value.</p>
            </div>
            <div class="walkthrough-card">
                <h4>Balance the result</h4>
                <p>The solver keeps adjusting the percentages until the final cost and internal checks agree.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_notebook_cell(
        "How to use this dashboard",
        """
        Open <strong>Sheet One</strong> for capital investment and <strong>Sheet Two</strong> for product cost.
        Use <strong>Randomize</strong> to start from a fresh set of assumptions, then press <strong>Solve</strong>
        to let the dashboard settle into a consistent answer. The preview slider shows how the numbers move from
        the random starting point toward the final balanced sheet.
        """,
    )


def render_capital_page(max_evals: int, method: str) -> None:
    target = float(st.session_state["capital_target"])
    current_values = dict(st.session_state["capital_values"])
    current_calc = capital_calculator(current_values, target)
    current_path = st.session_state.get("capital_path", [dict(current_values)])

    render_hero(
        "Sheet One",
        "Capital investment sheet for estimating the total money required to build the plant, including fixed capital and working capital.",
    )

    top_left, top_mid, top_right, top_last = st.columns(4)
    top_left.metric("Target TCI (E31)", format_money(target))
    top_mid.metric("Calculated TCI", format_money(current_calc["E31"]), f"{current_calc['target_residual']:+.4f}")
    top_right.metric("FCI Balance (E5 - E30)", f"{current_calc['fci_balance']:+.4f}")
    top_last.metric(
        "Last solver evaluations",
        str(st.session_state["capital_meta"]["nfev"]),
        method,
    )

    info_col, solver_col = st.columns([1, 1], gap="large")
    with info_col:
        render_notebook_cell(
            "What Sheet One Calculates",
            """
            This sheet estimates total capital investment. Direct costs are built from purchased equipment,
            indirect costs are added on top, and working capital is included at the end to reach the final plant investment.
            """,
        )
    with solver_col:
        render_notebook_cell(
            "How The Solver Helps",
            f"""
            The solver changes the selected percentage cells and repeatedly recalculates the sheet until
            the final capital investment is close to the target and the fixed-capital check balances.
            Current method: <strong>{method}</strong>. Evaluation budget: <strong>{max_evals}</strong>.
            """,
        )

    updated_values = render_editor(
        "Editable capital variables",
        CAPITAL_SPECS,
        current_values,
        current_calc,
        editor_key=f"capital_editor_full_{st.session_state['capital_nonce']}",
        sheet_height=330,
    )
    has_unsaved_edits = any(
        abs(updated_values[spec.cell] - current_values[spec.cell]) > 1e-12
        for spec in CAPITAL_SPECS
    )
    if has_unsaved_edits:
        display_path = [dict(updated_values)]
        st.session_state["capital_iteration_view"] = 0
    else:
        display_path = current_path
    if "capital_iteration_view" not in st.session_state:
        st.session_state["capital_iteration_view"] = PREVIEW_STEPS
    st.session_state["capital_iteration_view"] = min(
        PREVIEW_STEPS,
        max(0, int(st.session_state["capital_iteration_view"])),
    )
    preview_step = int(st.session_state["capital_iteration_view"])
    display_index = path_index_from_preview_step(preview_step, len(display_path))
    display_values = dict(display_path[display_index])
    calc = capital_calculator(display_values, target)
    sheet_rows = build_capital_sheet_rows(display_values, target, calc)
    formula_map = build_grid_formula_map(
        sheet_rows,
        build_capital_formula_overrides(display_values, target, calc),
    )

    capital_contrib = {
        "Working Capital": calc["D4_cost"],
        "Direct Costs": calc["E18"],
        "Indirect Costs": calc["E28"],
    }
    st.markdown("### Interactive Sheet")
    zoom_pct, font_family, font_size, active_cell = render_sheet_chrome("sheet_one", formula_map)
    workbook_left, _, workbook_right = st.columns([6.4, 0.25, 2.35], gap="small")
    with workbook_left:
        render_excel_sheet(
            sheet_rows,
            ["440px", "92px", "92px", "138px", "136px", "72px", "40px", "260px", "110px", "170px"],
            page_key="sheet_one",
            active_cell=active_cell,
            clickable_map=formula_map,
            zoom_pct=zoom_pct,
            font_family=font_family,
            font_size=font_size,
        )

    with workbook_right:
        st.markdown("#### Solver Rail")
        action_left, action_right = st.columns(2, gap="small")
        with action_left:
            if st.button("Solve", use_container_width=True, type="primary", key="sheet_one_solve_inline"):
                solved, trace, path, meta = solve_capital(updated_values, target, max_evals, method)
                st.session_state["capital_values"] = solved
                st.session_state["capital_trace"] = trace
                st.session_state["capital_path"] = path
                st.session_state["capital_meta"] = meta
                st.session_state["capital_iteration_view"] = PREVIEW_STEPS
                st.session_state["capital_nonce"] += 1
                st.rerun()
        with action_right:
            if st.button("Randomize", use_container_width=True, key="sheet_one_randomize_inline"):
                random_values = randomize_values(CAPITAL_SPECS)
                _, trace, path, meta = solve_capital(random_values, target, max_evals, method)
                st.session_state["capital_values"] = random_values
                st.session_state["capital_trace"] = trace
                st.session_state["capital_path"] = path
                st.session_state["capital_meta"] = {
                    **meta,
                    "message": "Randomized assumptions | preview path ready",
                }
                st.session_state["capital_iteration_view"] = 0
                st.session_state["capital_nonce"] += 1
                st.rerun()
        st.caption(
            f"Solver status: {st.session_state['capital_meta']['message']} | success={st.session_state['capital_meta']['success']}"
        )
        render_iteration_panel(
            title="Iteration Preview",
            subtitle="Scrub the optimization path from the randomized starting assumptions to the converged capital solution.",
            display_index=preview_step,
            has_unsaved_edits=has_unsaved_edits,
            iteration_key="capital_iteration_view",
            compact=True,
        )
        if has_unsaved_edits:
            st.caption("You have unsolved manual edits, so the sheet is showing the edited state directly. Run the solver to generate a full path from random start to steady state.")
        elif len(display_path) > 1:
            st.caption("Step 0 is the randomized start. Step 2000 is the steady-state solution.")
        else:
            st.caption("Randomize or solve to create an iteration path.")

        st.markdown("#### Contribution Split")
        contrib_fig = contribution_chart(capital_contrib, "Contribution split")
        contrib_fig.update_layout(height=305, margin=dict(l=88, r=6, t=32, b=4), title_font_size=13)
        contrib_fig.update_traces(textposition="inside", textfont_size=10)
        st.plotly_chart(contrib_fig, use_container_width=True)

    st.markdown("### Sheet One Solver Trace")
    trace_fig = trace_chart(st.session_state["capital_trace"], "Sheet One solver trace")
    trace_fig.update_layout(height=330, margin=dict(l=10, r=10, t=45, b=20))
    st.plotly_chart(trace_fig, use_container_width=True)


def render_product_page(max_evals: int, method: str) -> None:
    target = float(st.session_state["product_target"])
    rent_base = float(st.session_state["product_rent_base"])
    depreciation_cost = float(st.session_state["product_depreciation"])
    current_values = dict(st.session_state["product_values"])
    current_calc = product_calculator(current_values, target, rent_base, depreciation_cost)
    current_path = st.session_state.get("product_path", [dict(current_values)])

    render_hero(
        "Sheet Two",
        "Product cost sheet for estimating the cost of manufacturing the product, including production costs, fixed charges, overhead, and general expenses.",
    )

    top_left, top_mid, top_right, top_last = st.columns(4)
    top_left.metric("Target TCP (E39)", format_money(target))
    top_mid.metric("Calculated TCP", format_money(current_calc["E39"]), f"{current_calc['target_residual']:+.4f}")
    top_right.metric(
        "Summary balance",
        f"{current_calc['direct_balance']:+.2f} / {current_calc['fixed_balance']:+.2f} / {current_calc['general_balance']:+.2f}",
    )
    top_last.metric(
        "Last solver evaluations",
        str(st.session_state["product_meta"]["nfev"]),
        method,
    )

    info_col, solver_col = st.columns([1, 1], gap="large")
    with info_col:
        render_notebook_cell(
            "What Sheet Two Calculates",
            """
            This sheet estimates total product cost. It combines direct production costs, fixed charges,
            plant overhead, and general expenses so the final product cost reaches the required total.
            """,
        )
    with solver_col:
        render_notebook_cell(
            "How The Solver Helps",
            f"""
            The solver adjusts the selected percentages while keeping the direct-production, fixed-charge,
            and general-expense summaries consistent with their detailed rows. Rent base and depreciation
            are exposed as separate assumptions because the original sheet leaves them implicit.
            <br><br>
            Current method: <strong>{method}</strong><br>
            Iteration budget: <strong>{max_evals}</strong><br>
            Rent base: <strong>{format_money(rent_base)}</strong><br>
            Depreciation cost: <strong>{format_money(depreciation_cost)}</strong>
            """,
        )

    updated_values = render_editor(
        "Editable product-cost variables",
        PRODUCT_SPECS,
        current_values,
        current_calc,
        editor_key=f"product_editor_full_{st.session_state['product_nonce']}",
        sheet_height=400,
    )
    has_unsaved_edits = any(
        abs(updated_values[spec.cell] - current_values[spec.cell]) > 1e-12
        for spec in PRODUCT_SPECS
    )
    if has_unsaved_edits:
        display_path = [dict(updated_values)]
        st.session_state["product_iteration_view"] = 0
    else:
        display_path = current_path
    if "product_iteration_view" not in st.session_state:
        st.session_state["product_iteration_view"] = PREVIEW_STEPS
    st.session_state["product_iteration_view"] = min(
        PREVIEW_STEPS,
        max(0, int(st.session_state["product_iteration_view"])),
    )
    preview_step = int(st.session_state["product_iteration_view"])
    display_index = path_index_from_preview_step(preview_step, len(display_path))
    display_values = dict(display_path[display_index])
    calc = product_calculator(display_values, target, rent_base, depreciation_cost)
    sheet_rows = build_product_sheet_rows(display_values, target, calc)
    formula_map = build_grid_formula_map(
        sheet_rows,
        build_product_formula_overrides(
            display_values,
            target,
            calc,
            rent_base,
            depreciation_cost,
        ),
    )

    product_contrib = {
        "Direct Production": calc["E14"],
        "Fixed Charges": calc["E23"],
        "Plant Overhead": calc["D26_cost"],
        "General Expenses": calc["E37"],
    }
    st.markdown("### Interactive Sheet")
    zoom_pct, font_family, font_size, active_cell = render_sheet_chrome("sheet_two", formula_map)
    workbook_left, _, workbook_right = st.columns([6.4, 0.25, 2.35], gap="small")
    with workbook_left:
        render_excel_sheet(
            sheet_rows,
            ["520px", "98px", "98px", "146px", "138px", "76px", "76px", "210px", "110px", "170px", "50px"],
            page_key="sheet_two",
            active_cell=active_cell,
            clickable_map=formula_map,
            zoom_pct=zoom_pct,
            font_family=font_family,
            font_size=font_size,
        )

    with workbook_right:
        st.markdown("#### Solver Rail")
        action_left, action_right = st.columns(2, gap="small")
        with action_left:
            if st.button("Solve", use_container_width=True, type="primary", key="sheet_two_solve_inline"):
                solved, trace, path, meta = solve_product(
                    updated_values,
                    target,
                    rent_base,
                    depreciation_cost,
                    max_evals,
                    method,
                )
                st.session_state["product_values"] = solved
                st.session_state["product_trace"] = trace
                st.session_state["product_path"] = path
                st.session_state["product_meta"] = meta
                st.session_state["product_iteration_view"] = PREVIEW_STEPS
                st.session_state["product_nonce"] += 1
                st.rerun()
        with action_right:
            if st.button("Randomize", use_container_width=True, key="sheet_two_randomize_inline"):
                random_values = randomize_values(PRODUCT_SPECS)
                _, trace, path, meta = solve_product(
                    random_values,
                    target,
                    rent_base,
                    depreciation_cost,
                    max_evals,
                    method,
                )
                st.session_state["product_values"] = random_values
                st.session_state["product_trace"] = trace
                st.session_state["product_path"] = path
                st.session_state["product_meta"] = {
                    **meta,
                    "message": "Randomized assumptions | preview path ready",
                }
                st.session_state["product_iteration_view"] = 0
                st.session_state["product_nonce"] += 1
                st.rerun()
        st.caption(
            f"Solver status: {st.session_state['product_meta']['message']} | success={st.session_state['product_meta']['success']}"
        )
        render_iteration_panel(
            title="Iteration Preview",
            subtitle="Scrub the optimization path from the randomized starting assumptions to the converged product-cost solution.",
            display_index=preview_step,
            has_unsaved_edits=has_unsaved_edits,
            iteration_key="product_iteration_view",
            compact=True,
        )
        if has_unsaved_edits:
            st.caption("You have unsolved manual edits, so the sheet is showing the edited state directly. Run the solver to generate a full path from random start to steady state.")
        elif len(display_path) > 1:
            st.caption("Step 0 is the randomized start. Step 2000 is the steady-state solution.")
        else:
            st.caption("Randomize or solve to create an iteration path.")

        st.markdown("#### Contribution Split")
        contrib_fig = contribution_chart(product_contrib, "Contribution split")
        contrib_fig.update_layout(height=305, margin=dict(l=102, r=6, t=32, b=4), title_font_size=13)
        contrib_fig.update_traces(textposition="inside", textfont_size=10)
        st.plotly_chart(contrib_fig, use_container_width=True)

    st.markdown("### Sheet Two Solver Trace")
    trace_fig = trace_chart(st.session_state["product_trace"], "Sheet Two solver trace")
    trace_fig.update_layout(height=330, margin=dict(l=10, r=10, t=45, b=20))
    st.plotly_chart(trace_fig, use_container_width=True)


def main() -> None:
    inject_css()
    init_state()
    render_control_drawer()

    page = st.session_state["active_page"]
    max_evals = FIXED_SOLVER_EVALS
    method = str(st.session_state["solver_method"])

    if page == "Chemical Plant Design Economics":
        render_theory_page()
    elif page == "Sheet One":
        render_capital_page(max_evals, method)
    else:
        render_product_page(max_evals, method)


if __name__ == "__main__":
    main()
