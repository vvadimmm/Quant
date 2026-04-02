import numpy as np
from scipy.stats import norm
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
#  BLACK-SCHOLES ENGINE
# ─────────────────────────────────────────────

class Option:
    def __init__(self, S, K, T, r, sigma, option_type):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()

def _d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def _d2(S, K, T, r, sigma):
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def bs_price(opt):
    d1 = _d1(opt.S, opt.K, opt.T, opt.r, opt.sigma)
    d2 = d1 - opt.sigma * np.sqrt(opt.T)
    if opt.option_type == "call":
        return opt.S * norm.cdf(d1) - opt.K * np.exp(-opt.r * opt.T) * norm.cdf(d2)
    else:
        return -opt.S * norm.cdf(-d1) + opt.K * np.exp(-opt.r * opt.T) * norm.cdf(-d2)

def bs_delta(opt):
    d1 = _d1(opt.S, opt.K, opt.T, opt.r, opt.sigma)
    if opt.option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

def bs_gamma(opt):
    d1 = _d1(opt.S, opt.K, opt.T, opt.r, opt.sigma)
    return norm.pdf(d1) / (opt.S * opt.sigma * np.sqrt(opt.T))

def bs_vega(opt):
    d1 = _d1(opt.S, opt.K, opt.T, opt.r, opt.sigma)
    return opt.S * norm.pdf(d1) * np.sqrt(opt.T)

def bs_theta(opt):
    d1 = _d1(opt.S, opt.K, opt.T, opt.r, opt.sigma)
    d2 = d1 - opt.sigma * np.sqrt(opt.T)
    if opt.option_type == "call":
        return (-opt.S * norm.pdf(d1) * opt.sigma / (2 * np.sqrt(opt.T))
                - opt.r * opt.K * np.exp(-opt.r * opt.T) * norm.cdf(d2))
    else:
        return (-opt.S * norm.pdf(d1) * opt.sigma / (2 * np.sqrt(opt.T))
                + opt.r * opt.K * np.exp(-opt.r * opt.T) * norm.cdf(-d2))

def bs_rho(opt):
    d2 = _d1(opt.S, opt.K, opt.T, opt.r, opt.sigma) - opt.sigma * np.sqrt(opt.T)
    if opt.option_type == "call":
        return opt.K * opt.T * np.exp(-opt.r * opt.T) * norm.cdf(d2)
    else:
        return -opt.K * opt.T * np.exp(-opt.r * opt.T) * norm.cdf(-d2)

def compute_greeks(opt):
    return {
        "Price": bs_price(opt),
        "Delta": bs_delta(opt),
        "Gamma": bs_gamma(opt),
        "Vega":  bs_vega(opt),
        "Theta": bs_theta(opt),
        "Rho":   bs_rho(opt),
    }

# ─────────────────────────────────────────────
#  PORTFOLIO
# ─────────────────────────────────────────────

class Portfolio:
    def __init__(self):
        self.positions = []   # list of (Option, quantity, label)

    def add(self, option, quantity, label=""):
        self.positions.append((option, quantity, label))

    def remove(self, index):
        del self.positions[index]

    def total_greeks(self):
        totals = {"Price": 0, "Delta": 0, "Gamma": 0, "Vega": 0, "Theta": 0, "Rho": 0}
        for opt, qty, _ in self.positions:
            g = compute_greeks(opt)
            for k in totals:
                totals[k] += g[k] * qty
        return totals

    def total_greeks_with_override(self, x_param, x_val):
        """Compute aggregated greeks while overriding one parameter across all positions."""
        param_map = {"S": "S", "K": "K", "T": "T", "r": "r", "σ": "sigma"}
        attr = param_map[x_param]
        totals = {"Price": 0, "Delta": 0, "Gamma": 0, "Vega": 0, "Theta": 0, "Rho": 0}
        for opt, qty, _ in self.positions:
            # Build a modified copy of the option
            kwargs = dict(S=opt.S, K=opt.K, T=opt.T, r=opt.r,
                          sigma=opt.sigma, option_type=opt.option_type)
            if x_param in ("r", "σ"):
                kwargs[attr] = x_val / 100
            else:
                kwargs[attr] = x_val
            mod_opt = Option(**kwargs)
            g = compute_greeks(mod_opt)
            for k in totals:
                totals[k] += g[k] * qty
        return totals

# ─────────────────────────────────────────────
#  THEME
# ─────────────────────────────────────────────

BG        = "#0d0f14"
BG2       = "#13161e"
BG3       = "#1c2030"
BORDER    = "#252a3a"
ACCENT    = "#4fffb0"       # neon green
ACCENT2   = "#ff6b6b"       # coral red
ACCENT3   = "#7b8cde"       # periwinkle
TEXT      = "#e8ecf0"
TEXT_DIM  = "#6b7394"
CALL_CLR  = "#4fffb0"
PUT_CLR   = "#ff6b6b"
FONT_MONO = ("Courier New", 10)
FONT_HEAD = ("Georgia", 12, "bold")
FONT_BIG  = ("Georgia", 16, "bold")
FONT_SM   = ("Courier New", 9)

GREEK_COLORS = {
    "Price": "#f0c060",
    "Delta": "#4fffb0",
    "Gamma": "#7b8cde",
    "Vega":  "#ff9f6b",
    "Theta": "#ff6b6b",
    "Rho":   "#c06bff",
}

def apply_theme(root):
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TFrame",        background=BG)
    style.configure("TLabel",        background=BG,  foreground=TEXT, font=FONT_MONO)
    style.configure("TButton",       background=BG3, foreground=TEXT, font=FONT_MONO,
                    borderwidth=1, relief="flat")
    style.map("TButton",
              background=[("active", BORDER)],
              foreground=[("active", ACCENT)])
    style.configure("TEntry",        fieldbackground=BG3, foreground=TEXT,
                    insertcolor=ACCENT, font=FONT_MONO, borderwidth=0)
    style.configure("TCombobox",     fieldbackground=BG3, foreground=TEXT,
                    selectbackground=BG3, selectforeground=ACCENT, font=FONT_MONO)
    style.map("TCombobox",
              fieldbackground=[("readonly", BG3)],
              selectbackground=[("readonly", BG3)],
              foreground=[("readonly", TEXT)])
    style.configure("TNotebook",     background=BG,  borderwidth=0)
    style.configure("TNotebook.Tab", background=BG3, foreground=TEXT_DIM,
                    font=FONT_MONO, padding=[14, 6])
    style.map("TNotebook.Tab",
              background=[("selected", BG2)],
              foreground=[("selected", ACCENT)])
    style.configure("Treeview",      background=BG2, foreground=TEXT,
                    fieldbackground=BG2, font=FONT_MONO, rowheight=26)
    style.configure("Treeview.Heading", background=BG3, foreground=ACCENT,
                    font=FONT_MONO, relief="flat")
    style.map("Treeview", background=[("selected", BG3)], foreground=[("selected", ACCENT)])
    style.configure("Vertical.TScrollbar", background=BG3, troughcolor=BG, arrowcolor=TEXT_DIM)

# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────

class OptionsApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Options Portfolio — Greeks Analyzer")
        self.geometry("1200x800")
        self.configure(bg=BG)
        self.resizable(True, True)
        apply_theme(self)

        self.portfolio = Portfolio()

        # ── Header ──────────────────────────────────
        hdr = tk.Frame(self, bg=BG, pady=0)
        hdr.pack(fill="x")
        tk.Label(hdr, text="⬡  OPTIONS PORTFOLIO", bg=BG,
                 fg=ACCENT, font=("Courier New", 14, "bold"),
                 padx=20, pady=12).pack(side="left")
        tk.Label(hdr, text="Black-Scholes Greeks Analyzer",
                 bg=BG, fg=TEXT_DIM, font=FONT_SM, pady=12).pack(side="left")
        sep = tk.Frame(self, bg=BORDER, height=1)
        sep.pack(fill="x")

        # ── Tabs ────────────────────────────────────
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=0, pady=0)

        self.tab_portfolio = tk.Frame(nb, bg=BG)
        self.tab_analysis  = tk.Frame(nb, bg=BG)

        nb.add(self.tab_portfolio, text="  Portfolio  ")
        nb.add(self.tab_analysis,  text="  Sensitivity Analysis  ")

        self._build_portfolio_tab()
        self._build_analysis_tab()

    # ══════════════════════════════════════════
    #  TAB 1 — PORTFOLIO
    # ══════════════════════════════════════════

    def _build_portfolio_tab(self):
        tab = self.tab_portfolio

        # ── Left panel: form ──────────────────────
        left = tk.Frame(tab, bg=BG2, width=340)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        tk.Label(left, text="ADD OPTION", bg=BG2, fg=ACCENT,
                 font=("Courier New", 11, "bold"), padx=20, pady=14).pack(anchor="w")
        sep = tk.Frame(left, bg=BORDER, height=1)
        sep.pack(fill="x")

        form = tk.Frame(left, bg=BG2, padx=20, pady=16)
        form.pack(fill="x")

        fields = [
            ("S  — Spot Price",      "100"),
            ("K  — Strike",          "100"),
            ("T  — Maturity (yr)",   "0.25"),
            ("r  — Rate (%)",        "5"),
            ("σ  — Volatility (%)",  "20"),
        ]
        self.entries = {}
        for label, default in fields:
            row = tk.Frame(form, bg=BG2, pady=4)
            row.pack(fill="x")
            tk.Label(row, text=label, bg=BG2, fg=TEXT_DIM,
                     font=FONT_SM, anchor="w").pack(fill="x")
            e = tk.Entry(row, bg=BG3, fg=TEXT, insertbackground=ACCENT,
                         relief="flat", font=FONT_MONO, bd=4)
            e.insert(0, default)
            e.pack(fill="x", ipady=4)
            key = label.split("—")[0].strip().split()[0]
            self.entries[key] = e

        # Type + Direction
        row_type = tk.Frame(form, bg=BG2, pady=4)
        row_type.pack(fill="x")
        tk.Label(row_type, text="Type", bg=BG2, fg=TEXT_DIM,
                 font=FONT_SM, anchor="w").pack(fill="x")
        self.opt_type = ttk.Combobox(row_type, values=["Call", "Put"],
                                     state="readonly", font=FONT_MONO)
        self.opt_type.set("Call")
        self.opt_type.pack(fill="x", ipady=3)

        row_dir = tk.Frame(form, bg=BG2, pady=4)
        row_dir.pack(fill="x")
        tk.Label(row_dir, text="Direction", bg=BG2, fg=TEXT_DIM,
                 font=FONT_SM, anchor="w").pack(fill="x")
        self.opt_dir = ttk.Combobox(row_dir, values=["Long (+1)", "Short (-1)"],
                                    state="readonly", font=FONT_MONO)
        self.opt_dir.set("Long (+1)")
        self.opt_dir.pack(fill="x", ipady=3)

        row_qty = tk.Frame(form, bg=BG2, pady=4)
        row_qty.pack(fill="x")
        tk.Label(row_qty, text="Quantity", bg=BG2, fg=TEXT_DIM,
                 font=FONT_SM, anchor="w").pack(fill="x")
        self.entry_qty = tk.Entry(row_qty, bg=BG3, fg=TEXT, insertbackground=ACCENT,
                                  relief="flat", font=FONT_MONO, bd=4)
        self.entry_qty.insert(0, "1")
        self.entry_qty.pack(fill="x", ipady=4)

        # Buttons
        btn_frame = tk.Frame(left, bg=BG2, padx=20, pady=6)
        btn_frame.pack(fill="x")

        btn_add = tk.Button(btn_frame, text="＋  ADD TO PORTFOLIO",
                            bg=ACCENT, fg=BG, font=("Courier New", 10, "bold"),
                            relief="flat", cursor="hand2", bd=0,
                            activebackground="#2dcc80", activeforeground=BG,
                            command=self._add_option)
        btn_add.pack(fill="x", ipady=7, pady=(0, 6))

        btn_del = tk.Button(btn_frame, text="✕  REMOVE SELECTED",
                            bg=BG3, fg=ACCENT2, font=("Courier New", 10, "bold"),
                            relief="flat", cursor="hand2", bd=0,
                            activebackground=BORDER, activeforeground=ACCENT2,
                            command=self._remove_option)
        btn_del.pack(fill="x", ipady=5)

        btn_clr = tk.Button(btn_frame, text="↺  CLEAR ALL",
                            bg=BG3, fg=TEXT_DIM, font=("Courier New", 10),
                            relief="flat", cursor="hand2", bd=0,
                            activebackground=BORDER, activeforeground=TEXT,
                            command=self._clear_portfolio)
        btn_clr.pack(fill="x", ipady=5, pady=(4, 0))

        # ── Right panel ───────────────────────────
        right = tk.Frame(tab, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        # Positions table
        tbl_frame = tk.Frame(right, bg=BG, padx=16, pady=12)
        tbl_frame.pack(fill="both", expand=True)

        tk.Label(tbl_frame, text="POSITIONS", bg=BG, fg=TEXT_DIM,
                 font=FONT_SM).pack(anchor="w", pady=(0, 6))

        cols = ("Type", "Dir", "Qty", "S", "K", "T", "r%", "σ%",
                "Price", "Delta", "Gamma", "Vega", "Theta", "Rho")
        self.tree = ttk.Treeview(tbl_frame, columns=cols, show="headings", height=8)
        widths = [50, 55, 45, 55, 55, 55, 45, 45, 72, 72, 72, 72, 72, 72]
        for col, w in zip(cols, widths):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor="center", stretch=False)
        vsb = ttk.Scrollbar(tbl_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(tbl_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # Portfolio Greeks summary
        summary_frame = tk.Frame(right, bg=BG, padx=16, pady=4)
        summary_frame.pack(fill="x")

        sep2 = tk.Frame(right, bg=BORDER, height=1)
        sep2.pack(fill="x", padx=16)

        tk.Label(right, text="PORTFOLIO GREEKS", bg=BG, fg=TEXT_DIM,
                 font=FONT_SM, pady=8, padx=16).pack(anchor="w")

        greek_grid = tk.Frame(right, bg=BG, padx=16, pady=6)
        greek_grid.pack(fill="x")

        self.greek_labels = {}
        greek_names = ["Price", "Delta", "Gamma", "Vega", "Theta", "Rho"]
        for i, name in enumerate(greek_names):
            col = i % 3
            row = i // 3
            card = tk.Frame(greek_grid, bg=BG3, padx=16, pady=10,
                            highlightbackground=GREEK_COLORS[name],
                            highlightthickness=1)
            card.grid(row=row, column=col, padx=6, pady=5, sticky="ew")
            greek_grid.columnconfigure(col, weight=1)
            tk.Label(card, text=name.upper(), bg=BG3,
                     fg=GREEK_COLORS[name], font=FONT_SM).pack(anchor="w")
            lbl = tk.Label(card, text="—", bg=BG3, fg=TEXT,
                           font=("Courier New", 14, "bold"))
            lbl.pack(anchor="w")
            self.greek_labels[name] = lbl

    # ══════════════════════════════════════════
    #  TAB 2 — SENSITIVITY ANALYSIS
    # ══════════════════════════════════════════

    def _build_analysis_tab(self):
        tab = self.tab_analysis

        # Controls panel
        ctrl = tk.Frame(tab, bg=BG2, width=300)
        ctrl.pack(side="left", fill="y")
        ctrl.pack_propagate(False)

        tk.Label(ctrl, text="ANALYSIS SETTINGS", bg=BG2, fg=ACCENT,
                 font=("Courier New", 11, "bold"), padx=20, pady=14).pack(anchor="w")
        tk.Frame(ctrl, bg=BORDER, height=1).pack(fill="x")

        form2 = tk.Frame(ctrl, bg=BG2, padx=20, pady=16)
        form2.pack(fill="x")

        # ── Mode selector ──────────────────────────
        row_mode = tk.Frame(form2, bg=BG2, pady=4)
        row_mode.pack(fill="x")
        tk.Label(row_mode, text="Analysis Mode", bg=BG2, fg=TEXT_DIM,
                 font=FONT_SM, anchor="w").pack(fill="x")
        self.analysis_mode = ttk.Combobox(row_mode,
                                          values=["Portfolio", "Single Option"],
                                          state="readonly", font=FONT_MONO)
        self.analysis_mode.set("Portfolio")
        self.analysis_mode.pack(fill="x", ipady=3)
        self.analysis_mode.bind("<<ComboboxSelected>>", self._on_mode_change)

        tk.Frame(form2, bg=BORDER, height=1).pack(fill="x", pady=8)

        # Portfolio info label
        self.portfolio_info_label = tk.Label(
            form2, text="", bg=BG2, fg=ACCENT, font=FONT_SM,
            wraplength=240, justify="left"
        )
        self.portfolio_info_label.pack(fill="x", pady=(0, 4))

        # Single option params (toggled by mode)
        self.single_option_frame = tk.Frame(form2, bg=BG2)
        base_fields = [
            ("S  — Spot", "s_base", "100"),
            ("K  — Strike", "k_base", "100"),
            ("T  — Maturity (yr)", "t_base", "0.25"),
            ("r  — Rate (%)", "r_base", "5"),
            ("σ  — Volatility (%)", "v_base", "20"),
        ]
        self.analysis_entries = {}
        for label, key, default in base_fields:
            row = tk.Frame(self.single_option_frame, bg=BG2, pady=3)
            row.pack(fill="x")
            tk.Label(row, text=label, bg=BG2, fg=TEXT_DIM,
                     font=FONT_SM, anchor="w").pack(fill="x")
            e = tk.Entry(row, bg=BG3, fg=TEXT, insertbackground=ACCENT,
                         relief="flat", font=FONT_MONO, bd=4)
            e.insert(0, default)
            e.pack(fill="x", ipady=4)
            self.analysis_entries[key] = e

        row_type2 = tk.Frame(self.single_option_frame, bg=BG2, pady=4)
        row_type2.pack(fill="x")
        tk.Label(row_type2, text="Option Type", bg=BG2, fg=TEXT_DIM,
                 font=FONT_SM, anchor="w").pack(fill="x")
        self.analysis_type = ttk.Combobox(row_type2, values=["Call", "Put"],
                                          state="readonly", font=FONT_MONO)
        self.analysis_type.set("Call")
        self.analysis_type.pack(fill="x", ipady=3)

        # Default: Portfolio mode -> hide single option frame, show info
        self._update_portfolio_info()

        tk.Frame(form2, bg=BORDER, height=1).pack(fill="x", pady=8)

        # X-axis variable
        row_xvar = tk.Frame(form2, bg=BG2, pady=4)
        row_xvar.pack(fill="x")
        tk.Label(row_xvar, text="X-axis Variable", bg=BG2, fg=TEXT_DIM,
                 font=FONT_SM, anchor="w").pack(fill="x")
        self.x_var = ttk.Combobox(row_xvar,
                                   values=["S (Spot)", "K (Strike)", "T (Maturity)",
                                           "r (Rate)", "σ (Volatility)"],
                                   state="readonly", font=FONT_MONO)
        self.x_var.set("S (Spot)")
        self.x_var.pack(fill="x", ipady=3)

        # Range
        range_row = tk.Frame(form2, bg=BG2, pady=4)
        range_row.pack(fill="x")
        tk.Label(range_row, text="X Range  (min → max)", bg=BG2, fg=TEXT_DIM,
                 font=FONT_SM, anchor="w").pack(fill="x")
        rng_inner = tk.Frame(range_row, bg=BG2)
        rng_inner.pack(fill="x")
        self.x_min = tk.Entry(rng_inner, bg=BG3, fg=TEXT, insertbackground=ACCENT,
                              relief="flat", font=FONT_MONO, bd=4, width=8)
        self.x_min.insert(0, "50")
        self.x_min.pack(side="left", ipady=4, expand=True, fill="x")
        tk.Label(rng_inner, text="  →  ", bg=BG2, fg=TEXT_DIM, font=FONT_SM).pack(side="left")
        self.x_max = tk.Entry(rng_inner, bg=BG3, fg=TEXT, insertbackground=ACCENT,
                              relief="flat", font=FONT_MONO, bd=4, width=8)
        self.x_max.insert(0, "150")
        self.x_max.pack(side="left", ipady=4, expand=True, fill="x")

        # Y-axis (metric)
        row_yvar = tk.Frame(form2, bg=BG2, pady=4)
        row_yvar.pack(fill="x")
        tk.Label(row_yvar, text="Y-axis Metric", bg=BG2, fg=TEXT_DIM,
                 font=FONT_SM, anchor="w").pack(fill="x")
        self.y_var = ttk.Combobox(row_yvar,
                                   values=["Price", "Delta", "Gamma",
                                           "Vega", "Theta", "Rho", "All Greeks"],
                                   state="readonly", font=FONT_MONO)
        self.y_var.set("Delta")
        self.y_var.pack(fill="x", ipady=3)

        btn_plot = tk.Button(form2, text="▶  PLOT",
                             bg=ACCENT3, fg=BG, font=("Courier New", 10, "bold"),
                             relief="flat", cursor="hand2", bd=0,
                             activebackground="#9babf0", activeforeground=BG,
                             command=self._plot_sensitivity)
        btn_plot.pack(fill="x", ipady=8, pady=(12, 0))

        # Chart area
        chart_frame = tk.Frame(tab, bg=BG)
        chart_frame.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        self.fig = Figure(figsize=(7, 5.5), dpi=100, facecolor=BG)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Initial plot — skip if portfolio is empty (default mode)
        # (will be triggered by user clicking Plot)

    # ══════════════════════════════════════════
    #  ACTIONS
    # ══════════════════════════════════════════

    def _get_float(self, entry, name):
        try:
            return float(entry.get())
        except ValueError:
            raise ValueError(f"Invalid value for {name}")

    def _add_option(self):
        try:
            S     = self._get_float(self.entries["S"], "S")
            K     = self._get_float(self.entries["K"], "K")
            T     = self._get_float(self.entries["T"], "T")
            r     = self._get_float(self.entries["r"], "r") / 100
            sigma = self._get_float(self.entries["σ"], "σ") / 100
            qty   = self._get_float(self.entry_qty, "Quantity")
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return

        if T <= 0:
            messagebox.showerror("Input Error", "Maturity T must be > 0")
            return
        if sigma <= 0:
            messagebox.showerror("Input Error", "Volatility σ must be > 0")
            return

        opt_type  = self.opt_type.get().lower()
        direction = 1 if "Long" in self.opt_dir.get() else -1
        effective_qty = qty * direction

        opt = Option(S, K, T, r, sigma, opt_type)
        label = f"{'Call' if opt_type == 'call' else 'Put'} K={K}"
        self.portfolio.add(opt, effective_qty, label)
        self._refresh_table()
        self._refresh_greeks()

    def _remove_option(self):
        sel = self.tree.selection()
        if not sel:
            return
        idx = self.tree.index(sel[0])
        self.portfolio.remove(idx)
        self._refresh_table()
        self._refresh_greeks()

    def _clear_portfolio(self):
        self.portfolio.positions.clear()
        self._refresh_table()
        self._refresh_greeks()

    def _refresh_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for opt, qty, _ in self.portfolio.positions:
            g = compute_greeks(opt)
            direction = "Long" if qty > 0 else "Short"
            tag = "call" if opt.option_type == "call" else "put"
            self.tree.insert("", "end", tags=(tag,), values=(
                opt.option_type.capitalize(),
                direction,
                f"{abs(qty):.4g}",
                f"{opt.S:.2f}",
                f"{opt.K:.2f}",
                f"{opt.T:.4f}",
                f"{opt.r*100:.2f}",
                f"{opt.sigma*100:.2f}",
                f"{g['Price']*qty:+.4f}",
                f"{g['Delta']*qty:+.4f}",
                f"{g['Gamma']*qty:+.4f}",
                f"{g['Vega']*qty:+.4f}",
                f"{g['Theta']*qty:+.4f}",
                f"{g['Rho']*qty:+.4f}",
            ))
        self.tree.tag_configure("call", foreground=CALL_CLR)
        self.tree.tag_configure("put",  foreground=PUT_CLR)

    def _refresh_greeks(self):
        if not self.portfolio.positions:
            for lbl in self.greek_labels.values():
                lbl.config(text="—")
        else:
            totals = self.portfolio.total_greeks()
            for name, lbl in self.greek_labels.items():
                val = totals[name]
                lbl.config(text=f"{val:+.5f}")
        # Keep analysis tab info label in sync
        if hasattr(self, "portfolio_info_label"):
            self._update_portfolio_info()

    # ── SENSITIVITY PLOT ──────────────────────

    def _on_mode_change(self, event=None):
        mode = self.analysis_mode.get()
        if mode == "Portfolio":
            self.single_option_frame.pack_forget()
            self._update_portfolio_info()
        else:
            self.portfolio_info_label.config(text="")
            self.single_option_frame.pack(fill="x")

    def _update_portfolio_info(self):
        n = len(self.portfolio.positions)
        if n == 0:
            self.portfolio_info_label.config(
                text="⚠  Portfolio is empty.\nAdd options in the Portfolio tab.",
                fg=ACCENT2
            )
        else:
            labels = [lbl if lbl else f"pos {i+1}"
                      for i, (_, _, lbl) in enumerate(self.portfolio.positions)]
            self.portfolio_info_label.config(
                text=f"✓  {n} position{'s' if n>1 else ''}: {', '.join(labels)}",
                fg=ACCENT
            )

    def _get_analysis_opt(self, x_param, x_val):
        """Build a single Option with the specified x parameter overridden."""
        S     = float(self.analysis_entries["s_base"].get())
        K     = float(self.analysis_entries["k_base"].get())
        T     = float(self.analysis_entries["t_base"].get())
        r     = float(self.analysis_entries["r_base"].get()) / 100
        sigma = float(self.analysis_entries["v_base"].get()) / 100
        opt_type = self.analysis_type.get().lower()

        param_map = {"S": "S", "K": "K", "T": "T", "r": "r", "σ": "sigma"}
        kwargs = {"S": S, "K": K, "T": T, "r": r, "sigma": sigma, "option_type": opt_type}
        kwargs[param_map[x_param]] = x_val
        return Option(**kwargs)

    def _plot_sensitivity(self):
        try:
            x_label_full = self.x_var.get()
            x_param = x_label_full.split()[0]   # e.g. "S"
            x_min = float(self.x_min.get())
            x_max = float(self.x_max.get())
            metric = self.y_var.get()
            mode = self.analysis_mode.get()

            # Portfolio mode checks
            if mode == "Portfolio" and not self.portfolio.positions:
                messagebox.showwarning("Empty Portfolio",
                    "The portfolio has no positions.\n"
                    "Add options in the Portfolio tab, or switch to 'Single Option' mode.")
                return

            x_vals = np.linspace(x_min, x_max, 300)

            if mode == "Portfolio":
                # Compute aggregated greeks over the portfolio, varying x_param
                def get_greek(xv, greek_name):
                    g = self.portfolio.total_greeks_with_override(x_param, xv)
                    return g[greek_name]
            else:
                # Single option mode (original behaviour)
                r_val = float(self.analysis_entries["r_base"].get()) / 100
                v_val = float(self.analysis_entries["v_base"].get()) / 100

                def make_opt(xv):
                    S     = float(self.analysis_entries["s_base"].get())
                    K     = float(self.analysis_entries["k_base"].get())
                    T     = float(self.analysis_entries["t_base"].get())
                    opt_type = self.analysis_type.get().lower()
                    params = dict(S=S, K=K, T=T, r=r_val, sigma=v_val, option_type=opt_type)
                    if x_param == "S":   params["S"] = xv
                    elif x_param == "K": params["K"] = xv
                    elif x_param == "T": params["T"] = xv
                    elif x_param == "r": params["r"] = xv / 100
                    elif x_param == "σ": params["sigma"] = xv / 100
                    return Option(**params)

                greek_fn = {
                    "Price": bs_price, "Delta": bs_delta, "Gamma": bs_gamma,
                    "Vega": bs_vega, "Theta": bs_theta, "Rho": bs_rho,
                }

                def get_greek(xv, greek_name):
                    return greek_fn[greek_name](make_opt(xv))

            self.fig.clear()

            if metric == "All Greeks":
                gs = gridspec.GridSpec(2, 3, figure=self.fig,
                                       hspace=0.52, wspace=0.38,
                                       left=0.07, right=0.97,
                                       top=0.91, bottom=0.1)
                metrics = ["Price", "Delta", "Gamma", "Vega", "Theta", "Rho"]
                for i, m in enumerate(metrics):
                    ax = self.fig.add_subplot(gs[i // 3, i % 3])
                    y = [get_greek(xv, m) for xv in x_vals]
                    ax.plot(x_vals, y, color=GREEK_COLORS[m], linewidth=1.8)
                    ax.set_facecolor(BG2)
                    ax.tick_params(colors=TEXT_DIM, labelsize=7)
                    for spine in ax.spines.values():
                        spine.set_edgecolor(BORDER)
                    ax.set_title(m, color=GREEK_COLORS[m], fontsize=9,
                                 fontfamily="Courier New")
                    ax.set_xlabel(x_label_full, color=TEXT_DIM, fontsize=7,
                                  fontfamily="Courier New")
                    ax.axhline(0, color=BORDER, linewidth=0.8, linestyle="--")
                self.fig.patch.set_facecolor(BG)
                title_clr = ACCENT
            else:
                ax = self.fig.add_subplot(111)
                ax.set_facecolor(BG2)
                y = [get_greek(xv, metric) for xv in x_vals]
                color = GREEK_COLORS[metric]
                ax.plot(x_vals, y, color=color, linewidth=2.2)
                ax.fill_between(x_vals, y, alpha=0.08, color=color)
                ax.axhline(0, color=BORDER, linewidth=0.9, linestyle="--")
                ax.set_xlabel(x_label_full, color=TEXT_DIM, fontsize=9,
                              fontfamily="Courier New")
                ax.set_ylabel(metric, color=TEXT_DIM, fontsize=9,
                              fontfamily="Courier New")
                ax.tick_params(colors=TEXT_DIM, labelsize=8)
                for spine in ax.spines.values():
                    spine.set_edgecolor(BORDER)
                self.fig.patch.set_facecolor(BG)
                self.fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.12)
                title_clr = color

            # Build title depending on mode
            if mode == "Portfolio":
                n = len(self.portfolio.positions)
                mode_str = f"Portfolio ({n} pos.)"
            else:
                mode_str = self.analysis_type.get()
            self.fig.suptitle(
                f"{mode_str}  ·  {metric} vs {x_label_full}",
                color=title_clr, fontsize=10, fontfamily="Courier New",
                x=0.5, y=0.98
            )
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Plot Error", str(e))


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = OptionsApp()
    app.mainloop()
