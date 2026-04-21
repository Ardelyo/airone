"""
AirOne GUI — Easy compress & decompress desktop application.
Light cloud/sky theme.  Run: python airone_gui.py
"""

from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

# ── optional drag-and-drop ────────────────────────────────────────────────────
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    _DND = True
except ImportError:
    _DND = False


# ── Cloud / Sky palette ───────────────────────────────────────────────────────
BG        = "#EDF6FF"   # soft sky blue-white
SURFACE   = "#FFFFFF"   # pure cloud white
SURFACE2  = "#F0F7FF"   # very light sky
HEADER_BG = "#DAEEFF"   # gentle sky header
ACCENT    = "#3B9EE8"   # clear sky blue
ACCENT2   = "#6CBBF5"   # lighter horizon blue
ACCENT3   = "#1A73C9"   # deeper sky (hover)
SUCCESS   = "#2E9E60"
ERROR     = "#D94040"
WARNING   = "#C87A00"
TEXT      = "#1A2A3A"   # dark ink
TEXT_MID  = "#4A6070"
TEXT_DIM  = "#8FA8BC"
BORDER    = "#C5DFF5"
BORDER2   = "#A5C8E8"
WHITE     = "#FFFFFF"

# ── font shorthand ────────────────────────────────────────────────────────────
F         = "Segoe UI"
FC        = "Consolas"


# ─────────────────────────────────────────────────────────────────────────────
def _make_root() -> tk.Tk:
    return TkinterDnD.Tk() if _DND else tk.Tk()


# ─────────────────────────────────────────────────────────────────────────────
class AirOneGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AirOne — Intelligent Compression")
        self.root.geometry("800x660")
        self.root.minsize(680, 560)
        self.root.configure(bg=BG)

        self._file: str | None = None
        self._busy = False

        self._build_styles()
        self._build_ui()
        self._center()

    # ── setup ─────────────────────────────────────────────────────────────────
    def _center(self) -> None:
        self.root.update_idletasks()
        w, h = self.root.winfo_width(), self.root.winfo_height()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    def _build_styles(self) -> None:
        s = ttk.Style()
        s.theme_use("clam")
        s.configure(".",
                     background=BG,
                     foreground=TEXT,
                     borderwidth=0,
                     relief="flat")
        s.configure("TProgressbar",
                     troughcolor=BORDER,
                     background=ACCENT,
                     borderwidth=0,
                     thickness=5)

    # ── main UI ───────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        # ── HEADER ──────────────────────────────────────────────────────
        hdr = tk.Frame(self.root, bg=HEADER_BG)
        hdr.pack(fill="x")

        hdr_inner = tk.Frame(hdr, bg=HEADER_BG, pady=14, padx=28)
        hdr_inner.pack(fill="x")

        # cloud emoji + wordmark
        tk.Label(hdr_inner, text="☁", font=(F, 26),
                 fg=ACCENT, bg=HEADER_BG).pack(side="left", padx=(0, 6))

        title_col = tk.Frame(hdr_inner, bg=HEADER_BG)
        title_col.pack(side="left")
        tk.Label(title_col, text="AirOne", font=(F, 20, "bold"),
                 fg=ACCENT3, bg=HEADER_BG).pack(anchor="w")
        tk.Label(title_col, text="Intelligent Semantic Compression",
                 font=(F, 9), fg=TEXT_MID, bg=HEADER_BG).pack(anchor="w")

        # version pill
        pill = tk.Label(hdr_inner, text=" v1.0 - cloud edition ",
                         font=(F, 8), fg=ACCENT3, bg=BORDER,
                         relief="flat", padx=6, pady=2)
        pill.pack(side="right", padx=4)

        # thin blue separator line
        tk.Frame(self.root, bg=ACCENT2, height=1).pack(fill="x")

        # ── BODY ────────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=BG, padx=26, pady=18)
        body.pack(fill="both", expand=True)

        # ── DROP ZONE ───────────────────────────────────────────────────
        self._dz = tk.Frame(body, bg=SURFACE,
                             highlightthickness=2,
                             highlightbackground=BORDER2,
                             highlightcolor=ACCENT,
                             relief="flat", cursor="hand2")
        self._dz.pack(fill="x", pady=(0, 16))

        dz_pad = tk.Frame(self._dz, bg=SURFACE, pady=26)
        dz_pad.pack(fill="x")

        self._dz_cloud = tk.Label(dz_pad, text="⛅", font=(F, 30),
                                   bg=SURFACE, fg=ACCENT2)
        self._dz_cloud.pack()

        if _DND:
            hint = "Drag & drop a file here  —  or click to browse"
        else:
            hint = "Click to browse a file  (install tkinterdnd2 for drag-and-drop)"

        self._dz_hint = tk.Label(dz_pad, text=hint,
                                  font=(F, 10), fg=TEXT_DIM, bg=SURFACE)
        self._dz_hint.pack(pady=(6, 0))

        self._dz_name = tk.Label(dz_pad, text="",
                                  font=(F, 10, "bold"), fg=ACCENT3,
                                  bg=SURFACE, wraplength=560)
        self._dz_name.pack(pady=(4, 0))

        for w in (self._dz, dz_pad, self._dz_cloud, self._dz_hint, self._dz_name):
            w.bind("<Button-1>", lambda _e: self._browse_file())
            w.bind("<Enter>",    lambda _e: self._hover_dz(True))
            w.bind("<Leave>",    lambda _e: self._hover_dz(False))

        if _DND:
            self._dz.drop_target_register(DND_FILES)
            self._dz.dnd_bind("<<Drop>>", self._on_drop)

        # ── ACTION BUTTONS ──────────────────────────────────────────────
        btn_row = tk.Frame(body, bg=BG)
        btn_row.pack(fill="x", pady=(0, 14))
        btn_row.columnconfigure(0, weight=1)
        btn_row.columnconfigure(1, weight=1)

        self._btn_c = self._action_btn(
            btn_row, col=0,
            text="⬇  Compress", color=ACCENT, cmd=self._do_compress)
        self._btn_d = self._action_btn(
            btn_row, col=1,
            text="⬆  Decompress", color="#5C9ECC", cmd=self._do_decompress)

        # ── OUTPUT FOLDER ───────────────────────────────────────────────
        out_row = tk.Frame(body, bg=BG)
        out_row.pack(fill="x", pady=(0, 12))

        tk.Label(out_row, text="Output folder:", font=(F, 9),
                 fg=TEXT_MID, bg=BG).pack(side="left")

        self._out_var = tk.StringVar(value="Same folder as input")
        out_entry = tk.Entry(out_row, textvariable=self._out_var,
                              font=(F, 9), bg=SURFACE, fg=TEXT,
                              insertbackground=ACCENT, relief="flat",
                              highlightthickness=1,
                              highlightbackground=BORDER,
                              highlightcolor=ACCENT)
        out_entry.pack(side="left", fill="x", expand=True, padx=(8, 6), ipady=5)

        self._small_btn(out_row, "Browse…", self._browse_outdir)

        # ── PROGRESS ────────────────────────────────────────────────────
        self._prog = ttk.Progressbar(body, mode="indeterminate")
        self._prog.pack(fill="x", pady=(0, 12))

        # ── LOG PANEL ───────────────────────────────────────────────────
        log_hdr = tk.Frame(body, bg=BG)
        log_hdr.pack(fill="x")

        tk.Label(log_hdr, text="☁ Activity Log", font=(F, 9, "bold"),
                 fg=TEXT_MID, bg=BG).pack(side="left")
        self._small_btn(log_hdr, "Clear", self._clear_log, side="right")

        log_wrap = tk.Frame(body, bg=SURFACE,
                             highlightthickness=1, highlightbackground=BORDER)
        log_wrap.pack(fill="both", expand=True, pady=(4, 0))

        self._log = tk.Text(log_wrap, font=(FC, 9), bg=SURFACE, fg=TEXT,
                             relief="flat", state="disabled",
                             insertbackground=ACCENT, wrap="word",
                             padx=10, pady=8,
                             selectbackground=ACCENT2,
                             selectforeground=WHITE)
        scroll = tk.Scrollbar(log_wrap, command=self._log.yview,
                               bg=BG, troughcolor=BG,
                               activebackground=BORDER, relief="flat", width=10)
        self._log.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self._log.pack(side="left", fill="both", expand=True)

        # tags
        self._log.tag_config("ok",   foreground=SUCCESS)
        self._log.tag_config("err",  foreground=ERROR)
        self._log.tag_config("warn", foreground=WARNING)
        self._log.tag_config("info", foreground=ACCENT3)
        self._log.tag_config("dim",  foreground=TEXT_DIM)
        self._log.tag_config("head", foreground=TEXT_MID,
                             font=(FC, 9, "bold"))

        self._log_w("Welcome to AirOne ☁\n"
                    "Select a file above, then click Compress or Decompress.\n",
                    "dim")

    # ── widget builders ───────────────────────────────────────────────────────
    def _action_btn(self, parent, col, text, color, cmd) -> tk.Button:
        """Bordered pill-style action button."""
        outer = tk.Frame(parent, bg=color, padx=2, pady=2,
                         highlightthickness=0)
        pad = (0, 8) if col == 0 else (0, 0)
        outer.grid(row=0, column=col, padx=pad, sticky="ew")

        btn = tk.Button(
            outer, text=text, bg=SURFACE, fg=color,
            font=(F, 11, "bold"),
            activebackground=color, activeforeground=WHITE,
            relief="flat", cursor="hand2", bd=0, pady=11,
            command=cmd
        )
        btn.pack(fill="both")
        btn.bind("<Enter>", lambda _e, b=btn, c=color: b.config(bg=c, fg=WHITE))
        btn.bind("<Leave>", lambda _e, b=btn, c=color: b.config(bg=SURFACE, fg=c))
        return btn

    def _small_btn(self, parent, text, cmd, side="left") -> tk.Button:
        b = tk.Button(parent, text=text, bg=SURFACE, fg=TEXT_MID,
                       font=(F, 8), relief="flat", bd=0,
                       activebackground=BORDER, activeforeground=TEXT,
                       cursor="hand2", padx=7, pady=4, command=cmd,
                       highlightthickness=1, highlightbackground=BORDER)
        b.pack(side=side, padx=(4, 0))
        return b

    # ── drop zone interactivity ───────────────────────────────────────────────
    def _hover_dz(self, on: bool) -> None:
        c = ACCENT if on else BORDER2
        self._dz.config(highlightbackground=c, highlightcolor=c, bg=SURFACE2 if on else SURFACE)
        for w in self._dz.winfo_children():
            try:
                w.config(bg=SURFACE2 if on else SURFACE)
                for ww in w.winfo_children():
                    ww.config(bg=SURFACE2 if on else SURFACE)
            except Exception:
                pass

    def _set_file(self, path: str) -> None:
        self._file = path
        name = Path(path).name
        self._dz_cloud.config(text="📄", fg=ACCENT3)
        self._dz_hint.config(text="File selected:", fg=TEXT_MID)
        self._dz_name.config(text=name)
        # auto-set output to same folder
        folder = str(Path(path).parent)
        if self._out_var.get() in ("Same folder as input", folder):
            self._out_var.set(folder)

    def _on_drop(self, event) -> None:
        raw = event.data.strip()
        if raw.startswith("{"):
            raw = raw[1:raw.rfind("}")]
        path = raw.split("} {")[0] if "} {" in raw else raw
        if os.path.isfile(path):
            self._set_file(path)
        else:
            self._log_w(f"Dropped item is not a file: {path}\n", "warn")

    def _browse_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select file",
            filetypes=[("All files", "*.*"), ("AirOne archives", "*.air")]
        )
        if path:
            self._set_file(path)

    def _browse_outdir(self) -> None:
        d = filedialog.askdirectory(title="Choose output folder")
        if d:
            self._out_var.set(d)

    # ── log helpers ───────────────────────────────────────────────────────────
    def _log_w(self, msg: str, tag: str = "") -> None:
        self._log.config(state="normal")
        self._log.insert("end", msg, tag)
        self._log.see("end")
        self._log.config(state="disabled")

    def _clear_log(self) -> None:
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")

    # ── lock UI during work ───────────────────────────────────────────────────
    def _lock(self, busy: bool) -> None:
        self._busy = busy
        s = "disabled" if busy else "normal"
        self._btn_c.config(state=s)
        self._btn_d.config(state=s)
        if busy:
            self._prog.start(8)
        else:
            self._prog.stop()
            self._prog["value"] = 0

    # ── resolve output path ───────────────────────────────────────────────────
    def _out_path(self, src: str, ext: str) -> str:
        """Build output path; ext='' means strip .air, ext='.air' means append."""
        raw = self._out_var.get().strip()
        folder = raw if (raw and raw != "Same folder as input") else str(Path(src).parent)
        if ext:
            name = Path(src).name + ext
        else:
            stem = Path(src).stem if src.endswith(".air") else Path(src).name
            name = stem
        return str(Path(folder) / name)

    # ── COMPRESS ──────────────────────────────────────────────────────────────
    def _do_compress(self) -> None:
        if self._busy:
            return
        if not self._file:
            messagebox.showwarning("No file selected",
                                    "Please select a file using the drop zone above.")
            return

        src = self._file
        dst = self._out_path(src, ".air")

        self._lock(True)
        self._log_w("\n── Compress ─────────────────────────────────\n", "head")
        self._log_w(f"  Input :  {src}\n", "info")
        self._log_w(f"  Output:  {dst}\n", "info")

        def _run() -> None:
            try:
                from airone.api import AirOne
                result = AirOne().compress_file(src, dst)
                self.root.after(0, self._compress_done, result, dst)
            except Exception as exc:          # catch everything so UI never crashes
                self.root.after(0, self._show_error, str(exc))

        threading.Thread(target=_run, daemon=True).start()

    def _compress_done(self, result, dst: str) -> None:
        self._lock(False)
        self._log_w("\n  ✔  Compression complete!\n", "ok")
        try:
            orig = getattr(result, "original_size", None)
            comp = getattr(result, "compressed_size", None)
            ratio = getattr(result, "ratio", None)
            strat = getattr(result, "strategy_name", "—")
            if orig is not None:
                self._log_w(f"  Original  : {orig:>14,} bytes\n")
            if comp is not None:
                self._log_w(f"  Compressed: {comp:>14,} bytes\n")
            if ratio is not None:
                tag = "ok" if ratio >= 1.0 else "warn"
                self._log_w(f"  Ratio     : {ratio:>14.3f}×\n", tag)
            self._log_w(f"  Strategy  : {strat}\n", "dim")
        except Exception:
            pass
        self._log_w(f"  Saved to  : {dst}\n", "dim")
        self._log_w("─────────────────────────────────────────────\n", "head")
        self._set_file(dst)   # select output for quick decompress test

    # ── DECOMPRESS ────────────────────────────────────────────────────────────
    def _do_decompress(self) -> None:
        if self._busy:
            return
        if not self._file:
            messagebox.showwarning("No file selected",
                                    "Please select a file using the drop zone above.")
            return
        if not self._file.endswith(".air"):
            messagebox.showerror("Wrong file type",
                                  "Decompression needs an .air file.\n"
                                  "Please select a file that ends with .air")
            return

        src = self._file
        dst = self._out_path(src, "")
        # avoid overwriting the same path
        if os.path.normcase(dst) == os.path.normcase(src):
            dst = str(Path(src).with_suffix(".restored"))

        self._lock(True)
        self._log_w("\n── Decompress ───────────────────────────────\n", "head")
        self._log_w(f"  Input :  {src}\n", "info")
        self._log_w(f"  Output:  {dst}\n", "info")

        def _run() -> None:
            try:
                from airone.api import AirOne
                size = AirOne().decompress_file(src, dst)
                self.root.after(0, self._decompress_done, size, dst)
            except Exception as exc:
                self.root.after(0, self._show_error, str(exc))

        threading.Thread(target=_run, daemon=True).start()

    def _decompress_done(self, size, dst: str) -> None:
        self._lock(False)
        self._log_w("\n  ✔  Decompression complete!\n", "ok")
        try:
            if isinstance(size, int):
                self._log_w(f"  Restored  : {size:>14,} bytes\n")
        except Exception:
            pass
        self._log_w(f"  Saved to  : {dst}\n", "dim")
        self._log_w("─────────────────────────────────────────────\n", "head")
        self._set_file(dst)

    # ── error handler ─────────────────────────────────────────────────────────
    def _show_error(self, msg: str) -> None:
        self._lock(False)
        self._log_w(f"\n  ✘  Error: {msg}\n", "err")
        self._log_w("─────────────────────────────────────────────\n", "head")
        messagebox.showerror("AirOne Error", msg)


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # make sure the src package is importable when run directly
    here = str(Path(__file__).parent)
    if here not in sys.path:
        sys.path.insert(0, here)

    root = _make_root()
    AirOneGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
