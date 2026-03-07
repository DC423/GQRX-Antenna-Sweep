#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "numpy", "rich"]
# ///
"""
Antenna Sweep & Comparison Tool

Dual-backend antenna comparison tool supporting GQRX (any radio) and
hackrf_sweep (HackRF-specific fast sweeps). Sweeps spectrum, measures signal
strength per frequency bin, and optionally subtracts a baseline sweep to
isolate antenna gain from ambient conditions.

The compare subcommand loads all sweep CSVs in the current directory and
produces:
  - P90-based per-band ranking with activity detection (filters out quiet
    noise-floor-only bands) and adaptive significance thresholds (pooled SE)
  - Relative advantage table (each antenna vs group median, per active band)
  - Diverging heatmap PNG of advantages for visual comparison
  - Coverage ranking with greedy set-cover ("if you can only bring N antennas")
  - Overlay plot with per-band recommendation panel

Workflow:
  1. Sweep a baseline (50-ohm terminator or reference antenna):
     uv run gqrx_sweep.py sweep --antenna baseline --start 1M --end 6G --backend hackrf --sweeps 100

  2. Sweep each antenna under test:
     uv run gqrx_sweep.py sweep --antenna 01 --start 1M --end 6G --backend hackrf --sweeps 100
     uv run gqrx_sweep.py sweep --antenna 02 --start 1M --end 6G --backend hackrf --sweeps 100
     ...

  3. Compare all sweeps:
     uv run gqrx_sweep.py compare
     uv run gqrx_sweep.py compare --top-n 2
"""

import argparse
import re
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Frequency bands for per-band scoring
# ---------------------------------------------------------------------------

FREQUENCY_BANDS: list[tuple[str, float, float]] = [
    ("VHF", 30e6, 300e6),
    ("FM Broadcast", 88e6, 108e6),
    ("Airband", 108e6, 137e6),
    ("VHF Marine", 156e6, 174e6),
    ("UHF", 300e6, 3000e6),
    ("70cm Ham", 420e6, 450e6),
    ("GSM 850", 824e6, 894e6),
    ("GSM 900", 880e6, 960e6),
    ("GSM 1800", 1710e6, 1880e6),
    ("GSM 1900", 1850e6, 1990e6),
    ("LoRa (US)", 902e6, 928e6),
    ("LoRa (EU)", 863e6, 870e6),
    ("WiFi 2.4 GHz", 2400e6, 2500e6),
    ("BLE", 2400e6, 2484e6),
    ("WiFi 5 GHz", 5150e6, 5850e6),
    ("WiFi 6E", 5925e6, 7125e6),
]

BAND_COLORS: dict[str, str] = {
    "VHF": "#FF6B6B",
    "FM Broadcast": "#4ECDC4",
    "Airband": "#45B7D1",
    "VHF Marine": "#96CEB4",
    "UHF": "#FFEAA7",
    "70cm Ham": "#DDA0DD",
    "GSM 850": "#98D8C8",
    "GSM 900": "#F7DC6F",
    "GSM 1800": "#BB8FCE",
    "GSM 1900": "#85C1E9",
    "LoRa (US)": "#F8C471",
    "LoRa (EU)": "#82E0AA",
    "WiFi 2.4 GHz": "#F1948A",
    "BLE": "#AED6F1",
    "WiFi 5 GHz": "#A3E4D7",
    "WiFi 6E": "#FAD7A0",
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def parse_frequency(freq_str: str) -> int:
    """Parse a frequency string with optional suffix (K, M, G)."""
    freq_str = freq_str.strip().upper()
    multipliers = {"K": 1e3, "M": 1e6, "G": 1e9}
    for suffix, mult in multipliers.items():
        if freq_str.endswith(suffix):
            return int(float(freq_str[:-1]) * mult)
    return int(float(freq_str))


def format_freq(freq_hz: float) -> str:
    """Format a frequency in Hz to a human-readable string."""
    if freq_hz >= 1e9:
        return f"{freq_hz / 1e9:.3f} GHz"
    elif freq_hz >= 1e6:
        return f"{freq_hz / 1e6:.3f} MHz"
    elif freq_hz >= 1e3:
        return f"{freq_hz / 1e3:.1f} kHz"
    return f"{freq_hz:.0f} Hz"


def sanitize_filename(name: str) -> str:
    """Sanitize a string for safe use in filenames."""
    return re.sub(r"[^\w\-]", "_", name).strip("_").lower()


def estimate_noise_floor(power_values: list[float]) -> float:
    """Estimate noise floor as the 10th percentile of power readings."""
    return float(np.percentile(power_values, 10))


def interpolate_baseline(
    sweep_data: list[tuple[float, float]],
    baseline_data: list[tuple[float, float]],
) -> np.ndarray:
    """Interpolate baseline power values to match sweep frequencies."""
    baseline_freqs = np.array([d[0] for d in baseline_data])
    baseline_powers = np.array([d[1] for d in baseline_data])
    sweep_freqs = np.array([d[0] for d in sweep_data])
    return np.interp(sweep_freqs, baseline_freqs, baseline_powers)


# ---------------------------------------------------------------------------
# Per-band scoring
# ---------------------------------------------------------------------------


def compute_band_metrics(
    frequencies: np.ndarray,
    powers: np.ndarray,
    noise_floor: float,
    band_start: float,
    band_end: float,
) -> tuple[float, float, float, float] | None:
    """Compute actual dBm metrics for a frequency band.

    Returns (mean_dbm, p90_dbm, peak_dbm, std_dbm) or None if no data in band.
    P90 (90th percentile) is the primary comparison metric -- it captures
    actual signal reception while ignoring the noise-floor bins that
    dominate the mean. std_dbm measures within-band variance: low (~0.8 dB)
    indicates pure noise floor, high (5-10+ dB) indicates real signal activity.
    """
    mask = (frequencies >= band_start) & (frequencies <= band_end)
    if not np.any(mask):
        return None

    band_powers = powers[mask]
    mean_dbm = float(np.mean(band_powers))
    p90_dbm = float(np.percentile(band_powers, 90))
    peak_dbm = float(np.max(band_powers))
    std_dbm = float(np.std(band_powers))
    return mean_dbm, p90_dbm, peak_dbm, std_dbm


def _get_overlapping_bands(
    frequencies: np.ndarray,
) -> list[tuple[str, float, float]]:
    """Return frequency bands that overlap with the measured range."""
    freq_min = float(frequencies.min())
    freq_max = float(frequencies.max())
    return [
        (name, start, end)
        for name, start, end in FREQUENCY_BANDS
        if start <= freq_max and end >= freq_min
    ]


def _draw_band_overlays(ax: plt.Axes, freq_min_hz: float, freq_max_hz: float) -> None:
    """Draw colored transparent band regions and labels on a frequency plot."""
    freq_min_mhz = freq_min_hz / 1e6
    freq_max_mhz = freq_max_hz / 1e6

    for band_name, band_start, band_end in FREQUENCY_BANDS:
        start_mhz = band_start / 1e6
        end_mhz = band_end / 1e6
        if start_mhz > freq_max_mhz or end_mhz < freq_min_mhz:
            continue
        draw_start = max(start_mhz, freq_min_mhz)
        draw_end = min(end_mhz, freq_max_mhz)
        color = BAND_COLORS.get(band_name, "#CCCCCC")
        ax.axvspan(draw_start, draw_end, alpha=0.10, color=color, zorder=0)
        center_x = (draw_start + draw_end) / 2
        ax.text(
            center_x,
            0.97,
            band_name,
            ha="center",
            va="top",
            fontsize=7,
            rotation=90,
            alpha=0.7,
            color=color,
            fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )


# ---------------------------------------------------------------------------
# GQRX backend (default -- works with any radio via TCP remote control)
# ---------------------------------------------------------------------------


class GqrxConnection:
    """Manages the TCP socket connection to GQRX's remote control interface."""

    def __init__(
        self, host: str = "127.0.0.1", port: int = 7356, timeout: int = 5
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: socket.socket | None = None

    def connect(self) -> bool:
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect((self.host, self.port))
            console.print(
                f"[green]Connected to GQRX at {self.host}:{self.port}[/green]"
            )
            return True
        except socket.error as e:
            console.print(f"[red]Connection failed: {e}[/red]")
            return False

    def disconnect(self) -> None:
        if self.sock:
            try:
                self.sock.close()
            except socket.error:
                pass
            self.sock = None
            console.print("[green]Disconnected from GQRX[/green]")

    def _send_command(self, command: str) -> str:
        if not self.sock:
            raise ConnectionError("Not connected to GQRX")
        try:
            self.sock.sendall((command + "\n").encode())
            response = b""
            while True:
                chunk = self.sock.recv(4096)
                response += chunk
                if b"\n" in chunk:
                    break
            return response.decode().strip()
        except socket.error as e:
            raise ConnectionError(f"Communication error: {e}")

    def set_frequency(self, freq_hz: int) -> bool:
        resp = self._send_command(f"F {int(freq_hz)}")
        return "RPRT 0" in resp

    def get_signal_strength(self) -> float | None:
        resp = self._send_command("l STRENGTH")
        try:
            return float(resp.split("\n")[0])
        except (ValueError, IndexError):
            return None

    def set_mode(self, mode: str, bandwidth: int = 0) -> bool:
        resp = self._send_command(f"M {mode} {bandwidth}")
        return "RPRT 0" in resp


class GqrxSweepBackend:
    """GQRX-based sweep backend. Step-tunes through the range via TCP."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7356,
        mode: str | None = None,
        dwell_time: float = 0.5,
        samples_per_step: int = 3,
    ):
        self.host = host
        self.port = port
        self.mode = mode
        self.dwell_time = dwell_time
        self.samples_per_step = samples_per_step

    def sweep(
        self,
        start_hz: int,
        end_hz: int,
        step_hz: int = 100_000,
    ) -> list[tuple[float, float]]:
        conn = GqrxConnection(host=self.host, port=self.port)
        if not conn.connect():
            console.print("[red]Could not connect to GQRX. Ensure:[/red]")
            console.print("  1. GQRX is running")
            console.print(
                "  2. Remote Control is enabled (Tools > Remote Control)"
            )
            console.print(f"  3. It's listening on {self.host}:{self.port}")
            sys.exit(1)

        try:
            if self.mode:
                if conn.set_mode(self.mode.upper()):
                    console.print(
                        f"[green]Mode set to {self.mode.upper()}[/green]"
                    )
                else:
                    console.print(
                        f"[yellow]Failed to set mode to {self.mode}[/yellow]"
                    )

            results: list[tuple[float, float]] = []
            num_steps = int((end_hz - start_hz) / step_hz) + 1

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Sweeping (GQRX)...", total=num_steps
                )
                for i in range(num_steps):
                    freq = start_hz + i * step_hz
                    if freq > end_hz:
                        break
                    if not conn.set_frequency(int(freq)):
                        progress.advance(task)
                        continue

                    time.sleep(self.dwell_time)
                    readings: list[float] = []
                    for _ in range(self.samples_per_step):
                        strength = conn.get_signal_strength()
                        if strength is not None:
                            readings.append(strength)
                        time.sleep(0.05)

                    if readings:
                        avg_db = sum(readings) / len(readings)
                        results.append((float(freq), avg_db))
                    progress.advance(task)

            return results
        finally:
            conn.disconnect()

    def get_metadata(self) -> dict[str, str]:
        meta: dict[str, str] = {
            "backend": "gqrx",
            "host": self.host,
            "port": str(self.port),
            "dwell_time": str(self.dwell_time),
            "samples_per_step": str(self.samples_per_step),
        }
        if self.mode:
            meta["mode"] = self.mode
        return meta


# ---------------------------------------------------------------------------
# hackrf_sweep backend (fast, HackRF-specific, activated via --backend hackrf)
# ---------------------------------------------------------------------------


class HackRFSweepBackend:
    """hackrf_sweep backend for fast HackRF One spectrum sweeps.

    Shells out to the ``hackrf_sweep`` binary and parses its CSV output.
    """

    def __init__(
        self,
        lna_gain: int = 16,
        vga_gain: int = 22,
        bin_width: int = 100_000,
        amp_enable: bool = False,
        num_sweeps: int = 1,
    ):
        self.lna_gain = lna_gain
        self.vga_gain = vga_gain
        self.bin_width = bin_width
        self.amp_enable = amp_enable
        self.num_sweeps = num_sweeps

    def sweep(
        self,
        start_hz: int,
        end_hz: int,
        step_hz: int = 100_000,
    ) -> list[tuple[float, float]]:
        start_mhz = int(start_hz / 1e6)
        end_mhz = int(end_hz / 1e6) + 1  # Round up to include full range

        cmd = [
            "hackrf_sweep",
            "-f", f"{start_mhz}:{end_mhz}",
            "-l", str(self.lna_gain),
            "-g", str(self.vga_gain),
            "-w", str(self.bin_width),
            "-N", str(self.num_sweeps),
            "-a", "1" if self.amp_enable else "0",
            "-r", "/dev/stdout",
        ]

        console.print(f"[dim]Running: {' '.join(cmd)}[/dim]")
        if self.num_sweeps > 1:
            console.print(
                f"[dim]Averaging {self.num_sweeps} sweeps for noise reduction[/dim]"
            )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            progress.add_task(
                f"Sweeping (hackrf_sweep, {self.num_sweeps}x)...", total=None
            )
            try:
                timeout = 120 * max(self.num_sweeps, 1)
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout
                )
            except FileNotFoundError:
                console.print(
                    "[red]hackrf_sweep not found. Install it or check PATH.[/red]"
                )
                sys.exit(1)
            except subprocess.TimeoutExpired:
                console.print(
                    f"[red]hackrf_sweep timed out after {timeout}s.[/red]"
                )
                sys.exit(1)

        # hackrf_sweep often exits non-zero on USB hiccups even after
        # completing most sweeps.  Try to salvage whatever data we got.
        raw = self._parse_output(result.stdout, start_hz, end_hz)

        if result.returncode != 0:
            if raw:
                console.print(
                    f"[yellow]hackrf_sweep exited with code {result.returncode} "
                    f"but captured {len(raw)} data points — using partial data[/yellow]"
                )
            else:
                console.print(
                    f"[red]hackrf_sweep failed (exit {result.returncode}):[/red]"
                )
                if result.stderr:
                    console.print(result.stderr.strip())
                sys.exit(1)

        if self.num_sweeps > 1:
            return self._average_bins(raw)
        return raw

    def _parse_output(
        self, output: str, start_hz: int, end_hz: int
    ) -> list[tuple[float, float]]:
        """Parse hackrf_sweep CSV output into (freq_hz, power_dbm) tuples.

        hackrf_sweep output format per line:
          date, time, hz_low, hz_high, hz_bin_width, num_samples, dB, dB, ...
        Each line's dB values map to frequency bins from hz_low to hz_high.
        With -N >1, multiple sweeps produce repeated frequency bins that
        are averaged by _average_bins().
        """
        results: list[tuple[float, float]] = []
        for line in output.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            try:
                hz_low = float(parts[2].strip())
                hz_high = float(parts[3].strip())
                hz_bin_width = float(parts[4].strip())
                num_bins = int((hz_high - hz_low) / hz_bin_width)
                db_values = [
                    float(p.strip()) for p in parts[6 : 6 + num_bins]
                ]
                for i, db in enumerate(db_values):
                    freq = hz_low + i * hz_bin_width + hz_bin_width / 2
                    if start_hz <= freq <= end_hz:
                        results.append((freq, db))
            except (ValueError, IndexError):
                continue

        results.sort(key=lambda x: x[0])
        return results

    @staticmethod
    def _average_bins(
        raw: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Average power readings that share the same frequency bin.

        When hackrf_sweep runs multiple passes (-N), each bin appears
        multiple times. We average the dBm values per unique frequency.
        """
        from collections import defaultdict

        accumulator: dict[float, list[float]] = defaultdict(list)
        for freq, power in raw:
            accumulator[freq].append(power)
        averaged = [
            (freq, sum(powers) / len(powers))
            for freq, powers in sorted(accumulator.items())
        ]
        return averaged

    def get_metadata(self) -> dict[str, str]:
        return {
            "backend": "hackrf_sweep",
            "lna_gain": str(self.lna_gain),
            "vga_gain": str(self.vga_gain),
            "bin_width": str(self.bin_width),
            "amp_enable": str(self.amp_enable),
            "num_sweeps": str(self.num_sweeps),
        }


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------


BASELINE_CSV = Path("baseline.csv")


def save_sweep_csv(
    results: list[tuple[float, float]],
    antenna_name: str,
    start_hz: int,
    end_hz: int,
    backend_metadata: dict[str, str],
    noise_floor: float,
) -> Path:
    """Save sweep results with metadata comment header."""
    if antenna_name.lower() == "baseline":
        path = BASELINE_CSV
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = sanitize_filename(antenna_name)
        path = Path(f"sweep_{safe_name}_{timestamp}.csv")

    with path.open("w", newline="") as f:
        f.write(f"# antenna: {antenna_name}\n")
        f.write(
            f"# date: {datetime.now().isoformat(timespec='seconds')}\n"
        )
        f.write(f"# start_hz: {start_hz}\n")
        f.write(f"# end_hz: {end_hz}\n")
        for key, val in backend_metadata.items():
            f.write(f"# {key}: {val}\n")
        f.write(f"# noise_floor_dbm: {noise_floor:.1f}\n")
        f.write("frequency_hz,power_dbm\n")
        for freq, power in results:
            f.write(f"{freq:.0f},{power:.1f}\n")

    return path


def load_sweep_csv(
    path: Path,
) -> tuple[dict[str, str], list[tuple[float, float]]]:
    """Load a sweep CSV, returning (metadata_dict, [(freq, power), ...])."""
    metadata: dict[str, str] = {}
    data: list[tuple[float, float]] = []

    with path.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                content = line[1:].strip()
                if ":" in content:
                    key, _, val = content.partition(":")
                    metadata[key.strip()] = val.strip()
            elif line and not line.startswith("frequency_hz"):
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        data.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        continue

    return metadata, data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_single_sweep(
    results: list[tuple[float, float]],
    antenna_name: str,
    start_hz: int,
    end_hz: int,
    noise_floor: float,
    output_path: Path,
    baseline_subtracted: bool = False,
) -> None:
    """Generate a single-antenna sweep plot."""
    freqs_mhz = np.array([r[0] for r in results]) / 1e6
    powers = np.array([r[1] for r in results])

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(freqs_mhz, powers, linewidth=0.8, color="#2196F3")

    ref_label = "Baseline (0 dB)" if baseline_subtracted else f"Noise floor: {noise_floor:.1f} dBm"
    ax.axhline(
        y=noise_floor,
        color="#FF5722",
        linestyle="--",
        linewidth=0.7,
        label=ref_label,
    )
    ax.fill_between(freqs_mhz, noise_floor, powers, alpha=0.15, color="#2196F3")

    ax.set_xlabel("Frequency (MHz)")
    ylabel = "Gain above baseline (dB)" if baseline_subtracted else "Power (dBm)"
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{antenna_name} -- {format_freq(start_hz)} to {format_freq(end_hz)}"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    _draw_band_overlays(ax, start_hz, end_hz)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_comparison(
    sweeps: list[tuple[str, list[tuple[float, float]]]],
    output_path: Path,
    baseline_data: list[tuple[float, float]] | None = None,
    best_per_band: dict[str, str] | None = None,
) -> None:
    """Generate a comparison overlay plot for multiple antennas.

    If best_per_band is provided, a recommendation panel is drawn below the plot.
    """
    # Show all active bands (exclude "--" which means no activity)
    active_results: dict[str, str] = {}
    if best_per_band:
        active_results = {
            band: ant for band, ant in best_per_band.items()
            if ant != "--"
        }

    has_recommendations = len(active_results) > 0
    if has_recommendations:
        fig = plt.figure(figsize=(16, 9), layout="constrained")
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.25)
        ax = fig.add_subplot(gs[0])
        ax_rec = fig.add_subplot(gs[1])
    else:
        fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for i, (name, data) in enumerate(sweeps):
        freqs_mhz = np.array([d[0] for d in data]) / 1e6
        powers = np.array([d[1] for d in data])
        if baseline_data:
            baseline_powers = interpolate_baseline(data, baseline_data)
            powers = powers - baseline_powers
        color = colors[i % len(colors)]
        ax.plot(freqs_mhz, powers, linewidth=0.9, label=name, color=color)

    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel(
        "Power relative to baseline (dB)" if baseline_data else "Power (dBm)"
    )
    ax.set_title("Antenna Comparison")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    all_freqs = [f for _, data in sweeps for f, _ in data]
    if all_freqs:
        _draw_band_overlays(ax, min(all_freqs), max(all_freqs))

    # Recommendation panel (all active bands)
    if has_recommendations:
        ax_rec.axis("off")
        ax_rec.set_title(
            "Best Antenna Per Active Band", fontsize=11, fontweight="bold", loc="left"
        )

        items = list(active_results.items())
        cols = 4
        n_rows = (len(items) + cols - 1) // cols
        for idx, (band, antenna) in enumerate(items):
            col = idx % cols
            row = idx // cols
            x = 0.01 + col * (1.0 / cols)
            y = 0.82 - row * (0.82 / max(n_rows, 1))
            band_color = BAND_COLORS.get(band, "#666666")
            ax_rec.text(
                x, y, f"{band}: {antenna}",
                fontsize=9, fontweight="bold", color=band_color,
                transform=ax_rec.transAxes, va="top",
            )

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis & reporting
# ---------------------------------------------------------------------------


def compute_band_metrics_all(
    frequencies: np.ndarray,
    powers: np.ndarray,
    noise_floor: float,
) -> list[tuple[str, float, float, float, float, float, float]]:
    """Compute per-band metrics for a single antenna.

    Returns sorted list of (band_name, start_mhz, end_mhz, mean_dbm, p90_dbm, peak_dbm, std_dbm)
    sorted by P90 descending (strongest bands first).
    """
    rows: list[tuple[str, float, float, float, float, float, float]] = []
    for band_name, band_start, band_end in _get_overlapping_bands(frequencies):
        metrics = compute_band_metrics(
            frequencies, powers, noise_floor, band_start, band_end
        )
        if metrics is not None:
            mean_dbm, p90_dbm, peak_dbm, std_dbm = metrics
            rows.append((
                band_name,
                band_start / 1e6,
                band_end / 1e6,
                mean_dbm,
                p90_dbm,
                peak_dbm,
                std_dbm,
            ))
    rows.sort(key=lambda r: r[4], reverse=True)
    return rows


def print_band_metrics(
    antenna_name: str,
    band_data: list[tuple[str, float, float, float, float, float, float]],
) -> None:
    """Print per-band metrics for a single antenna as a Rich table."""
    if not band_data:
        console.print(
            "[dim]No known bands overlap with this sweep range.[/dim]"
        )
        return

    table = Table(title=f"Band Metrics: {antenna_name}")
    table.add_column("Band", style="cyan")
    table.add_column("Start (MHz)", justify="right")
    table.add_column("End (MHz)", justify="right")
    table.add_column("Mean (dBm)", justify="right")
    table.add_column("P90 (dBm)", justify="right", style="green")
    table.add_column("Peak (dBm)", justify="right", style="bold")
    table.add_column("Std (dB)", justify="right", style="dim")

    for band_name, start_mhz, end_mhz, mean_dbm, p90_dbm, peak_dbm, std_dbm in band_data:
        table.add_row(
            band_name,
            f"{start_mhz:.1f}",
            f"{end_mhz:.1f}",
            f"{mean_dbm:.1f}",
            f"{p90_dbm:.1f}",
            f"{peak_dbm:.1f}",
            f"{std_dbm:.1f}",
        )

    console.print(table)


def save_sweep_report_markdown(
    antenna_name: str,
    summary_rows: list[tuple[str, str]],
    band_data: list[tuple[str, float, float, float, float, float, float]],
    csv_path: Path,
    png_path: Path | None,
    output_path: Path,
) -> None:
    """Save the full sweep report (summary + band metrics) as markdown."""
    lines = [
        f"# Sweep Report: {antenna_name}",
        "",
        "## Files",
        "",
        f"- CSV: `{csv_path}`",
    ]
    if png_path is not None:
        lines.append(f"- Plot: `{png_path}`")
    lines += [
        "",
        "## Sweep Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for metric, value in summary_rows:
        lines.append(f"| {metric} | {value} |")
    lines += [
        "",
        "## Band Metrics",
        "",
        "| Band | Start (MHz) | End (MHz) | Mean (dBm) | P90 (dBm) | Peak (dBm) | Std (dB) |",
        "|------|-------------|-----------|------------|-----------|------------|----------|",
    ]
    for band_name, start_mhz, end_mhz, mean_dbm, p90_dbm, peak_dbm, std_dbm in band_data:
        lines.append(
            f"| {band_name} | {start_mhz:.1f} | {end_mhz:.1f} "
            f"| {mean_dbm:.1f} | {p90_dbm:.1f} | {peak_dbm:.1f} | {std_dbm:.1f} |"
        )
    lines.append("")
    output_path.write_text("\n".join(lines))


BAND_ACTIVITY_STD_THRESHOLD_DB = 2.0  # Bands below this std are noise-floor-only
BETWEEN_ANTENNA_SPREAD_THRESHOLD_DB = 1.5  # P90 spread across antennas that indicates real differentiation


def print_comparison_ranking(
    sweeps: list[tuple[str, dict[str, str], list[tuple[float, float]]]],
    baseline_data: list[tuple[float, float]] | None = None,
    markdown_path: Path | None = None,
) -> tuple[dict[str, str], dict[str, dict[str, float]], dict[str, dict[str, float]], list[str]]:
    """Print per-band ranking table comparing all antennas using P90 power.

    Returns:
        best_per_band: dict mapping band name -> best antenna name
        antenna_p90: dict[antenna][band] -> p90 value
        antenna_std: dict[antenna][band] -> std value
        active_bands: list of band names with real signal activity
    """
    all_freqs: list[float] = []
    for _, _, data in sweeps:
        all_freqs.extend(d[0] for d in data)
    if not all_freqs:
        console.print("[yellow]No data to compare.[/yellow]")
        return {}, {}, {}, []

    overlapping = _get_overlapping_bands(np.array(all_freqs))
    if not overlapping:
        console.print(
            "[dim]No known bands overlap with sweep ranges.[/dim]"
        )
        return {}, {}, {}, []

    band_mhz: dict[str, tuple[float, float]] = {
        name: (start / 1e6, end / 1e6)
        for name, start, end in overlapping
    }

    # Compute P90 + std + n_bins per antenna per band
    antenna_names = [name for name, _, _ in sweeps]
    antenna_p90: dict[str, dict[str, float]] = {}
    antenna_std: dict[str, dict[str, float]] = {}
    antenna_n_bins: dict[str, dict[str, int]] = {}
    for antenna_name, _metadata, data in sweeps:
        freqs = np.array([d[0] for d in data])
        powers = np.array([d[1] for d in data])
        if baseline_data:
            powers = powers - interpolate_baseline(data, baseline_data)
        noise_floor = estimate_noise_floor(powers.tolist())
        p90s: dict[str, float] = {}
        stds: dict[str, float] = {}
        n_bins: dict[str, int] = {}
        for band_name, band_start, band_end in overlapping:
            metrics = compute_band_metrics(
                freqs, powers, noise_floor, band_start, band_end
            )
            if metrics is not None:
                _mean, p90, _peak, std = metrics
                p90s[band_name] = p90
                stds[band_name] = std
                mask = (freqs >= band_start) & (freqs <= band_end)
                n_bins[band_name] = int(np.sum(mask))
        antenna_p90[antenna_name] = p90s
        antenna_std[antenna_name] = stds
        antenna_n_bins[antenna_name] = n_bins

    # Determine band activity via two criteria:
    # 1. Within-band std: high std means spectral variation (actual signals)
    # 2. Between-antenna P90 spread: antennas disagree = real gain differences,
    #    even when a better antenna lifts all bins by a flat offset (low std)
    band_max_std: dict[str, float] = {}
    band_p90_spread: dict[str, float] = {}
    for band_name, _, _ in overlapping:
        max_std = 0.0
        p90_values: list[float] = []
        for antenna_name in antenna_names:
            std = antenna_std.get(antenna_name, {}).get(band_name, 0.0)
            max_std = max(max_std, std)
            p90 = antenna_p90.get(antenna_name, {}).get(band_name)
            if p90 is not None:
                p90_values.append(p90)
        band_max_std[band_name] = max_std
        band_p90_spread[band_name] = (max(p90_values) - min(p90_values)) if len(p90_values) >= 2 else 0.0

    active_bands = [
        name for name, _, _ in overlapping
        if (band_max_std.get(name, 0.0) >= BAND_ACTIVITY_STD_THRESHOLD_DB
            or band_p90_spread.get(name, 0.0) >= BETWEEN_ANTENNA_SPREAD_THRESHOLD_DB)
    ]

    # Collect rows with margin calculation
    row_data: list[
        tuple[str, float, float, dict[str, float | None], float, str, float, bool]
    ] = []

    for band_name, _, _ in overlapping:
        start_mhz, end_mhz = band_mhz[band_name]
        per_antenna: dict[str, float | None] = {}
        values: list[tuple[str, float]] = []
        for antenna_name in antenna_names:
            p90 = antenna_p90.get(antenna_name, {}).get(band_name)
            per_antenna[antenna_name] = p90
            if p90 is not None:
                values.append((antenna_name, p90))
        if values:
            values.sort(key=lambda v: v[1], reverse=True)
            best_antenna, best_p90 = values[0]
            second_best = values[1][1] if len(values) > 1 else best_p90
            margin = best_p90 - second_best
            is_active = band_name in active_bands
            row_data.append((
                band_name, start_mhz, end_mhz,
                per_antenna, best_p90, best_antenna, margin,
                is_active,
            ))

    # Sort by best P90 descending
    row_data.sort(key=lambda r: r[4], reverse=True)

    if not row_data:
        console.print(
            "[dim]No overlapping bands found across sweeps.[/dim]"
        )
        return {}, antenna_p90, antenna_std, active_bands

    # Build Rich table — antenna columns show P90 (dBm)
    table = Table(
        title="Antenna Comparison -- P90 Power (dBm) Per Band",
        caption=(
            "P90 = 90th percentile power. Active = within-band std >= "
            f"{BAND_ACTIVITY_STD_THRESHOLD_DB:.1f} dB OR P90 spread >= "
            f"{BETWEEN_ANTENNA_SPREAD_THRESHOLD_DB:.1f} dB across antennas."
        ),
    )
    table.add_column("Band", style="cyan")
    table.add_column("Start (MHz)", justify="right")
    table.add_column("End (MHz)", justify="right")
    for name in antenna_names:
        table.add_column(name, justify="right")
    table.add_column("Best", style="bold green")
    table.add_column("Margin", justify="right")
    table.add_column("Activity", justify="center")

    best_per_band: dict[str, str] = {}
    for (band_name, start_mhz, end_mhz, per_antenna, _, best_antenna,
         margin, is_active) in row_data:
        row: list[str] = [band_name, f"{start_mhz:.1f}", f"{end_mhz:.1f}"]
        for antenna_name in antenna_names:
            p90 = per_antenna.get(antenna_name)
            row.append(f"{p90:.1f}" if p90 is not None else "--")

        if not is_active:
            row.append("[dim]--[/dim]")
            row.append("[dim]--[/dim]")
            spread = band_p90_spread.get(band_name, 0.0)
            max_std = band_max_std.get(band_name, 0.0)
            row.append(f"[dim]no (std={max_std:.1f}, spread={spread:.1f})[/dim]")
            best_per_band[band_name] = "--"
        else:
            # Find second-place antenna for context
            band_values = [
                (ant, per_antenna.get(ant))
                for ant in antenna_names if per_antenna.get(ant) is not None
            ]
            band_values.sort(key=lambda v: v[1], reverse=True)  # type: ignore[arg-type]
            second_name = band_values[1][0] if len(band_values) > 1 else "--"
            row.append(f"[bold green]{best_antenna}[/bold green]")
            row.append(f"+{margin:.1f} over {second_name}")
            spread = band_p90_spread.get(band_name, 0.0)
            max_std = band_max_std.get(band_name, 0.0)
            reasons: list[str] = []
            if max_std >= BAND_ACTIVITY_STD_THRESHOLD_DB:
                reasons.append(f"std={max_std:.1f}")
            if spread >= BETWEEN_ANTENNA_SPREAD_THRESHOLD_DB:
                reasons.append(f"spread={spread:.1f}")
            row.append(f"[green]yes ({', '.join(reasons)})[/green]")
            best_per_band[band_name] = best_antenna
        table.add_row(*row)

    console.print(table)
    console.print("\n[bold]Best antenna per active band:[/bold]")
    for band, antenna in best_per_band.items():
        if antenna == "--":
            continue
        console.print(
            f"  [cyan]{band}[/cyan]: [green]{antenna}[/green]"
        )

    # Save markdown
    if markdown_path is not None:
        md_lines = [
            "# Antenna Comparison -- P90 Power (dBm) Per Band",
            "",
            "_P90 = 90th percentile power. Captures signal peaks, ignores noise-floor bins._",
            f"_Active = within-band std >= {BAND_ACTIVITY_STD_THRESHOLD_DB:.1f} dB"
            f" OR P90 spread >= {BETWEEN_ANTENNA_SPREAD_THRESHOLD_DB:.1f} dB across antennas._",
            "",
            "| Band | Start (MHz) | End (MHz) | "
            + " | ".join(antenna_names)
            + " | Best | Margin | Activity |",
            "|------|-------------|-----------|"
            + "|".join("----------" for _ in antenna_names)
            + "|------|--------|----------|",
        ]
        for (band_name, start_mhz, end_mhz, per_antenna, _, best_antenna,
             margin, is_active) in row_data:
            cells = [band_name, f"{start_mhz:.1f}", f"{end_mhz:.1f}"]
            for antenna_name in antenna_names:
                p90 = per_antenna.get(antenna_name)
                cells.append(f"{p90:.1f}" if p90 is not None else "--")
            if not is_active:
                spread = band_p90_spread.get(band_name, 0.0)
                max_std = band_max_std.get(band_name, 0.0)
                cells.extend(["--", "--", f"no (std={max_std:.1f}, spread={spread:.1f})"])
            else:
                band_values = [
                    (ant, per_antenna.get(ant))
                    for ant in antenna_names if per_antenna.get(ant) is not None
                ]
                band_values.sort(key=lambda v: v[1], reverse=True)  # type: ignore[arg-type]
                second_name = band_values[1][0] if len(band_values) > 1 else "--"
                cells.append(f"**{best_antenna}**")
                cells.append(f"+{margin:.1f} over {second_name}")
                max_std = band_max_std.get(band_name, 0.0)
                spread = band_p90_spread.get(band_name, 0.0)
                md_reasons: list[str] = []
                if max_std >= BAND_ACTIVITY_STD_THRESHOLD_DB:
                    md_reasons.append(f"std={max_std:.1f}")
                if spread >= BETWEEN_ANTENNA_SPREAD_THRESHOLD_DB:
                    md_reasons.append(f"spread={spread:.1f}")
                cells.append(f"yes ({', '.join(md_reasons)})")
            md_lines.append("| " + " | ".join(cells) + " |")
        md_lines.append("")
        md_lines.append("## Best Antenna Per Active Band")
        md_lines.append("")
        for band, antenna in best_per_band.items():
            if antenna != "--":
                md_lines.append(f"- **{band}**: {antenna}")
        md_lines.append("")
        markdown_path.write_text("\n".join(md_lines))
        console.print(
            f"[green]Comparison report saved: {markdown_path}[/green]"
        )

    return best_per_band, antenna_p90, antenna_std, active_bands


# ---------------------------------------------------------------------------
# Relative advantage analysis (compare-all view)
# ---------------------------------------------------------------------------


def compute_relative_advantage(
    antenna_p90: dict[str, dict[str, float]],
    active_bands: list[str],
) -> dict[str, dict[str, float]]:
    """Compute each antenna's P90 advantage relative to the group median.

    Returns dict[antenna_name][band_name] -> advantage_db (positive = above median).
    """
    advantage: dict[str, dict[str, float]] = {}
    antenna_names = list(antenna_p90.keys())

    for band_name in active_bands:
        band_values = [
            antenna_p90[ant].get(band_name)
            for ant in antenna_names
        ]
        valid_values = [v for v in band_values if v is not None]
        if not valid_values:
            continue
        median_p90 = float(np.median(valid_values))
        for ant in antenna_names:
            p90 = antenna_p90[ant].get(band_name)
            if p90 is not None:
                advantage.setdefault(ant, {})[band_name] = p90 - median_p90

    return advantage


def print_compare_all(
    antenna_names: list[str],
    active_bands: list[str],
    advantage: dict[str, dict[str, float]],
    antenna_std: dict[str, dict[str, float]],
    markdown_path: Path | None = None,
) -> None:
    """Print a relative advantage table: bands as rows, antennas as columns.

    Cells show dB advantage over the group median. Green=above, red=below, dim=within noise.
    """
    # Compute per-band pooled SE for coloring threshold
    band_se: dict[str, float] = {}
    for band_name in active_bands:
        stds = [
            antenna_std.get(ant, {}).get(band_name, 0.0)
            for ant in antenna_names
        ]
        valid_stds = [s for s in stds if s > 0]
        if valid_stds:
            band_se[band_name] = float(np.mean(valid_stds)) / 10.0  # rough SE proxy
        else:
            band_se[band_name] = 0.5

    table = Table(
        title="Relative Advantage (dB vs Group Median)",
        caption="Green = above median, Red = below median, Dim = within noise (< pooled SE)",
    )
    table.add_column("Band", style="cyan")
    for name in antenna_names:
        table.add_column(name, justify="right")

    for band_name in active_bands:
        row: list[str] = [band_name]
        se_threshold = band_se.get(band_name, 0.5)
        for ant in antenna_names:
            adv = advantage.get(ant, {}).get(band_name)
            if adv is None:
                row.append("--")
            elif abs(adv) < se_threshold:
                row.append(f"[dim]{adv:+.1f}[/dim]")
            elif adv > 0:
                row.append(f"[green]{adv:+.1f}[/green]")
            else:
                row.append(f"[red]{adv:+.1f}[/red]")
        table.add_row(*row)

    console.print(table)

    # Append to markdown
    if markdown_path is not None:
        md_lines = [
            "",
            "## Relative Advantage (dB vs Group Median)",
            "",
            "_Positive = above median P90, Negative = below median P90._",
            "",
            "| Band | " + " | ".join(antenna_names) + " |",
            "|------|" + "|".join("----------" for _ in antenna_names) + "|",
        ]
        for band_name in active_bands:
            cells = [band_name]
            for ant in antenna_names:
                adv = advantage.get(ant, {}).get(band_name)
                cells.append(f"{adv:+.1f}" if adv is not None else "--")
            md_lines.append("| " + " | ".join(cells) + " |")
        md_lines.append("")

        existing = markdown_path.read_text() if markdown_path.exists() else ""
        markdown_path.write_text(existing + "\n".join(md_lines))


def plot_heatmap(
    antenna_names: list[str],
    active_bands: list[str],
    advantage: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """Generate a 2D heatmap: X=active bands, Y=antennas, cells=dB advantage."""
    n_bands = len(active_bands)
    n_ants = len(antenna_names)
    if n_bands == 0 or n_ants == 0:
        return

    data = np.zeros((n_ants, n_bands))
    for j, band in enumerate(active_bands):
        for i, ant in enumerate(antenna_names):
            data[i, j] = advantage.get(ant, {}).get(band, 0.0)

    fig, ax = plt.subplots(figsize=(max(10, n_bands * 0.9), max(4, n_ants * 0.7)))
    abs_max = max(abs(data.min()), abs(data.max()), 1.0)
    im = ax.imshow(
        data, aspect="auto", cmap="RdYlGn",
        vmin=-abs_max, vmax=abs_max,
    )

    ax.set_xticks(range(n_bands))
    ax.set_xticklabels(active_bands, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_ants))
    ax.set_yticklabels(antenna_names, fontsize=9)

    # Annotate cells
    for i in range(n_ants):
        for j in range(n_bands):
            val = data[i, j]
            color = "black" if abs(val) < abs_max * 0.6 else "white"
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")

    ax.set_title("Antenna Advantage vs Group Median (dB)", fontsize=11)
    fig.colorbar(im, ax=ax, label="dB advantage", shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    console.print(f"[green]Heatmap saved: {output_path}[/green]")


# ---------------------------------------------------------------------------
# Coverage ranking
# ---------------------------------------------------------------------------


def compute_coverage_ranking(
    antenna_p90: dict[str, dict[str, float]],
    active_bands: list[str],
) -> dict[str, dict[str, int | list[str]]]:
    """Score each antenna's coverage across active bands.

    For each active band, find the best P90 across all antennas.
    Scoring per antenna per band:
      - Within 1 dB of best: +2 points (competitive)
      - Within 3 dB of best: +1 point (adequate)
      - Beyond 3 dB: 0 points

    Returns dict[antenna] -> {
        "score": total_score,
        "competitive_bands": [band names within 1 dB],
        "adequate_bands": [band names within 3 dB but not 1 dB],
        "unique_bands": [bands where only this antenna is within 1 dB],
    }
    """
    antenna_names = list(antenna_p90.keys())

    # Best P90 per band
    best_p90: dict[str, float] = {}
    for band in active_bands:
        values = [antenna_p90[ant].get(band) for ant in antenna_names]
        valid = [v for v in values if v is not None]
        if valid:
            best_p90[band] = max(valid)

    # Score each antenna
    ranking: dict[str, dict] = {}
    competitive_per_band: dict[str, list[str]] = {b: [] for b in active_bands}

    for ant in antenna_names:
        score = 0
        competitive: list[str] = []
        adequate: list[str] = []
        for band in active_bands:
            p90 = antenna_p90[ant].get(band)
            bp = best_p90.get(band)
            if p90 is None or bp is None:
                continue
            gap = bp - p90
            if gap <= 1.0:
                score += 2
                competitive.append(band)
                competitive_per_band[band].append(ant)
            elif gap <= 3.0:
                score += 1
                adequate.append(band)

        ranking[ant] = {
            "score": score,
            "competitive_bands": competitive,
            "adequate_bands": adequate,
            "unique_bands": [],
        }

    # Unique value: bands where only this antenna is within 1 dB
    for band in active_bands:
        if len(competitive_per_band[band]) == 1:
            sole_ant = competitive_per_band[band][0]
            ranking[sole_ant]["unique_bands"].append(band)

    return ranking


def _greedy_set_cover(
    antenna_p90: dict[str, dict[str, float]],
    active_bands: list[str],
    ranking: dict[str, dict],
    top_n: int,
) -> list[tuple[str, list[str], str]]:
    """Greedy set-cover: pick antennas that cover the most uncovered bands.

    "Covered" = within 3 dB of best for that band.
    Always returns exactly top_n picks (or all antennas if fewer available).
    After all bands are covered, continues picking by highest coverage score.
    Returns list of (antenna_name, new_bands_covered, reason).
    """
    antenna_names = list(ranking.keys())
    best_p90: dict[str, float] = {}
    for band in active_bands:
        values = [antenna_p90[ant].get(band) for ant in antenna_names]
        valid = [v for v in values if v is not None]
        if valid:
            best_p90[band] = max(valid)

    # Which bands each antenna covers (within 3 dB)
    coverage: dict[str, set[str]] = {}
    for ant in antenna_names:
        covered = set()
        for band in active_bands:
            p90 = antenna_p90[ant].get(band)
            bp = best_p90.get(band)
            if p90 is not None and bp is not None and (bp - p90) <= 3.0:
                covered.add(band)
        coverage[ant] = covered

    picks: list[tuple[str, list[str], str]] = []
    uncovered = set(active_bands)
    used: set[str] = set()

    for pick_idx in range(min(top_n, len(antenna_names))):
        best_ant = None
        best_new: set[str] = set()
        best_score = -1

        if uncovered:
            # Phase 1: pick by most uncovered bands, break ties by score
            for ant in antenna_names:
                if ant in used:
                    continue
                new_covered = coverage[ant] & uncovered
                score = ranking[ant]["score"]
                if len(new_covered) > len(best_new) or (
                    len(new_covered) == len(best_new) and (
                        score > best_score or (
                            score == best_score and (best_ant is None or ant < best_ant)
                        )
                    )
                ):
                    best_ant = ant
                    best_new = new_covered
                    best_score = score
            reason = "covers new bands"
        else:
            # Phase 2: all bands covered, pick by highest score for redundancy
            for ant in antenna_names:
                if ant in used:
                    continue
                score = ranking[ant]["score"]
                if score > best_score or (
                    score == best_score and (best_ant is None or ant < best_ant)
                ):
                    best_ant = ant
                    best_score = score
            best_new = coverage.get(best_ant, set()) if best_ant else set()
            reason = "backup (all bands already covered)"

        if best_ant is None:
            break
        picks.append((best_ant, sorted(best_new if uncovered else coverage.get(best_ant, set())), reason))
        uncovered -= best_new
        used.add(best_ant)

    return picks


def print_coverage_ranking(
    antenna_names: list[str],
    active_bands: list[str],
    ranking: dict[str, dict],
    antenna_p90: dict[str, dict[str, float]],
    top_n: int = 3,
    markdown_path: Path | None = None,
) -> None:
    """Print coverage ranking table and greedy set-cover recommendation."""
    max_possible = len(active_bands) * 2

    # Sort by score descending
    sorted_ants = sorted(
        antenna_names,
        key=lambda a: ranking.get(a, {}).get("score", 0),
        reverse=True,
    )

    table = Table(title="Coverage Ranking (Active Bands Only)")
    table.add_column("Rank", justify="center", style="bold")
    table.add_column("Antenna", style="cyan")
    table.add_column("Score", justify="right", style="green")
    table.add_column(f"Max ({max_possible})", justify="right", style="dim")
    table.add_column("Competitive (<=1dB)", justify="right")
    table.add_column("Adequate (<=3dB)", justify="right")
    table.add_column("Unique Bands", style="yellow")

    for rank, ant in enumerate(sorted_ants, 1):
        r = ranking.get(ant, {})
        score = r.get("score", 0)
        competitive = r.get("competitive_bands", [])
        adequate = r.get("adequate_bands", [])
        unique = r.get("unique_bands", [])
        table.add_row(
            str(rank),
            ant,
            str(score),
            f"{score}/{max_possible}",
            str(len(competitive)),
            str(len(adequate)),
            ", ".join(unique) if unique else "--",
        )

    console.print(table)

    # Greedy set-cover
    picks = _greedy_set_cover(antenna_p90, active_bands, ranking, top_n)
    console.print(f"\n[bold]If you can only bring {top_n} antenna(s):[/bold]")
    all_covered: set[str] = set()
    for i, (ant, bands, reason) in enumerate(picks, 1):
        if reason == "covers new bands":
            all_covered.update(bands)
            console.print(
                f"  {i}. [green]{ant}[/green] (score {ranking[ant]['score']}) "
                f"-- covers: {', '.join(bands)}"
            )
        else:
            console.print(
                f"  {i}. [green]{ant}[/green] (score {ranking[ant]['score']}) "
                f"-- {reason}, strong in: {', '.join(bands)}"
            )

    remaining = set(active_bands) - all_covered
    if remaining:
        console.print(f"  [yellow]Uncovered bands: {', '.join(sorted(remaining))}[/yellow]")

    # Append to markdown
    if markdown_path is not None:
        md_lines = [
            "",
            "## Coverage Ranking (Active Bands Only)",
            "",
            f"_Scoring: +2 per band within 1 dB of best, +1 within 3 dB. Max = {max_possible}._",
            "",
            "| Rank | Antenna | Score | Competitive (<=1dB) | Adequate (<=3dB) | Unique Bands |",
            "|------|---------|-------|---------------------|------------------|--------------|",
        ]
        for rank, ant in enumerate(sorted_ants, 1):
            r = ranking.get(ant, {})
            score = r.get("score", 0)
            competitive = r.get("competitive_bands", [])
            adequate = r.get("adequate_bands", [])
            unique = r.get("unique_bands", [])
            md_lines.append(
                f"| {rank} | {ant} | {score}/{max_possible} "
                f"| {len(competitive)} | {len(adequate)} "
                f"| {', '.join(unique) if unique else '--'} |"
            )

        md_lines.append("")
        md_lines.append(f"### Set-Cover Recommendation (top {top_n})")
        md_lines.append("")
        for i, (ant, bands, reason) in enumerate(picks, 1):
            if reason == "covers new bands":
                md_lines.append(f"{i}. **{ant}** (score {ranking[ant]['score']}) -- covers: {', '.join(bands)}")
            else:
                md_lines.append(
                    f"{i}. **{ant}** (score {ranking[ant]['score']}) -- {reason}, "
                    f"strong in: {', '.join(bands)}"
                )
        if remaining:
            md_lines.append(f"\n_Uncovered: {', '.join(sorted(remaining))}_")
        md_lines.append("")

        existing = markdown_path.read_text() if markdown_path.exists() else ""
        markdown_path.write_text(existing + "\n".join(md_lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gqrx_sweep.py",
        description="Antenna sweep & comparison tool with GQRX and hackrf_sweep backends",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- sweep --
    sweep_parser = subparsers.add_parser(
        "sweep",
        help="Run a single antenna sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run gqrx_sweep.py sweep --antenna "Diamond X-50" --start 88M --end 108M
  uv run gqrx_sweep.py sweep --antenna "Nagoya NA-771" --start 1M --end 6G --backend hackrf --sweeps 100
  uv run gqrx_sweep.py sweep --antenna "Test" --start 144M --end 148M --step 100K --dwell 1.0

Baseline calibration (isolates antenna gain from ambient RF):
  1. Connect a reference antenna (or 50-ohm terminator)
  2. uv run gqrx_sweep.py sweep --antenna baseline --start 1M --end 6G --backend hackrf --sweeps 100
     (saves as baseline.csv, auto-detected by future sweeps and compare)
  3. uv run gqrx_sweep.py sweep --antenna "My Whip" --start 1M --end 6G --backend hackrf --sweeps 100
     (automatically subtracts baseline.csv if present)

Frequency suffixes: K (kHz), M (MHz), G (GHz)
        """,
    )
    sweep_parser.add_argument(
        "--antenna", required=True, help="Name of the antenna under test"
    )
    sweep_parser.add_argument(
        "--start", required=True, help="Start frequency (e.g., 88M)"
    )
    sweep_parser.add_argument(
        "--end", required=True, help="End frequency (e.g., 108M)"
    )
    sweep_parser.add_argument(
        "--backend",
        choices=["gqrx", "hackrf"],
        default="gqrx",
        help="Sweep backend (default: gqrx)",
    )
    # GQRX-specific args
    sweep_parser.add_argument(
        "--host", default="127.0.0.1", help="GQRX host (default: 127.0.0.1)"
    )
    sweep_parser.add_argument(
        "--port", type=int, default=7356, help="GQRX port (default: 7356)"
    )
    sweep_parser.add_argument(
        "--step",
        default="100K",
        help="Step size for GQRX sweep (default: 100K)",
    )
    sweep_parser.add_argument(
        "--mode",
        default=None,
        help="GQRX demodulator mode (FM, AM, WFM, USB, LSB, CW)",
    )
    sweep_parser.add_argument(
        "--dwell",
        type=float,
        default=0.5,
        help="GQRX dwell time per step in seconds (default: 0.5)",
    )
    sweep_parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="GQRX signal samples per step (default: 3)",
    )
    # HackRF-specific args
    sweep_parser.add_argument(
        "--lna-gain",
        type=int,
        default=16,
        help="HackRF LNA gain 0-40, 8 dB steps (default: 16)",
    )
    sweep_parser.add_argument(
        "--vga-gain",
        type=int,
        default=22,
        help="HackRF VGA gain 0-62, 2 dB steps (default: 22)",
    )
    sweep_parser.add_argument(
        "--bin-width",
        type=int,
        default=100_000,
        help="HackRF bin width in Hz (default: 100000)",
    )
    sweep_parser.add_argument(
        "--amp", action="store_true", help="Enable HackRF RF amplifier"
    )
    sweep_parser.add_argument(
        "--sweeps",
        type=int,
        default=1,
        help="Number of hackrf_sweep passes to average (default: 1). "
        "Higher values reduce noise and improve accuracy.",
    )
    # Baseline / output options
    sweep_parser.add_argument(
        "--baseline",
        default=None,
        help="Baseline CSV (e.g. a 50-ohm terminator sweep) to subtract. "
        "Shows gain above receiver noise floor per band.",
    )
    sweep_parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip PNG plot generation",
    )

    # -- compare --
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare multiple antenna sweep CSVs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Auto-discovers all sweep_*.csv files in the current directory.

Outputs:
  - comparison_<timestamp>.md   Per-band ranking, relative advantage, coverage ranking
  - comparison_<timestamp>.png  Overlay plot with recommendation panel
  - heatmap_<timestamp>.png     Diverging heatmap of dB advantage vs group median

Examples:
  uv run gqrx_sweep.py compare
  uv run gqrx_sweep.py compare --baseline baseline.csv
  uv run gqrx_sweep.py compare --output my_comparison.png
  uv run gqrx_sweep.py compare --top-n 2
        """,
    )
    compare_parser.add_argument(
        "--baseline",
        default=None,
        help="Baseline CSV for normalization (subtract reference sweep)",
    )
    compare_parser.add_argument(
        "--output",
        default=None,
        help="Output PNG filename (default: comparison_<timestamp>.png)",
    )
    compare_parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of antennas for set-cover recommendation (default: 3)",
    )

    return parser


def cmd_sweep(args: argparse.Namespace) -> None:
    """Execute the sweep subcommand."""
    start_hz = parse_frequency(args.start)
    end_hz = parse_frequency(args.end)
    step_hz = parse_frequency(args.step)

    if start_hz >= end_hz:
        console.print(
            "[red]Start frequency must be less than end frequency.[/red]"
        )
        sys.exit(1)

    console.rule(f"[bold]Antenna Sweep: {args.antenna}[/bold]")
    console.print(f"  Range: {format_freq(start_hz)} -> {format_freq(end_hz)}")
    console.print(f"  Backend: {args.backend}")

    # Build backend
    backend: GqrxSweepBackend | HackRFSweepBackend
    if args.backend == "hackrf":
        backend = HackRFSweepBackend(
            lna_gain=args.lna_gain,
            vga_gain=args.vga_gain,
            bin_width=args.bin_width,
            amp_enable=args.amp,
            num_sweeps=args.sweeps,
        )
    else:
        backend = GqrxSweepBackend(
            host=args.host,
            port=args.port,
            mode=args.mode,
            dwell_time=args.dwell,
            samples_per_step=args.samples,
        )

    results = backend.sweep(start_hz, end_hz, step_hz)

    if not results:
        console.print("[red]No measurements collected.[/red]")
        sys.exit(1)

    console.print(f"[green]{len(results)} measurements collected.[/green]")

    # Compute noise floor on raw data
    powers = [r[1] for r in results]
    noise_floor = estimate_noise_floor(powers)
    frequencies = np.array([r[0] for r in results])
    powers_arr = np.array(powers)

    # Save CSV (always raw data, before baseline subtraction)
    metadata = backend.get_metadata()
    csv_path = save_sweep_csv(
        results, args.antenna, start_hz, end_hz, metadata, noise_floor
    )
    console.print(f"[green]CSV saved: {csv_path}[/green]")

    # Apply baseline subtraction
    # Auto-detect baseline.csv if not explicitly provided (skip for baseline sweeps)
    is_baseline_sweep = args.antenna.lower() == "baseline"
    baseline_applied = False
    baseline_source: str | None = args.baseline

    if not baseline_source and not is_baseline_sweep and BASELINE_CSV.exists():
        baseline_source = str(BASELINE_CSV)
        console.print(
            f"[dim]Auto-detected {BASELINE_CSV} for baseline subtraction[/dim]"
        )

    if baseline_source:
        baseline_path = Path(baseline_source)
        if not baseline_path.exists():
            console.print(
                f"[red]Baseline file not found: {baseline_source}[/red]"
            )
            sys.exit(1)
        _, baseline_data = load_sweep_csv(baseline_path)
        if baseline_data:
            baseline_powers = interpolate_baseline(results, baseline_data)
            powers_arr = powers_arr - baseline_powers
            baseline_applied = True
            console.print(
                f"[green]Baseline subtracted: {baseline_source} "
                f"({len(baseline_data)} points)[/green]"
            )

    # Save plot
    png_path: Path | None = None
    if not args.no_plot:
        png_path = csv_path.with_suffix(".png")
        if baseline_applied:
            sub_results = list(zip(frequencies.tolist(), powers_arr.tolist()))
            plot_single_sweep(
                sub_results, args.antenna, start_hz, end_hz,
                0.0, png_path, baseline_subtracted=True,
            )
        else:
            plot_single_sweep(
                results, args.antenna, start_hz, end_hz, noise_floor, png_path
            )
        console.print(f"[green]Plot saved: {png_path}[/green]")

    # Summary data
    unit = "dB above baseline" if baseline_applied else "dBm"
    peak_idx = int(np.argmax(powers_arr))
    peak_freq = format_freq(float(frequencies[peak_idx]))
    summary_rows = [
        ("Antenna", args.antenna),
        ("Start", format_freq(start_hz)),
        ("End", format_freq(end_hz)),
        ("Measurements", str(len(results))),
        ("Noise Floor", f"{noise_floor:.1f} dBm"),
    ]
    if baseline_applied:
        summary_rows.append(("Baseline", baseline_source or ""))
    summary_rows += [
        ("Peak Power", f"{float(powers_arr[peak_idx]):.1f} {unit}"),
        ("Peak Frequency", peak_freq),
        ("Mean Power", f"{float(np.mean(powers_arr)):.1f} {unit}"),
        ("Std Dev", f"{float(np.std(powers_arr)):.1f} dB"),
    ]

    # Print summary table
    table = Table(title="Sweep Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    for metric, value in summary_rows:
        table.add_row(metric, value)
    console.print(table)

    # Per-band metrics (on potentially baseline-subtracted data)
    analysis_noise = 0.0 if baseline_applied else noise_floor
    band_data = compute_band_metrics_all(frequencies, powers_arr, analysis_noise)
    print_band_metrics(args.antenna, band_data)

    # Save full report as markdown
    md_path = csv_path.with_suffix(".md")
    save_sweep_report_markdown(
        args.antenna, summary_rows, band_data, csv_path, png_path, md_path
    )
    console.print(f"[green]Report saved: {md_path}[/green]")


def cmd_compare(args: argparse.Namespace) -> None:
    """Execute the compare subcommand."""
    csv_files = sorted(Path(".").glob("sweep_*.csv"))
    if not csv_files:
        console.print(
            "[yellow]No sweep_*.csv files found in current directory.[/yellow]"
        )
        sys.exit(1)

    console.rule("[bold]Antenna Comparison[/bold]")
    console.print(f"Found {len(csv_files)} sweep file(s):")
    for f in csv_files:
        console.print(f"  [dim]{f}[/dim]")

    # Load all sweeps
    sweeps: list[tuple[str, dict[str, str], list[tuple[float, float]]]] = []
    for csv_file in csv_files:
        metadata, data = load_sweep_csv(csv_file)
        antenna_name = metadata.get("antenna", csv_file.stem)
        console.print(
            f"  Loaded [cyan]{antenna_name}[/cyan]: {len(data)} points"
        )
        sweeps.append((antenna_name, metadata, data))

    # Load baseline (explicit or auto-detected)
    baseline_data: list[tuple[float, float]] | None = None
    baseline_source: str | None = args.baseline
    if not baseline_source and BASELINE_CSV.exists():
        baseline_source = str(BASELINE_CSV)
        console.print(
            f"[dim]Auto-detected {BASELINE_CSV} for baseline subtraction[/dim]"
        )
    if baseline_source:
        baseline_path = Path(baseline_source)
        if not baseline_path.exists():
            console.print(
                f"[red]Baseline file not found: {baseline_source}[/red]"
            )
            sys.exit(1)
        _, baseline_data = load_sweep_csv(baseline_path)
        console.print(
            f"  Baseline: [cyan]{baseline_source}[/cyan] ({len(baseline_data)} points)"
        )

    # Print ranking + save markdown
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = Path(f"comparison_{timestamp}.md")
    best_per_band, antenna_p90, antenna_std, active_bands = print_comparison_ranking(
        sweeps, baseline_data, markdown_path=md_path,
    )

    # Generate overlay plot with recommendation panel
    plot_sweeps = [(name, data) for name, _, data in sweeps]
    output_path = (
        Path(args.output) if args.output else Path(f"comparison_{timestamp}.png")
    )
    plot_comparison(plot_sweeps, output_path, baseline_data, best_per_band=best_per_band)
    console.print(f"\n[green]Comparison plot saved: {output_path}[/green]")

    # Relative advantage analysis (compare-all view)
    antenna_names = [name for name, _, _ in sweeps]
    if active_bands and antenna_p90:
        advantage = compute_relative_advantage(antenna_p90, active_bands)
        print_compare_all(antenna_names, active_bands, advantage, antenna_std, md_path)
        heatmap_path = Path(f"heatmap_{timestamp}.png")
        plot_heatmap(antenna_names, active_bands, advantage, heatmap_path)

    # Coverage ranking
    if active_bands and antenna_p90:
        top_n = getattr(args, "top_n", 3)
        ranking = compute_coverage_ranking(antenna_p90, active_bands)
        print_coverage_ranking(antenna_names, active_bands, ranking, antenna_p90, top_n, md_path)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "sweep":
        cmd_sweep(args)
    elif args.command == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
