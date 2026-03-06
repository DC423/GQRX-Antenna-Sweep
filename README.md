# GQRX Antenna Sweep

A Python tool that connects to [GQRX](https://gqrx.dk/)'s remote control socket and sweeps across a frequency range to figure out where your antenna performs best. Zero dependencies — just Python 3.6+ and a running GQRX instance.

## What It Does

1. Connects to GQRX over TCP (Hamlib `rigctld` protocol)
2. Tunes through your chosen frequency range step by step
3. Takes multiple signal strength samples at each frequency and averages them
4. Shows a real-time bar chart in the terminal as it sweeps
5. Analyzes the data to find peak performance and optimal range
6. Saves everything to CSV for further analysis

## Quick Start

### Prerequisites

- **GQRX** running with Remote Control enabled:  
  `Tools → Remote Control → Start`
- **Python 3.6+** (stdlib only, nothing to `pip install`)

### Basic Usage

```bash
python3 gqrx_sweep.py --host <GQRX_IP> --start <FROM> --end <TO> --step <SIZE>
```

Frequencies accept suffixes: **K** (kHz), **M** (MHz), **G** (GHz)

### Examples

```bash
# Sweep FM broadcast band on a remote Pi running GQRX
python3 gqrx_sweep.py --host 192.168.1.50 --start 88M --end 108M --step 500K

# 2-meter ham band, slower dwell for better accuracy
python3 gqrx_sweep.py --start 144M --end 148M --step 100K --dwell 1.0

# 70cm band, more samples, custom output file
python3 gqrx_sweep.py --start 430M --end 440M --step 250K --samples 5 --output 70cm.csv

# AM broadcast band sweep, skip CSV
python3 gqrx_sweep.py --start 500K --end 1700K --step 50K --mode AM --no-save

# Local GQRX, full RTL-SDR range scan (takes a while)
python3 gqrx_sweep.py --start 24M --end 1.7G --step 10M --dwell 0.3
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | GQRX IP address |
| `--port` | `7356` | GQRX remote control port |
| `--start` | *required* | Start frequency (e.g. `88M`) |
| `--end` | *required* | End frequency (e.g. `108M`) |
| `--step` | *required* | Step size (e.g. `500K`) |
| `--dwell` | `0.5` | Seconds to wait at each step before sampling |
| `--samples` | `3` | Number of readings to average per step |
| `--mode` | *unchanged* | Set demod mode: `FM` `AM` `WFM` `USB` `LSB` `CW` |
| `--output` | auto | CSV filename (defaults to `antenna_sweep_YYYYMMDD_HHMMSS.csv`) |
| `--no-save` | off | Skip saving CSV |

## Sample Output

```
============================================================
  ANTENNA SWEEP
  Range: 144.000 MHz -> 148.000 MHz
  Step:  100.0 kHz  |  Steps: 41
  Dwell: 0.5s  |  Samples/step: 3
============================================================

    144.000 MHz   -85.3 dBFS  █████████████████
    144.100 MHz   -82.1 dBFS  ██████████████████
    ...
    146.500 MHz   -62.4 dBFS  ████████████████████████████
    ...

[+] Sweep complete. 41 measurements taken.

============================================================
  ANTENNA ANALYSIS REPORT
============================================================
  Best frequency:    146.500 MHz (-62.4 dBFS)
  Worst frequency:   148.000 MHz (-95.1 dBFS)
  Average signal:    -79.8 dBFS
  Good range:        145.800 MHz - 147.200 MHz
  Threshold used:    -71.1 dBFS
  Measurements:      41
============================================================
  Verdict: Antenna has a moderate preference for the good range.
============================================================
```

## CSV Output Format

```csv
frequency_hz,frequency_readable,signal_dbfs
144000000,144.000 MHz,-85.3
144100000,144.100 MHz,-82.1
...
```

## Tips

- **Dwell time matters** — Shorter dwell (`0.2s`) = faster sweep but noisier. Longer (`1-2s`) = slower but more accurate readings.
- **More samples = smoother data** — `--samples 5` or higher helps in noisy environments.
- **Step size tradeoff** — Smaller steps give finer resolution but take longer. Start coarse, then zoom into interesting ranges.
- **Ctrl+C is safe** — The sweep can be interrupted at any time; you'll still get results for the frequencies already measured.

## How It Works Under the Hood

The tool speaks GQRX's remote control protocol (compatible with Hamlib `rigctld`):

- `F <hz>` — Set frequency
- `f` — Get frequency  
- `l STRENGTH` — Read signal strength (dBFS)
- `M <mode> <bw>` — Set demodulator mode

All communication happens over a plain TCP socket on port 7356 (default).


