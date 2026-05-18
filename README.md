# CBGTPy: An extensible cortico-basal ganglia-thalamic framework for modeling biological decisions

**Original authors:** Matthew Clapp, Jyotika Bahuguna, Cristina Giossi, Jonathan E. Rubin, Timothy Verstynen, Catalina Vich

CBGTPy is a virtual environment for designing and testing goal-directed agents with internal dynamics modeled off of the cortico-basal-ganglia-thalamic (CBGT) pathways in the mammalian brain, including a physiologically-realistic implementation of dopamine-driven synaptic plasticity. CBGTPy enables researchers to investigate the internal dynamics of the CBGT system during a variety of tasks, allowing for the formation of testable predictions about animal behavior and neural activity.

---

> **This is a packaged fork of the [original CBGTPy](https://github.com/CoAxLab/CBGTPy) by [CoAxLab](https://github.com/CoAxLab).**
>
> The sole purpose of this fork is to make CBGTPy installable as a standard Python package via `pip`. No changes have been made to the core simulation logic, algorithms, or scientific functionality. All source modules have been reorganized under a `cbgtpy` namespace package so they can be properly installed and imported without manual path manipulation.
>
> **If you use CBGTPy in your research, please cite the original authors and the [original repository](https://github.com/CoAxLab/CBGTPy).**

---

## Installation

### Prerequisites

- **Python 3.8+**
- **A C compiler** — required for building Cython extensions
  - **Linux:** gcc (usually pre-installed)
  - **Mac:** Xcode Command Line Tools
    ```bash
    xcode-select --install
    ```
    Optionally install gcc via Homebrew:
    ```bash
    brew install gcc
    ```
  - **Windows:** Microsoft Visual C++ Build Tools or MSVC (install via [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/))

### Install directly with pip (recommended)

```bash
pip install git+https://github.com/SwaragThaikkandi/CBGTPy.git
```

This will automatically install all dependencies (NumPy, Pandas, SciPy, Matplotlib, Seaborn, Cython, PyYAML, Pathos) and compile the Cython extensions.

#### Optional: with Ray support

```bash
pip install "cbgtpy[ray] @ git+https://github.com/SwaragThaikkandi/CBGTPy.git"
```

#### Optional: with development tools (Jupyter)

```bash
pip install "cbgtpy[dev] @ git+https://github.com/SwaragThaikkandi/CBGTPy.git"
```

### Install from cloned source

```bash
git clone https://github.com/SwaragThaikkandi/CBGTPy.git
cd CBGTPy
pip install .
```

For an editable/development install:
```bash
pip install -e ".[dev]"
```

### Verifying the installation

```python
import cbgtpy
print(cbgtpy.__version__)
```

## What changed from the original repository

This fork restructures the project into a proper Python package. The table below summarizes all differences:

| Aspect | Original ([CoAxLab/CBGTPy](https://github.com/CoAxLab/CBGTPy)) | This fork (packaged) |
|--------|----------|------|
| **Installation** | `conda create` + `python install.py <env>` | `pip install git+https://github.com/SwaragThaikkandi/CBGTPy.git` |
| **Dependencies** | Manually managed via conda `environment.yml` | Automatically resolved by pip via `pyproject.toml` |
| **Source layout** | `common/`, `nchoice/`, `stopsignal/` at repo root | Nested under `cbgtpy/` package directory |
| **Imports** | `import common.cbgt as cbgt` (requires `sys.path` hack) | `import cbgtpy.common.cbgt as cbgt` (works anywhere after install) |
| **Cython build** | Run `setup.py` from `notebooks/` dir, manually move `.so`/`.pyd` files | Handled automatically by `pip install` |
| **Notebooks** | Need `sys.path.append('../')` at top | Work directly after package install |
| **Multiprocessing setup** | Interactive prompt during `install.py` | Install extras: `pip install "cbgtpy[ray]"` |
| **Simulation code** | — | **Unchanged** |

### Import migration reference

If you have existing scripts written for the original CBGTPy, update imports as follows:

| Original import | Packaged import |
|----------------|-----------------|
| `import common.cbgt as cbgt` | `import cbgtpy.common.cbgt as cbgt` |
| `import common.pipeline_creation as pl_creat` | `import cbgtpy.common.pipeline_creation as pl_creat` |
| `import common.plotting_functions as plt_func` | `import cbgtpy.common.plotting_functions as plt_func` |
| `import common.plotting_helper_functions as plt_help` | `import cbgtpy.common.plotting_helper_functions as plt_help` |
| `import common.postprocessing_helpers as post_help` | `import cbgtpy.common.postprocessing_helpers as post_help` |
| `import nchoice.paramfile_nchoice as paramfile` | `import cbgtpy.nchoice.paramfile_nchoice as paramfile` |
| `import stopsignal.paramfile_stopsignal as paramfile` | `import cbgtpy.stopsignal.paramfile_stopsignal as paramfile` |

**General rule:** prefix all imports with `cbgtpy.` and remove any `sys.path.append` lines.

## Usage

### Quick start

```python
import cbgtpy.common.cbgt as cbgt
import cbgtpy.common.pipeline_creation as pl_creat
import cbgtpy.common.plotting_functions as plt_func
import cbgtpy.common.plotting_helper_functions as plt_help
import cbgtpy.common.postprocessing_helpers as post_help

# Choose experiment type
pl_creat.choose_pipeline('n-choice')  # or 'stop-signal'
```

### Example notebooks

Example notebooks are provided in the `notebooks/` directory of the repository:

| Notebook | Description |
|----------|-------------|
| `network_simulation-n-choice.ipynb` | N-choice decision task simulation |
| `network_simulation-stop-signal.ipynb` | Stop-signal (response inhibition) task |
| `network_simulation-n-choice-optostim.ipynb` | N-choice with optogenetic stimulation |

To run notebooks after installing from source:
```bash
cd CBGTPy/notebooks
jupyter notebook
```

If you installed via pip (not from cloned source), clone the repo separately to get notebooks:
```bash
git clone https://github.com/SwaragThaikkandi/CBGTPy.git
cd CBGTPy/notebooks
jupyter notebook
```

## Multiprocessing

CBGTPy supports three multiprocessing modes:

| Mode | Install command | Notes |
|------|----------------|-------|
| **Pathos** (default) | Included automatically | Good cross-platform support |
| **Ray** | `pip install "cbgtpy[ray] @ git+https://github.com/SwaragThaikkandi/CBGTPy.git"` | Better performance, requires server start |
| **Single-threaded** | Default install | No parallel execution |

To start Ray server (if using Ray):
```bash
ray start --head --port=6379 --redis-password="cbgt2"
```

### Benchmarks (5 simulations, 3 trials each)

| Machine | Single-threaded | Pathos | Ray |
|---------|----------------|--------|-----|
| Apple M1, macOS Ventura 13.2.1 | 664s | 331s | 266s |
| Intel i7-11800H, Windows 10 | 525s | 386s | 232s |

## Project Structure

```
CBGTPy/
├── cbgtpy/                          # Installable Python package
│   ├── __init__.py                  # Package root (version, top-level imports)
│   ├── common/                      # Core framework modules
│   │   ├── backend.py               # Pipeline computation architecture
│   │   ├── cbgt.py                  # Central module (imports core components)
│   │   ├── tracetype.py             # Custom Trace type for pandas ExtensionDtype
│   │   ├── frontendhelpers.py       # Utility functions
│   │   ├── agentmatrixinit.py       # Agent/synapse initialization
│   │   ├── pipeline_creation.py     # High-level pipeline composition
│   │   ├── generateepochs.py        # Trial epoch/stimulus timing generation
│   │   ├── generate_opt_dataframe.py # Optogenetic stimulation dataframes
│   │   ├── qvalues.py               # Q-value computation (reinforcement learning)
│   │   ├── pathwayconstruct.py      # Neural pathway/connection specs
│   │   ├── plotting_functions.py    # Visualization
│   │   ├── plotting_helper_functions.py
│   │   ├── postprocessing_helpers.py # Post-simulation analysis
│   │   └── agent_timestep.pyx       # Base Cython timestep module
│   ├── nchoice/                     # N-choice decision task
│   │   ├── interface_nchoice.py     # Task entry point (mega_loop)
│   │   ├── paramfile_nchoice.py     # Default parameters
│   │   ├── init_params_nchoice.py   # Parameter initialization helpers
│   │   ├── popconstruct_nchoice.py  # Population construction
│   │   └── agent_timestep_plasticity.pyx  # Cython timestep with plasticity
│   └── stopsignal/                  # Stop-signal task
│       ├── interface_stopsignal.py  # Task entry point (mega_loop)
│       ├── paramfile_stopsignal.py  # Default parameters
│       ├── init_params_stopsignal.py
│       ├── popconstruct_stopsignal.py
│       ├── generate_stop_dataframe.py
│       └── agent_timestep_stop_signal.pyx  # Cython timestep for stop-signal
├── notebooks/                       # Example Jupyter notebooks
├── pyproject.toml                   # Package metadata and dependencies
├── setup.py                         # Cython extension build configuration
├── MANIFEST.in                      # Source distribution file list
└── README.md
```

## Dependencies

All dependencies are installed automatically via pip:

| Package | Minimum version | Purpose |
|---------|----------------|---------|
| numpy | 1.20 | Numerical computation |
| pandas | 1.3 | DataFrames and data manipulation |
| scipy | 1.7 | Statistical functions |
| matplotlib | 3.4 | Plotting |
| seaborn | 0.11 | Statistical visualization |
| Cython | 0.29 | Compiling performance-critical timestep loops |
| pyyaml | 5.4 | YAML configuration |
| pathos | 0.2.8 | Multiprocessing (default backend) |

**Optional:**
| Package | Install extra | Purpose |
|---------|--------------|---------|
| ray | `[ray]` | Distributed multiprocessing |
| jupyter, notebook, ipykernel | `[dev]` | Running example notebooks |

## Troubleshooting

### Cython compilation fails
Ensure a C compiler is available. On Windows, install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/). On Mac, run `xcode-select --install`.

### `ModuleNotFoundError: No module named 'cbgtpy'`
Ensure you installed the package (`pip install ...`), not just cloned the repo. If using notebooks from a cloned repo, run `pip install -e .` from the repo root first.

### Pathos/Ray import errors after install
Try deactivating and reactivating your virtual environment or conda environment, then verify:
```python
import pathos
```

## Credits

- **Original project:** [CoAxLab/CBGTPy](https://github.com/CoAxLab/CBGTPy)
- **Original authors:** Matthew Clapp, Jyotika Bahuguna, Cristina Giossi, Jonathan E. Rubin, Timothy Verstynen, Catalina Vich
- **Packaging fork:** [SwaragThaikkandi/CBGTPy](https://github.com/SwaragThaikkandi/CBGTPy)
