# wildfire-codes 🔥⚡

Code for wildfire-aware power system studies, with a focus on **optimal transmission switching (OTS)**, **topology control**, and **Public Safety Power Shutoff (PSPS)**–style line outages on benchmark test systems (e.g., RTS-GMLC).

The repository collects several generations of code in **MATLAB**, **Julia**, and **Python** for:

- Solving DC-OPF and DC-OTS models on transmission grids  
- Running Benders decomposition and scenario-based OTS  
- Generating and analyzing contingency / wildfire scenarios  
- Post-processing and visualization of results

---

## Repository structure

```text
wildfire-codes/
├── config/    # Text / script-based configs for experiments and solver options
├── io/        # Input data (cases, scenarios) and saved outputs/results
├── matlab/    # Core OTS, OPF, Benders, and RTS-GMLC scripts (legacy + current)
├── python/    # Helper scripts for data processing, plotting, and automation
└── README.md
