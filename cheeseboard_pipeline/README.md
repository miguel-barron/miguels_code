# Cheeseboard Analysis Pipeline
This repository contains a modular Python pipeline designed for analyzing rodent spatial memory and contextual 
learning tasks used in the ATN–RSC Cheeseboard and Free Explore projects. The codebase supports trial 
segmentation, behavioral metric computation, context manipulation, and visualization of performance across 
multiple sessions and days.

The pipeline was built with flexibility in mind: collaborators can adjust trial definitions,
behavioral metrics, or visualization routines as new research questions emerge.

---
## Usage
### Run the full pipeline
```bash
python main.py RecList.xlsx ExperimentName
```

### Generate summary metric figures
```bash
python make_three_metric_figs.py /Path/experiment_data.csv --outdir /Path/Output_Folder
```

---
## Outputs
The pipeline generates:
- **Trial Metrics** (start/stop frames, context, trial type, trial level summaries). 
- **Figures** (trajectories, heatmaps, metric summaries).  

These outputs can be fed into higher-level statistical analyses or integrated into lab presentations and manuscripts.

---
## Analysis Flow Diagram
flowchart
    A[Raw Tracking Data] --> B[Position Processing]
    B --> C[Trial Extraction]
    C --> D[Trial Metrics]
    D --> E[Context Manipulation (Curtain)]
        E --> F[Figures & Visualization]
    D --> F[Figures & Visualization]
    F --> G[Results & Exports]

---
## Installation
Core dependencies:
- Python 3.9+  
- NumPy  
- Pandas  
- Matplotlib  
- tqdm  

---
## Customization & Lab Integration
Julia (and others) can easily extend the pipeline by:
- Editing **`trial_extraction.py`** to define new trial structures.  
- Adding new metrics in **`trial_metrics.py`**.  
- Updating **`visualization.py`** for new figure layouts.  
- Expanding **`curtain_context.py`** for additional manipulations.  

This modular design ensures that future variants of Cheeseboard or contextual learning tasks can be 
integrated without rewriting the whole pipeline.

## Structure & Module Details
### **`main.py`**
The entry point that coordinates the full pipeline:
- Loads raw position/tracking data using `data_io.py`.
- Runs preprocessing (`position_processing.py`).
- Extracts trials (`trial_extraction.py`).
- Computes metrics (`trial_metrics.py`).
- Optionally applies context manipulations (`curtain_context.py`).
- Generates figures (`make_three_metric_figs.py`, `visualization.py`).

**Adjustments:**  
- Change which pipeline stages run in sequence (e.g., bypass trial metrics if only preprocessing is needed).  
- Add command-line flags for specific experiments (e.g., Cheeseboard vs. Free Explore).  

---
### **`data_io.py`**
Handles file I/O for tracking data, trial definitions, and metrics:
- Reads raw `.csv`/`.xlsx` tracking files and organizes them by day/session.  
- Writes processed data, trial tables, and metrics outputs in CSV format.  

**Adjustments:**  
- Adapt file path logic to match new server organization.  
- Extend loaders if new data formats appear (e.g., DeepLabCut vs. SLEAP outputs).  

---
### **`position_processing.py`**
Prepares raw tracking data for analysis:
- Interpolates missing coordinates.  
- Applies smoothing or filters to reduce noise.  
- Ensures data is aligned with video frames and timestamps.  

**Adjustments:**  
- Modify interpolation method if data dropout patterns change.  
- Add/remove filters for different camera setups.  

---
### **`trial_extraction.py`**
Defines and extracts behavioral trials:
- Identifies start/stop frames for **Cheeseboard trials** (30s or 10s segments, single or multiple wells).  
- Handles **Free Explore sessions** (one long trial, ~30 min).  
- Annotates trial contexts (e.g., well visits, exploration patterns).  

**Adjustments:**  
- Update logic for different experimental designs (e.g., 3-well Cheeseboard vs. 1-well).  
- Add new event detection (e.g., curtain drop/start cues).  

---
### **`trial_metrics.py`**
Computes per-trial behavioral metrics:
- Path length (raw vs. cleaned).  
- Speed, turning angles, head-direction alignment.  
- Number and timing of well visits.  
- Success vs. failure outcomes (defined by reaching target wells and trial timing).  

**Adjustments:**  
- Add/remove metrics depending on project needs (e.g., first-well latency, exploration ratios).  
- Redefine trial success criteria for novel experiments.  

---
### **`curtain_context.py`**
Implements **context-specific manipulations**, especially for tasks where curtains change visual access:
- Encodes whether trials occur in open vs. blocked conditions.  
- Allows comparisons of spatial strategies under altered sensory contexts.  

**Adjustments:**  
- Expand for new environmental manipulations beyond curtain contexts.  
- Modify annotations to include additional trial metadata.  

---
### **`visualization.py`**
Generates visual summaries of behavioral data:
- Plots trial trajectories with wells marked.  
- Heatmaps of position density across trials.  
- Success/failure outcome plots.  

**Adjustments:**  
- Change plotting style for conference posters vs. manuscripts.  
- Add figure panels for new metrics as they are developed.  

---
### **`make_three_metric_figs.py`**
Standalone script for quick figure generation:
- Produces summary figures across sessions for **three selected metrics** 
  (e.g., path length, trial success, well visits).  
- Saves figures (SVG/PNG).  

**Adjustments:**  
- Choose which three metrics to plot in the script.  
- Customize axis labels, colors, and layout for presentations.
