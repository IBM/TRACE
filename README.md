# TRACE: Trace-to-Runtime Analyzer for CIM Exploration

TRACE is a hardware-aware performance profiler for PyTorch models that bridges the gap between neural network definitions and custom hardware specifications. It provides detailed energy and latency metrics for inference runs on user-defined hardware architectures, from simple digital processors to complex analog-in-memory computing (AIMC) accelerators.

## Overview

TRACE transforms PyTorch models into hardware-executable job graphs through a series of progressive lowerings and transformations. The system integrates seamlessly with PyTorch's `torch.compile` infrastructure, enabling frictionless profiling of existing models.

### Key Features

- **Flexible Hardware Modeling**: Define custom hardware architectures via YAML configuration files
- **PyTorch Integration**: Works with standard PyTorch models through `torch.compile`
- **Multi-Stage Transformation Pipeline**: Progressive lowering from high-level operations to hardware-specific instructions
- **Multiple Scheduling Strategies**: From simple ASAP to constraint-based optimal scheduling
- **Detailed Performance Analysis**: Interactive visualization of latency, energy, and resource utilization
- **Hardware-Aware Optimization**: Automatic or user-defined transformation based on target hardware capabilities

## Quick Start

### Installation

1. **Prerequisites**: Install VSCode and GCC (clang is temporarily not supported)

2. **Clone and Build**:

```bash
git clone --recursive git@github.com:IBM/TRACE.git
cd trace

# Recommended: Use mamba environment
mamba env create --name trace_env python=3.13 cmake=3.27 gxx=13.3.0 ninja clang-format

# Optional but highly recommended: Install torch 2.9.1 according to your specific system
# See: https://pytorch.org/get-started/previous-versions/
# Example for CUDA 12.8:
python3 -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# Install dependencies and build
pip install -r requirements.txt
bash scripts/apply_patches.sh
pre-commit install
python3 -m pip install -e . --no-build-isolation # optional to debug build outout, --verbose

# Alternative: build with custom isomorphisms/transformations
python3 -m pip install -e . --no-build-isolation --config-settings=override=cmake.options.EXTERNAL_ISO_TRANS_SOURCES=$PWD/transformations

```

3. **Verify Installation**:

```bash
python examples/MLP/compile_model.py
```

### Basic Usage

```python
import torch
from trace.torch_compile_backend.compile_options import CompileOptions
from trace.ops.torch_fx_patterns import MMPattern, AddMMPatternSplit

# Load your model
model = MyModel()
model.eval()

# Configure TRACE
options = CompileOptions(
    system_path="yaml/system.yaml",
    pipeline_path="yaml/transformation_pipeline.yaml",
    torch_patterns=[MMPattern, AddMMPatternSplit],
    torch_pattern_matching=True,
    scheduling_strategy="ConflictAwareScheduler",
    performance_logging=True,
    performance_logging_path="performance.yaml"
)

# Compile with TRACE backend
torch.set_grad_enabled(False)
compiled_model = torch.compile(model, backend="trace", options=options)

# Run inference (triggers compilation and profiling)
output = compiled_model(input_tensor)
```

## Architecture

TRACE operates through a multi-stage pipeline:

```txt
PyTorch Model
    ↓
[torch.compile captures FX graph]
    ↓
PyTorch Transformations (pattern matching, graph cleanup)
    ↓
[Lower to GHL/C++ job graph]
    ↓
GHL Transformations (mapping, lowering, placement)
    ↓
[Execution on inputs]
    ↓
[Scheduling with resource awareness]
    ↓
Performance Metrics (latency, energy, utilization)
```

### Core Components

1. **Hardware Specification**: YAML-based definition of processing units, operations, and interconnects
2. **Transformation Pipeline**: Progressive lowering from PyTorch ops to hardware-specific jobs
3. **Torch Compile Backend**: Integration with PyTorch's compilation infrastructure
4. **Execution Engine**: Runs job graph on actual inputs to produce outputs
5. **Scheduling System**: Assigns timing and resources while respecting dependencies and constraints
6. **Performance Profiling**: Exports detailed metrics for analysis and visualization

## Documentation

### Core Documentation

- **[Hardware Specification](docs/guides/hardware_specification.md)**: Define custom hardware architectures via YAML
  - Implementations, ops, interconnects, and hierarchical nodes
  - System graph construction and validation
  - Examples and best practices

- **[Transformations](docs/guides/transformations.md)**: Progressive lowering pipeline
  - PyTorch transformations (pattern matching, graph passes)
  - GHL transformations (mapping, lowering, placement, transfers)
  - Transformation pipeline YAML configuration
  - Writing custom transformations

- **[Torch Compile Backend](docs/guides/torch_compile_backend.md)**: Integration with PyTorch
  - Compilation flow and caching
  - CompileOptions configuration
  - Interactive mode and debugging
  - Multi-tenant support (experimental)

- **[Scheduling](docs/guides/scheduling.md)**: Job scheduling strategies
  - BareBonesScheduler (fast, no resource checking)
  - ConflictAwareScheduler (resource-aware, production-ready)
  - SchedulingEdgesScheduler (advanced constraints)
  - ConstraintScheduler (optimal with memory constraints)

- **[Execution](docs/guides/execution.md)**: Job graph execution
  - Execution flow and boundaries
  - Post-execution transformation hooks
  - Debug artifacts

- **[Profiling](docs/guides/profiling.md)**: Performance analysis and visualization
  - PerformanceLogger API
  - Interactive GUI for timeline and energy analysis
  - CSV export and custom analysis scripts

## Workflow

### 1. Define Hardware

Create a YAML file describing your target hardware architecture:

```yaml
implementations:
  add:
    - aten::add
  vmm:
    - trace_ops::dense_layer

ops:
  alu_add:
    type: add
    width: 32
    base_latency: 5
    dynamic_energy: 0.5

  tile_vmm:
    type: vmm
    tags: [analog]
    width: 512
    base_latency: 100
    transformations:
      split_tall_mm:
        max_height: 512

interconnects:
  fabric_bus:
    type: bus
    dir: DUAL
    width: 64
    base_latency: 30

nodes:
  system:
    vertices:
      (0,0): alu_unit
      (1,0): tile_unit
```

See [Hardware Specification](docs/guides/hardware_specification.md) for details.

### 2. Configure Transformations

Define PyTorch-level transformations:

```python
class MMPattern(BasePattern):
    @staticmethod
    def pattern_to_match(x, y):
        p = torch.ops.aten.t.default(y)
        return torch.ops.aten.mm.default(x, p)

    @staticmethod
    def replacement(x, y):
        return torch.ops.trace_ops.dense_layer.default(x, y, None)
```

Define the GHL transformation pipeline in YAML:

```yaml
- map_jobs:
    import: .mapping_placement
    arguments:
      - layer_job_graph
      - options
    return: mapping_d

- apply_transformation:
    import: trace.trace_bindings
    arguments:
      - layer_job_graph
      - "lower_layers_to_jobs"
      - system: system
      - mappings: mapping_d

- place_jobs:
    import: .mapping_placement
    arguments:
      - layer_job_graph
      - options

- apply_transformation:
    import: trace.trace_bindings
    arguments:
      - layer_job_graph
      - "insert_transfer_jobs"
      - system: system
```

Select transformations in CompileOptions:

```python
from TRACE.torch_compile_backend.compile_options import CompileOptions
from TRACE.ops.torch_fx_patterns import (
    MMPattern,           # Convert mm(x, transpose(y)) to dense_layer(x, y)
    AddMMPatternSplit,   # Split addmm into separate operations
    LayerNorm,           # Replace layer norm with single op
    cleaning_patterns,   # Remove useless ops (reshape, clone, etc.)
    common_patterns      # Common pattern replacements
)


options = CompileOptions(
    system_path="yaml/system.yaml",
    pipeline_path="yaml/transformation_pipeline.yaml",
    torch_pattern_matching=True,
    torch_patterns=[
        *cleaning_patterns,  # Apply cleanup first
        *common_patterns,    # Then common replacements
        MMPattern,           # Custom patterns
        AddMMPatternSplit,
        LayerNorm,
    ]
)
```

See [Transformations](docs/guides/transformations.md) for details.

### 3. Compile and Profile

Use `torch.compile` with TRACE backend:

```python
options = CompileOptions(
    system_path="yaml/system.yaml",
    pipeline_path="yaml/transformation_pipeline.yaml",
    scheduling_strategy="ConflictAwareScheduler",
    performance_logging=True
)

compiled_model = torch.compile(model, backend="TRACE", options=options)
output = compiled_model(input_tensor)
```

### 4. Analyze Results

Visualize performance metrics:

```bash
python -m TRACE.yaml_parser.performance_explorer performance.yaml plot_options.yaml .
```

This generates:

- Interactive GUI with timeline and energy plots
- SVG visualizations
- CSV summary with latency, energy, and power metrics

See [Profiling](docs/guides/profiling.md) for details.

## Key Concepts

### Virtual vs. Physical Jobs

- **Virtual Jobs**: Default node in GHL graph when converted from Python. Have 0 latency/energy.
- **Physical Jobs**: Hardware-specific operations after mapping and lowering (e.g., tiled MVM on analog crossbar)

The transformation pipeline converts virtual jobs to physical jobs through:

1. **Mapping**: Assign virtual ops to hardware op types
2. **Lowering**: Apply hardware-specific transformations (e.g., tiling)
3. **Placement**: Assign jobs to specific hardware nodes
4. **Transfer Insertion**: Add data movement between nodes

### Compilation Caching

Torch.compile caches compiled graphs per input shape. Compilation occurs once per unique shape, with subsequent calls reusing the cached graph.

### Interactive Mode

Pause compilation for manual intervention:

```python
options = CompileOptions(
    system_path="yaml/system.yaml",
    interactive=["mapping", "placement", "scheduling"]
)
```

TRACE will pause after generating mapping/placement files, allowing manual editing before continuing.

## Development

### Debugging

1. Install the [Python C++ Debugger extension](https://marketplace.visualstudio.com/items?itemName=benjamin-simmonds.pythoncpp-debug)

2. Add to `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Model",
            "type": "pythoncpp",
            "request": "launch",
            "pythonLaunchName": "Python: Current File",
            "cppAttachName": "(lldb) Attach"
        }
    ]
}
```

### Code Formatting

```bash
# Python
black python_project

# C++
./clang-format-all.sh src/cpp
```

## Important Notes

- **Disable Gradients**: TRACE does not support backward passes. Always use `torch.set_grad_enabled(False)`

## Examples

Working examples are available in the `examples/` directory. The best supported models are:

- **MLP**: Simple multi-layer perceptron
- **Llama**: Large language model with bus architecture

Each example includes:

- Model definition
- System YAML configuration
- Transformation pipeline YAML
- Compilation script
- Performance analysis utilities

## License

[License information to be added]

## Support

For questions or issues:

1. Check the detailed documentation in `docs/`
2. Review working examples in `examples/`
3. Ask [code maintainers](MAINTAINERS.md).