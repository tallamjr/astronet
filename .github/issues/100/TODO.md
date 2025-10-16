# AstroNet Modernization Plan: TensorFlow/Keras → PyTorch + Polars/Arrow

**Version**: 2.0 (Revised)
**Last Updated**: 2025-10-12
**Status**: Planning Phase

---

## Executive Summary

This document outlines the comprehensive plan to migrate AstroNet from
TensorFlow/Keras to PyTorch, whilst modernizing the data pipeline with Polars
and Apache Arrow for true zero-copy operations. The migration will maintain the
three core architectures (T2, Tinho, ATX) whilst improving performance,
maintainability, and adopting modern ML engineering best practices.

**Key Revisions in v2.0**:

- Added Phase 0 for bridge layer and migration verification
- Adopted PyTorch Lightning for training infrastructure
- Enhanced testing strategy with ML-specific tests
- Added data versioning and experiment tracking
- Improved zero-copy implementation strategy
- Added astronomy-specific considerations

---

## Current State Architecture

````
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CURRENT ARCHITECTURE (TF/Keras)                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CSV/Parquet → Pandas → GP Interpolation → NumPy Arrays (.npy)              │
│       ↓            ↓                              ↓                         │
│   Raw Data    Processing              Memory-mapped loading                 │
│                  (preprocess.py)                  ↓                         │
│                       ↓                   X_train/test.npy                  │
│                 Mixed Pandas/           Z_train/test.npy                    │
│                 Polars usage            y_train/test.npy                    │
│                                                  ↓                          │
│                                    tf.data.Dataset.from_generator           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│  MODEL LAYER                                                               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                      │
│  │   T2 Model   │  │ Tinho Model  │  │   ATX Model  │                      │
│  │  (t2/model)  │  │(tinho/func)  │  │  (atx/model) │                      │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤                      │
│  │ Transformer  │  │ Transformer  │  │  Xception-   │                      │
│  │   + Conv     │  │   + Conv     │  │   inspired   │                      │
│  │  Embedding   │  │  Embedding   │  │  + Residual  │                      │
│  │              │  │              │  │              │                      │
│  │ Multi-head   │  │ Multi-head   │  │ Entry/Middle │                      │
│  │  Attention   │  │  Attention   │  │  /Exit Flow  │                      │
│  └──────────────┘  └──────────────┘  └──────────────┘                      │
│                                                                            │
│  Implementation: tf.keras.Model subclasses                                 │
│  Custom layers: ConvEmbedding, TransformerBlock, PositionalEncoding        │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  TRAINING LAYER                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Training Loop: Keras .fit() API                                            │
│       ↓                                                                     │
│  Optimizer: tf.keras.optimizers.Adam                                        │
│  Loss: Custom WeightedLogLoss / DistributedWeightedLogLoss                  │
│  Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger    │
│       ↓                                                                     │
│  Hyperparameter Opt: Optuna (opt/hypertrain.py)                             │
│       ↓                                                                     │
│  Model Compression: TFLite, tfmot pruning/clustering                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

**FOLDER STRUCTURE**

```bash
astronet/
├── t2/                  # Architecture 1 (everything together)
│   ├── model.py
│   ├── transformer.py
│   ├── attention.py
│   ├── train.py
│   └── opt/
│       └── hypertrain.py
├── tinho/               # Architecture 2
│   ├── funcmodel.py
│   ├── lite.py
│   └── opt/
├── atx/                 # Architecture 3
│   ├── model.py
│   ├── layers.py
│   └── opt/
├── train.py             # Shared training
├── datasets.py          # Data loading
├── preprocess.py        # Mixed pandas/polars preprocessing
├── metrics.py
├── utils.py
└── tests/
    ├── unit/
    ├── int/
    ├── func/
    └── reg/
````

ISSUES WITH CURRENT STATE:
- Data pipeline: Inefficient copies between pandas/numpy/tensorflow
- Mixed pandas/polars usage causes confusion and extra copies
- Memory-mapped .npy files still require full arrays in memory for batching
- TensorFlow/Keras coupling makes migration to other frameworks difficult
- Architecture-specific folders mix concerns (models, training, optimization)
- No clear separation between data processing, model definition, training
- Limited type hints and modern Python features
- Test organisation could be clearer (unit/int/func/reg distinctions)
- No experiment tracking or data versioning

```

---

## Proposed State Architecture

```

┌─────────────────────────────────────────────────────────────────────────────┐
│ PROPOSED ARCHITECTURE (PyTorch Lightning + Arrow) │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ DATA LAYER (True Zero-Copy with Apache Arrow) │
├─────────────────────────────────────────────────────────────────────────────┤
│ │
│ Parquet → Polars LazyFrame → GP Interpolation → Arrow IPC │
│ ↓ ↓ ↓ ↓ │
│ Raw Data Lazy Eval Polars/tinygp Memory-mapped │
│ (DVC) (streaming) (faster GP) Arrow files │
│ ↓ ↓ ↓ ↓ │
│ Version Predicate Vectorised Zero-copy │
│ Tracked pushdown operations to NumPy │
│ ↓ ↓ ↓ ↓ │
│ Partitioned Efficient Pure Polars torch.Tensor │
│ by class filtering (no pandas) (via Arrow) │
│ ↓ ↓ ↓ │
│ Schema Cache as DataLoader │
│ validation Arrow IPC (zero-copy) │
│ │
│ TRUE ZERO-COPY IMPLEMENTATION: │
│ ┌────────────────────────────────────────────────────────────────┐ │
│ │ df_pl.to_arrow() # Polars → Arrow (zero-copy) │ │
│ │ arrow_col.to_numpy(zero_copy_only=True) # Arrow → NumPy view │ │
│ │ torch.from_numpy(np_view) # NumPy → Torch (zero-copy) │ │
│ └────────────────────────────────────────────────────────────────┘ │
│ │
│ Benefits: │
│ • True zero-copy tensor conversion (Arrow → PyTorch) │
│ • Lazy evaluation with predicate pushdown │
│ • Columnar format optimised for time-series │
│ • 5-10x faster than pandas for large datasets │
│ • Native support for Parquet with partitioning │
│ • Streaming for datasets larger than memory │
│ • Data versioning with DVC │
│ │
└─────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ MODEL LAYER (PyTorch Native) │
├───────────────────────────────────────────────────────────────────────────┤
│ │
│ ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐ │
│ │ T2 Model │ │ Tinho Model │ │ ATX Model │ │
│ │ (models/t2.py) │ │ (models/tinho.py) │ │ (models/atx.py) │ │
│ ├────────────────────┤ ├────────────────────┤ ├────────────────────┤ │
│ │ nn.Transformer │ │ nn.Transformer │ │ Custom Xception │ │
│ │ (modern PyTorch) │ │ (modern PyTorch) │ │ with residuals │ │
│ │ + │ │ + │ │ + │ │
│ │ Conv1d Embedding │ │ Conv1d Embedding │ │ Depthwise Sep. │ │
│ │ + │ │ + │ │ Convolutions │ │
│ │ Learned Pos. │ │ Learned Pos. │ │ + │ │
│ │ Encoding │ │ Encoding │ │ Adaptive Pooling │ │
│ │ ↓ │ │ ↓ │ │ ↓ │ │
│ │ nn.MultiheadAttn │ │ nn.MultiheadAttn │ │ Global Avg Pool │ │
│ │ ↓ │ │ ↓ │ │ ↓ │ │
│ │ Feed-Forward │ │ Feed-Forward │ │ Classification │ │
│ │ (nn.Sequential) │ │ (nn.Sequential) │ │ Head │ │
│ │ ↓ │ │ ↓ │ │ │ │
│ │ Classification │ │ Classification │ │ │ │
│ └────────────────────┘ └────────────────────┘ └────────────────────┘ │
│ │
│ Implementation: nn.Module with PyTorch Lightning wrapper │
│ • Type hints throughout │
│ • Configurable via Pydantic models │
│ • ONNX export compatible │
│ • Mixed precision training ready (torch.cuda.amp) │
│ │
└───────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ TRAINING LAYER (PyTorch Lightning) │
├─────────────────────────────────────────────────────────────────────────────┤
│ │
│ Lightning Module: Handles training loop boilerplate │
│ ↓ │
│ Optimizer: torch.optim.AdamW (modern default) │
│ Scheduler: CosineAnnealingLR / OneCycleLR │
│ Loss: Custom weighted cross-entropy with class weights │
│ ↓ │
│ Mixed Precision: Automatic via Lightning precision plugin │
│ Gradient Clipping: Configured in Lightning trainer │
│ ↓ │
│ Multi-GPU: DistributedDataParallel (automatic) │
│ Checkpointing: ModelCheckpoint callback │
│ ↓ │
│ Experiment Tracking: Weights & Biases / MLflow │
│ ↓ │
│ Hyperparameter Opt: Optuna with modern features │
│ • TPE Sampler (multivariate) │
│ • Hyperband pruner (early stopping) │
│ • Parallel trials │
│ ↓ │
│ Model Export: │
│ • ONNX (primary) via torch.onnx.export │
│  │
│ • LiteRT (optional) via ONNX → TFLite conversion │
│ ↓ │
│ Model Optimization: │
│ • Quantization: torch.quantization (QAT, PTQ) │
│ • Pruning: torch.nn.utils.prune │
│ • ONNX Runtime optimisation │
│ │
└─────────────────────────────────────────────────────────────────────────────┘

**NEW FOLDER STRUCTURE (Separation of Concerns):**

```bash
astronet/
│
├── bridge/                   # MIGRATION BRIDGE (Phase 0)
│   ├── __init__.py
│   ├── tf_to_pt.py           # Weight conversion utilities
│   ├── verify.py             # Numerical equivalence checker
│   ├── ensemble.py           # Ensemble TF + PT models
│   └── dual_inference.py     # Support both backends during transition
│
├── data/                     # DATA PROCESSING & LOADING
│   ├── __init__.py
│   ├── loaders.py            # Arrow-based DataLoaders
│   ├── preprocessing.py      # GP interpolation, scaling (pure Polars)
│   ├── augmentation.py       # Astronomy-specific augmentation
│   ├── datasets.py           # torch.utils.data.Dataset implementations
│   ├── transforms.py         # Time-series specific transformations
│   ├── samplers.py           # Class-balanced sampling for imbalance
│   └── validation.py         # Schema and distribution validation
│
├── models/                   # MODEL ARCHITECTURES
│   ├── __init__.py
│   ├── base.py               # Base model class with common methods
│   ├── t2.py                 # T2 Transformer architecture
│   ├── tinho.py              # Tinho architecture
│   ├── atx.py                # ATX architecture
│   ├── lightning_module.py   # Lightning wrapper for all models
│   ├── components/           # Shared components
│   │   ├── __init__.py
│   │   ├── embeddings.py     # Conv embeddings, positional encoding
│   │   ├── attention.py      # Attention mechanisms
│   │   ├── transformers.py   # Transformer blocks
│   │   └── heads.py          # Classification heads
│   └── configs.py            # Model configurations (Pydantic)
│
├── training/                 # TRAINING INFRASTRUCTURE
│   ├── __init__.py
│   ├── trainer.py            # Lightning Trainer wrapper
│   ├── callbacks.py          # Custom Lightning callbacks
│   ├── metrics.py            # Custom metrics (weighted log loss)
│   ├── losses.py             # Loss functions
│   ├── optimizers.py         # Optimizer configs
│   └── schedulers.py         # Learning rate schedulers
│
├── optimization/             # HYPERPARAMETER OPTIMIZATION
│   ├── __init__.py
│   ├── optuna_search.py      # Optuna with Hyperband pruning
│   └── search_spaces.py      # Search space definitions
│
├── inference/                # MODEL INFERENCE & EXPORT
│   ├── __init__.py
│   ├── predictor.py          # Inference wrapper
│   ├── export_onnx.py        # ONNX export
│   ├── quantization.py       # Model quantization
│   └── benchmarks.py         # Performance benchmarking
│
├── profiling/                # PERFORMANCE PROFILING
│   ├── __init__.py
│   ├── profile_data.py       # Data loading profiling
│   ├── profile_model.py      # Model forward/backward profiling
│   ├── profile_memory.py     # Memory profiling
│   └── reports/              # Generated profiling reports
│
├── utils/                    # UTILITIES
│   ├── __init__.py
│   ├── logging.py            # Logging configuration
│   ├── checkpointing.py      # Model checkpointing
│   ├── config.py             # Configuration management (Pydantic)
│   └── reproducibility.py    # Seed setting, determinism
│
├── tests/                    # TESTING (ML-SPECIFIC)
│   ├── __init__.py
│   ├── unit/                 # Unit tests (individual functions)
│   │   ├── test_data/
│   │   │   ├── test_preprocessing.py
│   │   │   ├── test_loaders.py
│   │   │   ├── test_transforms.py
│   │   │   └── test_schema_validation.py
│   │   ├── test_models/
│   │   │   ├── test_t2.py
│   │   │   ├── test_tinho.py
│   │   │   ├── test_atx.py
│   │   │   ├── test_architecture.py    # Shape checks, layer counts
│   │   │   ├── test_numerical.py       # Numerical stability
│   │   │   └── test_components/
│   │   └── test_training/
│   │       ├── test_losses.py
│   │       ├── test_metrics.py
│   │       └── test_callbacks.py
│   ├── integration/          # Integration tests (components together)
│   │   ├── test_data_pipeline.py
│   │   ├── test_training_loop.py
│   │   ├── test_inference.py
│   │   └── test_export.py
│   ├── regression/           # Regression tests (baselines)
│   │   ├── test_model_outputs.py     # Output baselines vs TF
│   │   ├── test_metrics.py           # Metric baselines
│   │   └── baselines/                # Stored baseline results
│   ├── behavioural/          # ML-specific behavioural tests
│   │   ├── test_invariances.py       # Brightness scaling, etc.
│   │   ├── test_convergence.py       # Overfitting on small batch
│   │   └── test_determinism.py       # Reproducibility
│   ├── performance/          # Performance tests
│   │   ├── test_speed.py             # Speed benchmarks
│   │   └── test_memory.py            # Memory profiling
│   └── fixtures/             # Shared test fixtures
│       ├── data.py
│       └── models.py
│
├── scripts/                  # CLI SCRIPTS
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   ├── predict.py            # Prediction script
│   ├── export.py             # Model export script
│   └── preprocess_data.py    # Data preprocessing script
│
├── configs/                  # CONFIGURATION FILES (Pydantic + YAML)
│   ├── datasets/             # Dataset-specific configs
│   │   ├── plasticc.yaml
│   │   ├── ztf.yaml
│   │   └── avocado.yaml
│   ├── models/               # Model configs
│   │   ├── t2.yaml
│   │   ├── tinho.yaml
│   │   └── atx.yaml
│   └── training/             # Training configs
│       ├── base.yaml
│       └── hyperopt.yaml
│
├── __init__.py
├── constants.py              # Constants (kept from original)
└── version.py                # Version info

data/                         # DATA DIRECTORY (external, DVC-tracked)
├── raw/                      # Raw parquet files
├── processed/                # Processed Arrow IPC files
└── interim/                  # Intermediate processing

docs/                         # DOCUMENTATION
├── architecture/
│   └── decisions/            # Architecture Decision Records (ADRs)
│       ├── 001-pytorch-over-tensorflow.md
│       ├── 002-polars-over-pandas.md
│       ├── 003-lightning-for-training.md
│       └── template.md
├── migration_guide.md        # TF → PyTorch migration guide
└── api_reference/            # API documentation

notebooks/                    # ANALYSIS NOTEBOOKS
models/                       # SAVED MODELS
logs/                         # TRAINING LOGS
results/                      # EXPERIMENT RESULTS

.dvc/                         # DVC configuration
data.dvc                      # DVC data tracking
```

BENEFITS OF PROPOSED STATE:
- Data pipeline: True zero-copy Arrow → PyTorch tensor conversion
- Pure Polars throughout (5-10x faster than pandas)
- Data versioning with DVC for reproducibility
- Streaming support for large datasets
- Clear separation of concerns (data/models/training/inference)
- PyTorch Lightning reduces boilerplate (50% less code)
- Modern PyTorch with type hints and best practices
- ONNX export for deployment flexibility
- ML-specific testing (behavioural, invariance, convergence)
- Better maintainability and extensibility
- Bridge layer for smooth transition from TF
- Astronomy-specific features (class balancing, realistic augmentation)

````

---

## Migration Phases (REVISED)

### Phase 0: Bridge Layer & Migration Infrastructure

**Objective**: Enable smooth transition and numerical verification

This critical phase was missing from v1.0. It provides the infrastructure needed
to verify the migration is correct and allows both TF and PyTorch models to
coexist during transition.

#### Tasks:

##### 0.1 Weight Conversion Utilities
- [ ] `bridge/tf_to_pt.py`
  - [ ] TensorFlow checkpoint → PyTorch state_dict converter
  - [ ] Layer name mapping (tf.keras.layers → nn.Module)
  - [ ] Weight shape transformation utilities
  - [ ] Test on all three architectures

##### 0.2 Numerical Equivalence Verification
- [ ] `bridge/verify.py`
  - [ ] ModelEquivalenceChecker class
  - [ ] Layer-by-layer output comparison
  - [ ] Forward pass verification (same input → same output)
  - [ ] Tolerance specification (atol=1e-5, rtol=1e-3)
  - [ ] Generate equivalence reports

##### 0.3 Dual Backend Support
- [ ] `bridge/dual_inference.py`
  - [ ] Unified interface for TF and PyTorch models
  - [ ] Model agnostic prediction API
  - [ ] Batch prediction support for both backends

##### 0.4 Ensemble Bridge
- [ ] `bridge/ensemble.py`
  - [ ] Ensemble TF + PyTorch predictions
  - [ ] Useful for validation during transition
  - [ ] Average or voting strategies

##### 0.5 Documentation
- [ ] Document weight conversion process
- [ ] Create verification checklist
- [ ] Write migration guide

##### 0.6 Testing
- [ ] Test weight conversion on real models
- [ ] Verify numerical equivalence within tolerances
- [ ] Test dual inference with both backends

**Deliverables**:
- Weight conversion utilities
- Numerical equivalence verification framework
- Bridge layer for TF/PyTorch coexistence
- Migration verification checklist

**Success Criteria**:
- Can convert TF weights to PyTorch
- Can verify outputs match within 0.1% tolerance
- Can run inference with both backends

---

### Phase 1: Data Pipeline Modernization

**Objective**: Replace pandas/numpy pipeline with pure Polars + Arrow with TRUE zero-copy

#### Tasks:

##### 1.1 Data Versioning Setup
- [ ] Install and configure DVC
  - [ ] `dvc init`
  - [ ] Configure remote storage (S3/GCS/local)
  - [ ] Track raw data directories
- [ ] Create data versioning workflow
  - [ ] Document data tracking process
  - [ ] Add DVC to CI/CD

##### 1.2 Convert to Pure Polars
- [ ] Rewrite `data/preprocessing.py` to use pure Polars LazyFrames
  - [ ] Remove all pandas dependencies
  - [ ] Use Polars expressions throughout
  - [ ] Implement lazy evaluation with `collect()` only when needed
- [ ] Implement streaming Parquet reading
  - [ ] Partition data by class and time
  - [ ] Use `scan_parquet()` for lazy loading
- [ ] Schema validation
  - [ ] Create `data/validation.py`
  - [ ] Validate column names, types, ranges
  - [ ] Distribution checks for data drift

##### 1.3 True Zero-Copy Implementation
- [ ] Create `data/loaders.py` with Arrow-based DataLoader
  - [ ] Implement true zero-copy: Arrow → NumPy view → PyTorch
  - [ ] **CRITICAL**: Verify with `zero_copy_only=True` flag
  - [ ] Add memory profiling to verify no copies

```python
# Correct zero-copy implementation
def zero_copy_arrow_to_torch(arrow_table, columns):
    """True zero-copy conversion."""
    # Arrow → NumPy view (zero-copy for numeric types)
    numpy_arrays = {
        col: arrow_table[col].to_numpy(zero_copy_only=True)
        for col in columns
    }
    # Stack and convert to torch (shares memory)
    np_array = np.stack([numpy_arrays[c] for c in columns], axis=-1)
    return torch.from_numpy(np_array)  # Zero-copy
````

- [ ] Implement `ArrowDataset` class extending `torch.utils.data.Dataset`
- [ ] Add memory benchmarks to verify zero-copy
- [ ] Compare memory usage: old (numpy) vs new (arrow)

##### 1.4 GP Interpolation Modernization

- [ ] Investigate GP alternatives
  - [ ] Benchmark `tinygp` (JAX-based, faster)
  - [ ] Benchmark `celerite2` (for astronomical time-series)
  - [ ] Compare with current `george` implementation
- [ ] Update GP interpolation to use Polars
  - [ ] Rewrite `fit_2d_gp` to accept Polars DataFrames
  - [ ] Update `generate_gp_all_objects` for streaming
  - [ ] Add progress tracking with `tqdm`
- [ ] Implement fallback interpolation
  - [ ] Cubic spline interpolation as faster alternative
  - [ ] Document trade-offs (accuracy vs speed)

##### 1.5 Data Augmentation

- [ ] Create `data/augmentation.py`
  - [ ] Realistic photometric noise
  - [ ] Random gaps (missing observations)
  - [ ] Time-shifting
  - [ ] Brightness scaling
- [ ] Create `data/samplers.py`
  - [ ] Class-balanced sampler for imbalanced data
  - [ ] Weighted random sampling
  - [ ] Stratified sampling

##### 1.6 Data Caching

- [ ] Implement Arrow IPC caching
  - [ ] Cache processed data as Arrow IPC files
  - [ ] Memory-mapped reading
  - [ ] Fast loading for repeated experiments

##### 1.7 Testing

- [ ] Unit tests for Polars operations
  - [ ] Test lazy evaluation
  - [ ] Test streaming operations
- [ ] Integration tests for full data pipeline
  - [ ] End-to-end test: raw data → tensors
  - [ ] Test with small dataset
- [ ] Schema validation tests
  - [ ] Test with valid and invalid data
- [ ] Zero-copy verification tests
  - [ ] Memory profiling tests
  - [ ] Verify no copies are made
- [ ] Benchmarks
  - [ ] Compare pandas vs Polars performance
  - [ ] Target: 5-10x speedup

**Deliverables**:

- Pure Polars data pipeline
- TRUE zero-copy Arrow → PyTorch conversion (verified)
- Data versioning with DVC
- GP interpolation with modern libraries
- Astronomy-specific augmentation
- 5-10x speedup in data loading (verified with benchmarks)

**Success Criteria**:

- All pandas dependencies removed from data processing
- Zero-copy verified with memory profiling
- Data loading 5-10x faster than baseline
- Schema validation passing
- DVC tracking all raw data

---

### Phase 2: Model Architecture Migration

**Objective**: Convert TensorFlow/Keras models to PyTorch (one at a time)

#### Migration Strategy: **Incremental per Architecture**

##### Tinho (Best Performing Model)

- [ ] Create base infrastructure

  - [ ] Implement `models/base.py` with common model functionality
  - [ ] Create `models/configs.py` with Pydantic configurations
  - [ ] Set up `models/components/` directory structure

- [ ] Convert shared components first

  - [ ] `components/embeddings.py`

    - [ ] Convert `ConvEmbedding` from Conv1D Keras → Conv1d PyTorch
    - [ ] Implement learned positional encoding
    - [ ] Add sinusoidal positional encoding option

  - [ ] `components/attention.py`

    - [ ] Use `nn.MultiheadAttention` (PyTorch native)
    - [ ] Ensure compatibility with batch_first=True
    - [ ] Add attention weight extraction for visualization

  - [ ] `components/transformers.py`

    - [ ] Convert `TransformerBlock` to PyTorch
    - [ ] Use `nn.TransformerEncoderLayer` as base
    - [ ] Add pre-norm and post-norm options

  - [ ] `components/heads.py`
    - [ ] Classification head with configurable pooling
    - [ ] Support for auxiliary features (redshift concatenation)

- [ ] Implement Tinho in PyTorch

  - [ ] `models/tinho.py`
    - [ ] Port transformer + functional model
    - [ ] Handle dual input (lightcurve + redshift)
    - [ ] Match original hyperparameters exactly
    - [ ] Verify output shapes match TF version

- [ ] Numerical verification (Tinho)

  - [ ] Convert TF Tinho weights to PyTorch
  - [ ] Load same weights in both models
  - [ ] Compare outputs layer-by-layer
  - [ ] Verify within tolerance (atol=1e-5, rtol=1e-3)
  - [ ] Document any numerical differences

- [ ] Testing (Tinho)
  - [ ] Unit tests for Tinho model
  - [ ] Test output shapes
  - [ ] Test forward pass
  - [ ] Test with/without redshift
  - [ ] Numerical comparison with TF
  - [ ] ONNX export test

**Milestone: Tinho PyTorch model verified equivalent to TF version**

##### T2 Architecture

- [ ] Implement `models/t2.py`

  - [ ] Port transformer architecture
  - [ ] Reuse shared components from Tinho
  - [ ] Handle dual input (lightcurve + redshift)
  - [ ] Match original hyperparameters
  - [ ] Verify output shapes match TF version

- [ ] Numerical verification (T2)

  - [ ] Convert TF T2 weights to PyTorch
  - [ ] Compare outputs layer-by-layer
  - [ ] Verify within tolerance

- [ ] Testing (T2)
  - [ ] Unit tests for T2 model
  - [ ] Test output shapes
  - [ ] Numerical comparison with TF
  - [ ] ONNX export test

**Milestone: T2 PyTorch model verified equivalent to TF version**

##### ATX Architecture

- [ ] Implement `models/atx.py`

  - [ ] Port Entry/Middle/Exit flow structure
  - [ ] Convert depthwise separable convolutions
  - [ ] Implement residual connections
  - [ ] Match Xception-inspired design

- [ ] ATX-specific components

  - [ ] Entry flow (strided convolutions + residuals)
  - [ ] Middle flow (repeated depthwise separable blocks)
  - [ ] Exit flow (global pooling + classification)

- [ ] Numerical verification (ATX)

  - [ ] Convert TF ATX weights to PyTorch
  - [ ] Compare outputs layer-by-layer
  - [ ] Verify within tolerance

- [ ] Testing (ATX)
  - [ ] Unit tests for ATX model
  - [ ] Test output shapes
  - [ ] Numerical comparison with TF
  - [ ] ONNX export test

**Milestone: ATX PyTorch model verified equivalent to TF version**

##### Cross-Architecture Tasks (Throughout)

- [ ] Comprehensive testing

  - [ ] Unit tests for all components
  - [ ] Integration tests (model + data pipeline)
  - [ ] Behavioural tests
    - [ ] Brightness invariance (scaling shouldn't change class)
    - [ ] Translation invariance (time-shifting)
  - [ ] Convergence tests (overfit on small batch)
  - [ ] Determinism tests (same seed → same output)

- [ ] Documentation
  - [ ] Document each architecture
  - [ ] API documentation with docstrings
  - [ ] Migration notes (TF → PyTorch differences)
  - [ ] Architecture Decision Records (ADRs)

**Deliverables**:

- All three architectures in PyTorch
- Shared component library
- Numerical equivalence verified for all models
- Comprehensive model tests
- ONNX export support for all models
- Migration documentation

**Success Criteria**:

- All models produce outputs within 0.1% of TF versions
- All tests passing
- ONNX export working
- Documentation complete

---

### Phase 3: Training Infrastructure with PyTorch Lightning

**Objective**: Build modern training pipeline with Lightning

**Why Lightning**: Reduces boilerplate by 50%, enforces best practices, handles multi-GPU/TPU automatically, integrates with loggers seamlessly.

#### Tasks:

##### 3.1 Lightning Module

- [ ] `models/lightning_module.py`
  - [ ] Create `AstroNetLightningModule` wrapping PyTorch models
  - [ ] Implement `training_step`, `validation_step`, `test_step`
  - [ ] Configure optimizers in `configure_optimizers()`
  - [ ] Add learning rate scheduling
  - [ ] Support for auxiliary inputs (redshift)

```python
class AstroNetLightningModule(L.LightningModule):
    def __init__(self, model: nn.Module, config: TrainingConfig):
        super().__init__()
        self.model = model  # Pure PyTorch model (T2/Tinho/ATX)
        self.config = config
        self.loss_fn = WeightedCrossEntropyLoss(...)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.model(X)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(...)
        return [optimizer], [scheduler]
```

##### 3.2 Loss Functions

- [ ] `training/losses.py`
  - [ ] Port `WeightedLogLoss` to PyTorch
  - [ ] Weighted cross-entropy with class weights
  - [ ] Focal loss variant for class imbalance
  - [ ] Label smoothing support
  - [ ] Unit tests for all loss functions

##### 3.3 Metrics

- [ ] `training/metrics.py`
  - [ ] Port weighted log loss metric
  - [ ] Integrate with `torchmetrics`
  - [ ] Astronomy-specific metrics:
    - [ ] Per-class accuracy
    - [ ] CC-SNe cross-contamination metric
    - [ ] Confusion matrix
  - [ ] Unit tests for all metrics

##### 3.4 Callbacks

- [ ] `training/callbacks.py`
  - [ ] Early stopping (use Lightning's built-in)
  - [ ] Model checkpointing (use Lightning's built-in)
  - [ ] Learning rate monitoring (use Lightning's built-in)
  - [ ] Custom callbacks:
    - [ ] TimeHistoryCallback (track epoch time)
    - [ ] PrintModelSparsity (if using pruning)
    - [ ] SavePredictionsCallback (save predictions for analysis)

##### 3.5 Trainer Wrapper

- [ ] `training/trainer.py`
  - [ ] Wrapper around Lightning Trainer
  - [ ] Default configuration for AstroNet
  - [ ] Multi-GPU support (DDP)
  - [ ] Mixed precision configuration
  - [ ] Logging configuration (W&B/MLflow/TensorBoard)

```python
class AstroNetTrainer:
    def __init__(self, config: TrainingConfig):
        self.trainer = L.Trainer(
            max_epochs=config.epochs,
            accelerator='gpu',
            devices=-1,  # Use all GPUs
            strategy='ddp',  # DistributedDataParallel
            precision='16-mixed',  # Mixed precision
            callbacks=[...],
            logger=WandbLogger(...),
        )
```

##### 3.6 Experiment Tracking Integration

- [ ] Choose experiment tracker (W&B recommended)
- [ ] Configure Weights & Biases
  - [ ] Project setup
  - [ ] Automatic logging of:
    - [ ] Hyperparameters
    - [ ] Metrics
    - [ ] Model architecture
    - [ ] System metrics
  - [ ] Artifact tracking (models, data)
- [ ] Alternative: MLflow setup (for self-hosted)

##### 3.7 Testing

- [ ] Unit tests for each component
  - [ ] Test loss functions
  - [ ] Test metrics
  - [ ] Test callbacks
- [ ] Integration test for full training loop
  - [ ] Smoke test (1 epoch on small data)
  - [ ] Test checkpointing and resuming
  - [ ] Test multi-GPU (if available)
- [ ] Convergence tests
  - [ ] Overfit on single batch
  - [ ] Verify loss decreases

**Deliverables**:

- Complete Lightning-based training infrastructure
- Mixed precision training support
- Multi-GPU training capability
- Experiment tracking integration
- Comprehensive training tests

**Success Criteria**:

- Can train all three models with Lightning
- Mixed precision working (speeds up training)
- Multi-GPU training working (if available)
- Experiment tracking logging correctly
- Can resume from checkpoints
- Training loop tests passing

---

### Phase 4: Hyperparameter Optimization

**Objective**: Modernize hyperparameter search with enhanced Optuna

#### Tasks:

##### 4.1 Optuna Integration (Enhanced)

- [ ] `optimization/optuna_search.py`
  - [ ] Update existing Optuna code for PyTorch + Lightning
  - [ ] Integrate with `AstroNetLightningModule`
  - [ ] Add modern Optuna features:
    - [ ] TPE Sampler with multivariate=True
    - [ ] HyperbandPruner for early stopping
    - [ ] Parallel trials (n_jobs=-1)
  - [ ] Optuna Dashboard for visualization

```python
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(multivariate=True),
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=1,
        max_resource=100,
        reduction_factor=3,
    ),
)

study.optimize(
    objective,
    n_trials=100,
    n_jobs=4,  # Parallel trials
    callbacks=[...],
)
```

##### 4.2 Search Spaces

- [ ] `optimization/search_spaces.py`
  - [ ] Define search spaces for each architecture
  - [ ] Add architecture-specific constraints
  - [ ] Hyperparameters to search:
    - [ ] Learning rate (log scale)
    - [ ] Batch size (categorical)
    - [ ] Dropout rate
    - [ ] Number of layers
    - [ ] Hidden dimensions
    - [ ] Optimizer (Adam vs AdamW vs Lion)

##### 4.3 Integration with Lightning

- [ ] Lightning PruningCallback
  - [ ] Prune unpromising trials early
  - [ ] Report intermediate metrics to Optuna
- [ ] Checkpoint best trials
  - [ ] Save models from top-k trials
  - [ ] Log to W&B/MLflow

##### 4.4 Testing

- [ ] Test hyperparameter search on small dataset
- [ ] Verify pruning works
- [ ] Test parallel trials
- [ ] Compare results with original Optuna runs

**Deliverables**:

- Modern Optuna-based hyperparameter optimization
- Enhanced with Hyperband pruning and parallel trials
- Search space configurations
- Integration with Lightning

**Success Criteria**:

- Hyperparameter search completes successfully
- Pruning reduces search time
- Parallel trials work correctly
- Results are logged to experiment tracker

---

### Phase 5: Inference & Export

**Objective**: Enable production deployment

#### Tasks:

##### 5.1 ONNX Export (Primary)

- [ ] `inference/export_onnx.py`
  - [ ] Export each architecture to ONNX
  - [ ] Handle dynamic batch sizes
  - [ ] Handle auxiliary inputs (redshift)
  - [ ] Test with ONNX Runtime
  - [ ] Optimize ONNX graphs
  - [ ] Verify numerical accuracy (vs PyTorch)

```python
def export_to_onnx(model, example_input, output_path):
    torch.onnx.export(
        model,
        example_input,
        output_path,
        input_names=['lightcurve', 'redshift'],
        output_names=['class_probabilities'],
        dynamic_axes={
            'lightcurve': {0: 'batch_size'},
            'redshift': {0: 'batch_size'},
            'class_probabilities': {0: 'batch_size'},
        },
        opset_version=17,
    )
```

##### 5.2 LiteRT Export (Optional)

- [ ] Research ONNX → TFLite/LiteRT conversion
  - [ ] Use `onnx-tf` converter
  - [ ] Test on edge devices (if available)
  - [ ] Document limitations

##### 5.3 Quantization

- [ ] `inference/quantization.py`
  - [ ] Post-training quantization (PTQ)
    - [ ] Dynamic quantization (weights only)
    - [ ] Static quantization (weights + activations)
  - [ ] Quantization-aware training (QAT) setup
  - [ ] INT8 quantization
  - [ ] Benchmark accuracy vs speed trade-off

##### 5.4 Inference Pipeline

- [ ] `inference/predictor.py`
  - [ ] High-level inference API
  - [ ] Support for PyTorch and ONNX
  - [ ] Batch prediction support
  - [ ] Pre/post-processing
  - [ ] Model ensemble support
  - [ ] Uncertainty quantification (optional)

```python
class AstroNetPredictor:
    def __init__(self, model_path, backend='onnx'):
        self.backend = backend
        if backend == 'onnx':
            self.session = onnxruntime.InferenceSession(model_path)
        elif backend == 'torchscript':
            self.model = torch.jit.load(model_path)
        # ...

    def predict(self, lightcurve, redshift=None):
        # Preprocessing
        # Inference
        # Postprocessing
        return class_probabilities
```

##### 5.5 Benchmarking

- [ ] `inference/benchmarks.py`
  - [ ] Latency benchmarks (p50, p95, p99)
  - [ ] Throughput benchmarks (samples/sec)
  - [ ] Memory usage profiling
  - [ ] Compare backends:
    - [ ] TensorFlow (baseline)
    - [ ] PyTorch
    - [ ] ONNX
    - [ ] ONNX + quantization

##### 5.6 Testing

- [ ] Test ONNX export for all models
- [ ] Test quantization
- [ ] Test inference API
- [ ] Numerical accuracy tests
- [ ] Performance benchmark tests

**Deliverables**:

- ONNX export for all models
- Quantization support (INT8)
- High-level inference API
- Performance benchmarks

**Success Criteria**:

- ONNX export working for all models
- Numerical accuracy within 1% of PyTorch
- Inference 2-5x faster with ONNX + quantization
- Inference API easy to use

---

### Phase 6: Testing & Documentation

**Objective**: Ensure quality and maintainability

#### Tasks:

##### 6.1 Comprehensive Testing

**Unit Tests**

- [ ] Data pipeline tests

  - [ ] `tests/unit/test_data/test_preprocessing.py`
  - [ ] `tests/unit/test_data/test_loaders.py`
  - [ ] `tests/unit/test_data/test_transforms.py`
  - [ ] `tests/unit/test_data/test_schema_validation.py`
  - [ ] `tests/unit/test_data/test_augmentation.py`

- [ ] Model tests

  - [ ] `tests/unit/test_models/test_t2.py`
  - [ ] `tests/unit/test_models/test_tinho.py`
  - [ ] `tests/unit/test_models/test_atx.py`
  - [ ] `tests/unit/test_models/test_architecture.py`
  - [ ] `tests/unit/test_models/test_numerical.py`
  - [ ] `tests/unit/test_models/test_components/`

- [ ] Training tests
  - [ ] `tests/unit/test_training/test_losses.py`
  - [ ] `tests/unit/test_training/test_metrics.py`
  - [ ] `tests/unit/test_training/test_callbacks.py`

**Integration Tests**

- [ ] `tests/integration/test_data_pipeline.py`

  - [ ] End-to-end: raw data → tensors
  - [ ] Test with small dataset

- [ ] `tests/integration/test_training_loop.py`

  - [ ] Full training (1 epoch on small data)
  - [ ] Test checkpointing and resuming

- [ ] `tests/integration/test_inference.py`

  - [ ] Inference pipeline
  - [ ] Test with all backends

- [ ] `tests/integration/test_export.py`
  - [ ] ONNX export

**Regression Tests**

- [ ] `tests/regression/test_model_outputs.py`

  - [ ] Compare PyTorch vs TensorFlow outputs
  - [ ] Store baselines

- [ ] `tests/regression/test_metrics.py`
  - [ ] Verify metrics match TF baseline
  - [ ] Compare log loss scores

**Behavioural Tests (ML-Specific)**

- [ ] `tests/behavioural/test_invariances.py`

  - [ ] Brightness scaling invariance
  - [ ] Time translation invariance
  - [ ] Model shouldn't be sensitive to input order

- [ ] `tests/behavioural/test_convergence.py`

  - [ ] Overfit on single batch (should reach near 0 loss)
  - [ ] Verify model can learn

- [ ] `tests/behavioural/test_determinism.py`
  - [ ] Same seed → same output
  - [ ] Verify reproducibility

**Performance Tests**

- [ ] `tests/performance/test_speed.py`

  - [ ] Data loading speed benchmarks
  - [ ] Training throughput benchmarks
  - [ ] Inference latency benchmarks

- [ ] `tests/performance/test_memory.py`
  - [ ] Memory usage profiling
  - [ ] Peak memory tracking

**Test Coverage**

- [ ] Run pytest-cov
- [ ] Aim for >80% coverage
- [ ] Generate coverage reports
- [ ] Add coverage badge to README

**Property-Based Testing (Advanced)**

- [ ] Use Hypothesis for property-based tests
  - [ ] Test model with random inputs
  - [ ] Test data transformations with edge cases
  - [ ] Test numerical stability

##### 6.2 Documentation

**Code Documentation**

- [ ] Add docstrings to all modules (Google style)
  - [ ] Document all functions
  - [ ] Document all classes
  - [ ] Include examples in docstrings
- [ ] Add type hints throughout
- [ ] Add inline comments for complex logic

**User Documentation**

- [ ] Update README.md

  - [ ] Installation instructions
  - [ ] Quick start guide
  - [ ] Usage examples
  - [ ] Performance comparison (TF vs PyTorch)

- [ ] Create user guide

  - [ ] Data preparation
  - [ ] Training models
  - [ ] Hyperparameter tuning
  - [ ] Model export
  - [ ] Inference

- [ ] API reference
  - [ ] Auto-generate with mkdocstrings
  - [ ] Host with mkdocs-material

**Developer Documentation**

- [ ] Architecture overview

  - [ ] System design
  - [ ] Data flow
  - [ ] Model architectures

- [ ] Migration guide

  - [ ] TensorFlow → PyTorch differences
  - [ ] Weight conversion guide
  - [ ] Troubleshooting guide

- [ ] Contributing guidelines

  - [ ] Code style (Black, Ruff)
  - [ ] Testing requirements
  - [ ] PR process

- [ ] Testing guide
  - [ ] How to run tests
  - [ ] How to write tests
  - [ ] Coverage requirements

**Architecture Decision Records (ADRs)**

- [ ] Document key decisions
  - [ ] `docs/architecture/decisions/001-pytorch-over-tensorflow.md`
  - [ ] `docs/architecture/decisions/002-polars-over-pandas.md`
  - [ ] `docs/architecture/decisions/003-lightning-for-training.md`
  - [ ] `docs/architecture/decisions/004-dvc-for-data-versioning.md`
  - [ ] `docs/architecture/decisions/005-optuna-for-hyperparameter-search.md`

**Notebooks**

- [ ] Update existing notebooks for PyTorch

  - [ ] Evaluation notebooks
  - [ ] Visualization notebooks
  - [ ] Analysis notebooks

- [ ] Create new notebooks
  - [ ] Migration comparison (TF vs PyTorch)
  - [ ] Performance benchmarks
  - [ ] Model interpretation
  - [ ] Data exploration with Polars

##### 6.3 CI/CD Updates

- [ ] Update GitHub Actions workflows

  - [ ] PyTorch testing
  - [ ] Test on multiple Python versions (3.10, 3.11, 3.12)
  - [ ] Multi-platform tests (Linux, macOS)
  - [ ] GPU tests (if available)
  - [ ] Coverage reporting (Codecov)
  - [ ] ONNX export tests

- [ ] Add data pipeline tests to CI

  - [ ] Smoke test with small dataset
  - [ ] Schema validation tests

- [ ] Add model smoke tests to CI

  - [ ] Fast dev run (1 epoch, small data)
  - [ ] Test all three architectures

- [ ] Add linting and formatting
  - [ ] Black (formatting)
  - [ ] Ruff (linting)
  - [ ] mypy (type checking)
  - [ ] Pre-commit hooks

**Deliverables**:

- > 80% test coverage
- Comprehensive test suite (unit, integration, regression, behavioural)
- Complete documentation (code, user, developer)
- Updated CI/CD pipeline
- Architecture Decision Records
- Migration guide

**Success Criteria**:

- All tests passing
- Coverage >80%
- Documentation complete and hosted
- CI/CD running on all PRs
- No missing type hints

---

### Phase 7: Validation & Optimization

**Objective**: Validate migration and optimize performance

#### Tasks:

##### 7.1 Numerical Validation

- [ ] Compare PyTorch vs TensorFlow outputs

  - [ ] Load same weights in both frameworks
  - [ ] Test on same data (use fixed seed)
  - [ ] Compare layer-by-layer activations
  - [ ] Verify numerical equivalence (within tolerance)
  - [ ] Document any differences
  - [ ] Generate validation report

- [ ] Reproduce baseline results

  - [ ] Train all three models from scratch on PLAsTiCC
  - [ ] Compare with TF baseline metrics:
    - [ ] Log loss scores
    - [ ] Accuracy
    - [ ] Per-class metrics
    - [ ] Confusion matrices
  - [ ] Target: Match within 1%

- [ ] Statistical significance tests
  - [ ] Multiple runs with different seeds
  - [ ] Compute mean and std of metrics
  - [ ] T-test for significance

##### 7.2 Performance Optimization

- [ ] Profile data pipeline

  - [ ] Use torch.profiler
  - [ ] Identify bottlenecks
  - [ ] Optimize Polars operations
  - [ ] Tune DataLoader workers (num_workers)
  - [ ] Test persistent workers
  - [ ] Pin memory for faster GPU transfer

- [ ] Profile training loop

  - [ ] GPU utilization (nvidia-smi, nvtop)
  - [ ] Mixed precision benefits
  - [ ] Batch size optimization (find optimal)
  - [ ] Gradient accumulation (if needed for large models)

- [ ] Profile model

  - [ ] Forward pass profiling
  - [ ] Backward pass profiling
  - [ ] Identify slow layers
  - [ ] Consider Flash Attention for transformers (optional)

- [ ] Memory optimization
  - [ ] Gradient checkpointing (trade compute for memory)
  - [ ] Activation checkpointing
  - [ ] Efficient attention variants
  - [ ] Find maximum batch size

##### 7.3 Benchmarking

- [ ] Create comprehensive benchmark suite

  - [ ] Data loading speed
    - [ ] Time to load full dataset
    - [ ] Time per batch
  - [ ] Training throughput
    - [ ] Samples/sec
    - [ ] Batches/sec
    - [ ] Time per epoch
  - [ ] Inference latency
    - [ ] Single sample latency (p50, p95, p99)
    - [ ] Batch inference latency
  - [ ] Memory usage
    - [ ] Peak GPU memory
    - [ ] Peak RAM usage

- [ ] Compare with TensorFlow baseline

  - [ ] Data loading: Target 5-10x faster
  - [ ] Training: Target match or better
  - [ ] Inference: Target 2-5x faster (ONNX + quantization)
  - [ ] Memory: Target match or better

- [ ] Generate benchmark report
  - [ ] Tables comparing TF vs PyTorch
  - [ ] Plots showing performance
  - [ ] Include in documentation

##### 7.4 Stress Testing

- [ ] Test with large datasets

  - [ ] Full PLAsTiCC dataset
  - [ ] ELAsTiCC dataset (if available)
  - [ ] Verify no OOM errors

- [ ] Test multi-GPU training

  - [ ] Verify linear speedup
  - [ ] Test with 2, 4, 8 GPUs

- [ ] Test long training runs
  - [ ] Full 100 epoch run
  - [ ] Verify no memory leaks
  - [ ] Verify checkpointing works

##### 7.5 Optimization Recommendations

- [ ] Document optimization findings

  - [ ] Optimal hyperparameters
  - [ ] Optimal batch sizes
  - [ ] Optimal number of workers
  - [ ] GPU utilization tips

- [ ] Create optimization guide
  - [ ] How to profile
  - [ ] Common bottlenecks
  - [ ] How to optimize

**Deliverables**:

- Numerical validation report
- Performance comparison (TF vs PyTorch)
- Optimized implementation
- Benchmark results
- Optimization guide

**Success Criteria**:

- PyTorch models match TF baseline (within 1%)
- Data loading 5-10x faster (verified)
- Training throughput match or better
- Inference 2-5x faster with ONNX
- No OOM errors with full dataset
- All benchmarks documented

---

### Phase 8: Production Readiness (Optional)

**Objective**: Prepare for production deployment (if needed)

This phase is optional and only needed if deploying to production.

#### Tasks:

##### 8.1 Model Registry

- [ ] Set up model registry (MLflow recommended)
  - [ ] Version models
  - [ ] Track model metadata
  - [ ] Tag production models

##### 8.2 Inference Server

- [ ] Choose inference server

  - [ ] Option 1: Triton Inference Server (NVIDIA)
  - [ ] Option 2: TorchServe (PyTorch native)
  - [ ] Option 3: Custom FastAPI server

- [ ] Implement inference server
  - [ ] `inference/server/`
  - [ ] Load models
  - [ ] Handle requests
  - [ ] Batch requests
  - [ ] Health checks

##### 8.3 Deployment

- [ ] Containerization

  - [ ] Create Dockerfile
  - [ ] Optimize image size
  - [ ] Multi-stage builds

- [ ] Kubernetes deployment (optional)
  - [ ] Deployment manifests
  - [ ] Service configuration
  - [ ] Auto-scaling

##### 8.4 Monitoring

- [ ] Set up monitoring

  - [ ] Prometheus metrics
  - [ ] Grafana dashboards
  - [ ] Alert on errors

- [ ] Model monitoring
  - [ ] Track prediction distribution
  - [ ] Detect data drift
  - [ ] Track model performance

**Deliverables**:

- Model registry
- Inference server
- Deployment configuration
- Monitoring setup

---

## Technology Stack (REVISED)

### Core Dependencies

```yaml
Python: ">=3.10,<3.13"

# Deep Learning
torch: ">=2.1.0" # Latest stable PyTorch
torchvision: ">=0.16.0"
lightning: ">=2.1.0" # NEW: PyTorch Lightning

# Data Processing
polars: ">=0.20.0" # Modern DataFrame library
pyarrow: ">=15.0.0" # Apache Arrow for zero-copy
numpy: ">=1.24.0,<2.0" # Pin to avoid breaking changes

# Data Versioning (NEW)
dvc: ">=3.30.0" # Data version control
dvc-s3: ">=3.0.0" # Optional: S3 remote storage
dvc-gs: ">=3.0.0" # Optional: GCS remote storage

# Astronomy-specific
astropy: ">=5.3"
george: ">=0.4.0" # GP interpolation (current)
tinygp: ">=0.3.0" # NEW: Faster GP alternative (JAX-based)
# OR
celerite2: ">=0.3.0" # NEW: GP for time-series

# Training & Optimization
optuna: ">=3.5.0" # Hyperparameter optimization
torchmetrics: ">=1.3.0" # Metrics

# Experiment Tracking (choose one)
wandb: ">=0.16.0" # Weights & Biases (recommended)
# OR
mlflow: ">=2.10.0" # MLflow (self-hosted)

# Model Export
onnx: ">=1.15.0"
onnxruntime: ">=1.17.0"
onnxruntime-gpu: ">=1.17.0" # For GPU inference

# Configuration (SIMPLIFIED)
pydantic: ">=2.5.0" # Configuration validation & YAML
pydantic-settings: ">=2.1.0" # Settings management

# Utilities
rich: ">=13.7.0" # Beautiful terminal output
tqdm: ">=4.66.0" # Progress bars

# Scientific Computing
scipy: ">=1.11.0"
scikit-learn: ">=1.4.0"

# Testing
pytest: ">=7.4.0"
pytest-cov: ">=4.1.0"
pytest-xdist: ">=3.5.0" # Parallel testing
hypothesis: ">=6.98.0" # Property-based testing

# Code Quality
black: ">=24.0.0"
ruff: ">=0.2.0" # Fast linter (replaces flake8, isort, etc.)
mypy: ">=1.8.0" # Type checking
pre-commit: ">=3.6.0"

# Profiling (NEW)
py-spy: ">=0.3.0"
memory_profiler: ">=0.61.0"
torch-tb-profiler: ">=0.4.0"
memray: ">=1.11.0" # Memory profiling

# Optional: Advanced Features
# flash-attn: ">=2.5.0"  # Flash Attention (requires compilation)
# triton: ">=2.1.0"  # Custom CUDA kernels
```

### Development Tools

```yaml
# Notebooks
jupyter: ">=1.0.0"
jupyterlab: ">=4.0.0"
ipywidgets: ">=8.1.0"

# Documentation
mkdocs: ">=1.5.0"
mkdocs-material: ">=9.5.0"
mkdocstrings[python]: ">=0.24.0"
```

---

## File-by-File Migration Mapping (REVISED)

### Phase 0: Bridge Layer (NEW)

| New File                            | Purpose                        | Notes                     |
| ----------------------------------- | ------------------------------ | ------------------------- |
| `astronet/bridge/tf_to_pt.py`       | TF → PyTorch weight conversion | Critical for verification |
| `astronet/bridge/verify.py`         | Numerical equivalence checker  | Layer-by-layer comparison |
| `astronet/bridge/dual_inference.py` | Support both TF and PyTorch    | During transition         |
| `astronet/bridge/ensemble.py`       | Ensemble TF + PyTorch models   | Validation                |

### Data Processing

| Original File            | New File(s)                      | Notes                               |
| ------------------------ | -------------------------------- | ----------------------------------- |
| `astronet/preprocess.py` | `astronet/data/preprocessing.py` | Pure Polars, remove pandas          |
|                          | `astronet/data/validation.py`    | Schema validation                   |
| `astronet/datasets.py`   | `astronet/data/datasets.py`      | PyTorch Dataset classes             |
|                          | `astronet/data/loaders.py`       | Arrow-based DataLoaders (zero-copy) |
|                          | `astronet/data/transforms.py`    | Time-series transforms              |
|                          | `astronet/data/augmentation.py`  | Astronomy-specific augmentation     |
|                          | `astronet/data/samplers.py`      | Class-balanced sampling             |

### Models

| Original File                 | New File(s)                                  | Notes                     |
| ----------------------------- | -------------------------------------------- | ------------------------- |
| `astronet/t2/model.py`        | `astronet/models/t2.py`                      | PyTorch nn.Module         |
| `astronet/t2/transformer.py`  | `astronet/models/components/transformers.py` | Shared components         |
| `astronet/t2/attention.py`    | `astronet/models/components/attention.py`    | Use nn.MultiheadAttention |
| `astronet/tinho/funcmodel.py` | `astronet/models/tinho.py`                   | PyTorch nn.Module         |
| `astronet/atx/model.py`       | `astronet/models/atx.py`                     | PyTorch nn.Module         |
| `astronet/atx/layers.py`      | `astronet/models/components/`                | Split into components     |
|                               | `astronet/models/lightning_module.py`        | Lightning wrapper (NEW)   |

### Training

| Original File                  | New File(s)                      | Notes                     |
| ------------------------------ | -------------------------------- | ------------------------- |
| `astronet/train.py`            | `astronet/training/trainer.py`   | Lightning Trainer wrapper |
|                                | `astronet/scripts/train.py`      | CLI script                |
| `astronet/custom_callbacks.py` | `astronet/training/callbacks.py` | Lightning callbacks       |
| `astronet/metrics.py`          | `astronet/training/metrics.py`   | torchmetrics integration  |

### Optimization

| Original File                  | New File(s)                              | Notes                      |
| ------------------------------ | ---------------------------------------- | -------------------------- |
| `astronet/*/opt/hypertrain.py` | `astronet/optimization/optuna_search.py` | Unified HPO with Hyperband |
|                                | `astronet/optimization/search_spaces.py` | Search space configs       |

### Utilities

| Original File           | New File(s)                       | Notes            |
| ----------------------- | --------------------------------- | ---------------- |
| `astronet/utils.py`     | `astronet/utils/logging.py`       | Split by concern |
|                         | `astronet/utils/checkpointing.py` |                  |
|                         | `astronet/utils/config.py`        | Pydantic configs |
| `astronet/constants.py` | `astronet/constants.py`           | Keep as-is       |

---

## Configuration Management (SIMPLIFIED)

### Pydantic-Based Configuration (SIMPLIFIED from v1.0)

Use Pydantic V2 exclusively (no OmegaConf) for simpler configuration management.

```python
# astronet/models/configs.py
from pydantic import BaseModel, Field
import yaml

class T2Config(BaseModel):
    """Configuration for T2 model."""
    architecture: str = "t2"
    input_dim: tuple[int, int] = (100, 6)  # (timesteps, features)
    embed_dim: int = Field(64, gt=0)
    num_heads: int = Field(8, gt=0)
    ff_dim: int = Field(256, gt=0)
    num_filters: int = Field(32, gt=0)
    num_layers: int = Field(4, gt=0, le=12)
    num_classes: int = 14
    dropout: float = Field(0.1, ge=0.0, le=1.0)
    use_redshift: bool = True

    @classmethod
    def from_yaml(cls, path: str):
        """Load config from YAML file."""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: str):
        """Save config to YAML file."""
        with open(path, 'w') as f:
            yaml.safe_dump(self.model_dump(), f)

# Usage
config = T2Config.from_yaml("configs/models/t2.yaml")
model = T2Model(config)
```

### Example YAML Configuration

```yaml
# configs/models/t2.yaml
architecture: t2
input_dim: [100, 6]
embed_dim: 64
num_heads: 8
ff_dim: 256
num_filters: 32
num_layers: 4
num_classes: 14
dropout: 0.1
use_redshift: true
```

---

## Testing Strategy (ENHANCED)

### Test Organization (ENHANCED with ML-Specific Tests)

```
tests/
├── unit/                      # Fast, isolated tests
│   ├── test_data/
│   │   ├── test_preprocessing.py
│   │   ├── test_loaders.py
│   │   ├── test_transforms.py
│   │   ├── test_schema_validation.py  # NEW
│   │   └── test_augmentation.py        # NEW
│   ├── test_models/
│   │   ├── test_t2.py
│   │   ├── test_tinho.py
│   │   ├── test_atx.py
│   │   ├── test_architecture.py        # NEW: Shape checks
│   │   ├── test_numerical.py           # NEW: Numerical stability
│   │   └── test_components/
│   │       ├── test_embeddings.py
│   │       ├── test_attention.py
│   │       └── test_transformers.py
│   └── test_training/
│       ├── test_losses.py
│       ├── test_metrics.py
│       └── test_callbacks.py
│
├── integration/               # Multi-component tests
│   ├── test_data_pipeline.py
│   ├── test_training_loop.py
│   ├── test_inference.py
│   └── test_export.py                  # NEW
│
├── regression/                # Baseline comparisons
│   ├── test_model_outputs.py          # Compare with TF
│   ├── test_metrics.py
│   └── baselines/             # Stored baseline results
│
├── behavioural/               # NEW: ML-specific behavioural tests
│   ├── test_invariances.py            # Brightness, translation invariance
│   ├── test_convergence.py            # Overfit on small batch
│   └── test_determinism.py            # Reproducibility
│
├── performance/               # NEW: Performance tests
│   ├── test_speed.py                  # Speed benchmarks
│   └── test_memory.py                 # Memory profiling
│
└── fixtures/                  # Shared test fixtures
    ├── data.py
    └── models.py
```

### Example Behavioural Test

```python
# tests/behavioural/test_invariances.py
import torch
import pytest

def test_brightness_invariance(model, lightcurve):
    """Model should be invariant to brightness scaling."""
    # Original prediction
    with torch.no_grad():
        pred_original = model(lightcurve)
        class_original = pred_original.argmax(dim=1)

    # Scale brightness by 2x
    lightcurve_scaled = lightcurve * 2.0
    with torch.no_grad():
        pred_scaled = model(lightcurve_scaled)
        class_scaled = pred_scaled.argmax(dim=1)

    # Class should be the same
    assert class_original == class_scaled, \
        "Model predictions should be invariant to brightness scaling"
```

---

## Performance Targets (REVISED)

### Data Pipeline

- **Current**: ~5-10 min for full PLAsTiCC loading (pandas)
- **Target**: <1 min with Polars LazyFrame + Arrow
- **Metric**: 5-10x speedup (verified with benchmarks)

### Training

- **Current**: TensorFlow baseline (time per epoch)
- **Target**: Match or improve with PyTorch + Lightning + mixed precision
- **Metric**: Training time per epoch (samples/sec)

### Inference

- **Current**: TensorFlow baseline latency
- **Target**: 2-5x faster with ONNX Runtime + INT8 quantization
- **Metric**: Latency (p50, p95, p99)

### Memory

- **Current**: TensorFlow baseline peak memory
- **Target**: Match or better with efficient batching
- **Metric**: Peak GPU/RAM usage

### Model Quality

- **Target**: Match or exceed baseline metrics (within 1%)
  - Log Loss: ≤0.450 (tinho baseline)
  - Accuracy: ≥78.6%
  - CC-SNe contamination: ≤4.65%

---

## Risk Mitigation (ENHANCED)

### Technical Risks

| Risk                                     | Probability | Impact | Mitigation                                                            |
| ---------------------------------------- | ----------- | ------ | --------------------------------------------------------------------- |
| Numerical differences between TF/PyTorch | Medium      | High   | Phase 0 bridge layer, rigorous verification, load same weights        |
| Zero-copy not working as expected        | Medium      | Medium | Verify with memory profiling, use `zero_copy_only=True` flag          |
| GP interpolation performance regression  | Low         | Medium | Benchmark early, try tinygp/celerite2, implement fallback             |
| ONNX export issues                       | Medium      | Medium | Test export early in Phase 2, validate with ONNX Runtime              |
| Memory issues with large datasets        | Medium      | High   | Implement streaming, gradient checkpointing, activation checkpointing |
| Training instability                     | Low         | High   | Match original hyperparameters exactly, test on small data first      |
| Lightning overhead                       | Low         | Medium | Profile to verify no significant overhead                             |

### Process Risks

| Risk              | Probability | Impact | Mitigation                                             |
| ----------------- | ----------- | ------ | ------------------------------------------------------ |
| Scope creep       | Medium      | Medium | Stick to phases, defer nice-to-haves to post-migration |
| Testing gaps      | Low         | High   | Comprehensive test plan with >80% coverage requirement |
| Documentation lag | Medium      | Medium | Document as you go, not at the end                     |
| Timeline overrun  | Medium      | Medium | 4-6 week buffer built into timeline, regular check-ins |
| Team availability | Medium      | High   | Clear milestones, can pause between phases if needed   |

---

## Success Criteria (REVISED)

### Phase Completion Criteria

Each phase is considered complete when:

1. All tasks are checked off
2. Tests pass (unit + integration + relevant domain-specific tests)
3. Documentation is updated
4. Code review is complete (if team)
5. Performance benchmarks meet targets (if applicable)
6. Profiling completed (if applicable)

### Overall Success Criteria

The migration is successful when:

1. All three architectures work in PyTorch
2. Data pipeline uses pure Polars + Arrow with TRUE zero-copy (verified)
3. Data versioning with DVC in place
4. Zero-copy tensor conversion verified with memory profiling
5. Test coverage >80%
6. Training metrics match TF baseline (within 1%)
7. Data loading 5-10x faster (benchmarked)
8. Inference 2-5x faster with ONNX + quantization
9. ONNX export working for all models
10. Documentation complete (code, user, developer)
11. CI/CD updated and passing
12. No pandas dependencies in core code
13. Architecture Decision Records written
14. Migration guide complete
15. All numerical validation tests passing

---

## Phase Dependencies

| Phase                       | Dependencies          | Key Deliverables                          |
| --------------------------- | --------------------- | ----------------------------------------- |
| 0. Bridge & Verification    | None                  | Weight conversion, numerical verification |
| 1. Data Pipeline            | Phase 0 (for testing) | Pure Polars, zero-copy, DVC               |
| 2. Model Migration          | Phase 0, 1            | All models in PyTorch, verified           |
| 3. Training Infrastructure  | Phase 2               | Lightning training, callbacks             |
| 4. Hyperparameter Opt       | Phase 3               | Enhanced Optuna with Hyperband            |
| 5. Inference & Export       | Phase 2, 3            | ONNX export, quantization                 |
| 6. Testing & Docs           | All previous          | >80% coverage, complete docs              |
| 7. Validation               | All previous          | Numerical validation, benchmarks          |
| 8. Production (Optional)    | All previous          | Inference server, deployment              |

---

## Post-Migration Improvements

After the migration is complete, consider:

### Immediate Next Steps

- [ ] Deploy to production (if Phase 8 not done)
- [ ] Monitor model performance in production
- [ ] Gather user feedback on new codebase

### Future Enhancements

- [ ] Implement Flash Attention for longer sequences
- [ ] Explore variable-length sequences (no interpolation)
- [ ] Model distillation for deployment
- [ ] AutoML for architecture search (Ray Tune, Auto-PyTorch)
- [ ] Streaming inference for real-time alerts (Fink integration)
- [ ] Multi-task learning (classification + regression)
- [ ] Self-supervised pretraining
- [ ] Ensemble methods (stacking, boosting)
- [ ] Uncertainty quantification (MC Dropout, ensembles)
- [ ] Explainability tools (Captum, attention visualization)
- [ ] Model compression (pruning, knowledge distillation)

### Research Directions

- [ ] State-space models (S4, Mamba) for long sequences
- [ ] Test on ELAsTiCC dataset
- [ ] Cross-survey transfer learning (PLAsTiCC → ELAsTiCC)
- [ ] Few-shot learning for rare classes
- [ ] Active learning for efficient labeling
- [ ] Contrastive learning for representation
- [ ] Graph neural networks for host galaxy features

---

## Questions & Decisions Log

### Decisions Made (v2.0)

1. **Phase 0**: Add bridge layer for smooth transition
2. **Data format**: Arrow IPC for processed data
3. **Training framework**: PyTorch Lightning (not pure PyTorch)
4. **Configuration**: Pydantic only (not OmegaConf)
5. **HPO**: Enhanced Optuna with Hyperband pruning
6. **Primary export**: ONNX (deployment flexibility)
7. **Folder structure**: Separation of concerns (data/models/training)
8. **Testing**: pytest with >80% coverage + ML-specific tests
9. **Linting**: Ruff (fast, modern)
10. **Data versioning**: DVC
11. **Experiment tracking**: Weights & Biases (recommended)

### Open Questions

1. Which GP library to use?

   - Options: george (current), tinygp (JAX, faster), celerite2 (time-series)
   - Decision: Benchmark in Phase 1, choose based on performance

2. Flash Attention worth the complexity?

   - Decision: Optional optimization, not critical for initial migration
   - Add in post-migration improvements if needed

3. Deploy to production?
   - Decision: Optional Phase 8, depends on user requirements

---

## Appendix: Code Examples (UPDATED)

### Example 1: True Zero-Copy Arrow → PyTorch

```python
# astronet/data/loaders.py
import polars as pl
import pyarrow as pa
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class ArrowTimeSeriesDataset(Dataset):
    """Zero-copy dataset using Arrow IPC files.

    This implementation ensures TRUE zero-copy conversion:
    Arrow → NumPy view → PyTorch tensor
    """

    def __init__(
        self,
        arrow_file: str,
        feature_columns: list[str],
        label_column: str,
        verify_zero_copy: bool = True,
    ):
        # Load Arrow IPC file (memory-mapped)
        self.table = pa.ipc.open_file(arrow_file).read_all()
        self.feature_columns = feature_columns
        self.label_column = label_column

        # Pre-extract columns for faster access
        self.features_table = self.table.select(feature_columns)
        self.labels_table = self.table.select([label_column])

        if verify_zero_copy:
            self._verify_zero_copy()

    def _verify_zero_copy(self):
        """Verify that zero-copy conversion is possible."""
        for col in self.feature_columns:
            column = self.table.column(col)
            try:
                # This will raise if zero-copy is not possible
                _ = column.to_numpy(zero_copy_only=True)
            except Exception as e:
                raise ValueError(
                    f"Column {col} cannot be converted with zero-copy. "
                    f"Ensure the column is a numeric type without nulls. "
                    f"Error: {e}"
                )

    def __len__(self):
        return self.table.num_rows

    def __getitem__(self, idx):
        # Slice single row (still Arrow Table)
        row_features = self.features_table.slice(idx, 1)
        row_label = self.labels_table.slice(idx, 1)

        # Zero-copy conversion: Arrow → NumPy view
        # This creates a NumPy array that shares memory with Arrow
        feature_arrays = [
            row_features.column(col).to_numpy(zero_copy_only=True)
            for col in self.feature_columns
        ]

        # Stack features (creates a new array, but columns are zero-copy)
        X = np.stack(feature_arrays, axis=-1).squeeze(0)

        # Convert to torch tensor (zero-copy if possible)
        X_tensor = torch.from_numpy(X).float()

        # Label
        y = row_label.column(self.label_column).to_numpy(zero_copy_only=True)
        y_tensor = torch.from_numpy(y).long().squeeze()

        return X_tensor, y_tensor


def create_dataloader(
    arrow_file: str,
    feature_columns: list[str],
    label_column: str,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader from Arrow IPC file with zero-copy."""
    dataset = ArrowTimeSeriesDataset(
        arrow_file=arrow_file,
        feature_columns=feature_columns,
        label_column=label_column,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True,  # Keep workers alive between epochs
    )


# Usage
loader = create_dataloader(
    arrow_file='data/processed/train.arrow',
    feature_columns=['lsstg', 'lsstr', 'lssti', 'lsstz', 'lssty', 'lsstu'],
    label_column='class',
    batch_size=32,
    num_workers=4,
)
```

### Example 2: PyTorch Model with Lightning Wrapper

```python
# astronet/models/t2.py
import torch
import torch.nn as nn
from pydantic import BaseModel, Field


class T2Config(BaseModel):
    """Configuration for T2 model."""
    input_dim: tuple[int, int] = (100, 6)  # (timesteps, features)
    embed_dim: int = Field(64, gt=0)
    num_heads: int = Field(8, gt=0)
    ff_dim: int = Field(256, gt=0)
    num_layers: int = Field(4, gt=0)
    num_classes: int = 14
    dropout: float = Field(0.1, ge=0.0, le=1.0)
    use_redshift: bool = False


class T2Model(nn.Module):
    """Time-Transformer (T2) for astronomical transient classification.

    Pure PyTorch implementation, wrapped by Lightning for training.
    """

    def __init__(self, config: T2Config):
        super().__init__()
        self.config = config

        # Conv embedding
        self.embedding = nn.Conv1d(
            in_channels=config.input_dim[1],
            out_channels=config.embed_dim,
            kernel_size=3,
            padding=1,
        )

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.input_dim[0], config.embed_dim)
        )

        # Transformer encoder (modern PyTorch)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        # Classification head
        classifier_dim = config.embed_dim
        if config.use_redshift:
            classifier_dim += 2  # Add redshift features

        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_dim),
            nn.Dropout(config.dropout),
            nn.Linear(classifier_dim, config.num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Lightcurve tensor of shape (batch, timesteps, features)
            z: Optional redshift features of shape (batch, 2)

        Returns:
            Class logits of shape (batch, num_classes)
        """
        # Conv embedding: (B, T, F) → (B, F, T) → (B, D, T) → (B, T, D)
        x = x.transpose(1, 2)  # (B, F, T)
        x = self.embedding(x)  # (B, D, T)
        x = x.transpose(1, 2)  # (B, T, D)

        # Add positional encoding
        x = x + self.pos_encoding

        # Transformer encoding
        x = self.transformer(x)  # (B, T, D)

        # Global average pooling
        x = x.mean(dim=1)  # (B, D)

        # Optional: concatenate redshift features
        if z is not None and self.config.use_redshift:
            x = torch.cat([x, z], dim=1)  # (B, D+2)

        # Classification
        return self.classifier(x)  # (B, num_classes)


# astronet/models/lightning_module.py
import lightning as L
import torch
import torch.nn as nn
from torchmetrics import Accuracy, F1Score
from astronet.training.losses import WeightedCrossEntropyLoss


class AstroNetLightningModule(L.LightningModule):
    """Lightning wrapper for AstroNet models.

    This class handles training loop boilerplate while keeping
    the model implementation pure PyTorch.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Loss function
        self.criterion = WeightedCrossEntropyLoss(
            weight=class_weights,
            label_smoothing=0.1,
        )

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')

    def forward(self, x, z=None):
        return self.model(x, z)

    def training_step(self, batch, batch_idx):
        """Training step (Lightning handles backward, optimizer step)."""
        # Unpack batch
        if len(batch) == 2:
            x, y = batch
            z = None
        else:
            x, z, y = batch

        # Forward pass
        logits = self(x, z)
        loss = self.criterion(logits, y)

        # Metrics
        preds = logits.argmax(dim=1)
        acc = self.train_acc(preds, y)

        # Logging (Lightning handles this)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Unpack batch
        if len(batch) == 2:
            x, y = batch
            z = None
        else:
            x, z, y = batch

        # Forward pass
        logits = self(x, z)
        loss = self.criterion(logits, y)

        # Metrics
        preds = logits.argmax(dim=1)
        acc = self.val_acc(preds, y)
        f1 = self.val_f1(preds, y)

        # Logging
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
```

### Example 3: Training Script with Lightning

```python
# astronet/scripts/train.py
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger

from astronet.models.t2 import T2Model, T2Config
from astronet.models.lightning_module import AstroNetLightningModule
from astronet.data.loaders import create_dataloader


def train():
    # Configuration
    config = T2Config.from_yaml("configs/models/t2.yaml")

    # Data
    train_loader = create_dataloader(
        arrow_file='data/processed/train.arrow',
        feature_columns=['lsstg', 'lsstr', 'lssti', 'lsstz', 'lssty', 'lsstu'],
        label_column='class',
        batch_size=32,
        shuffle=True,
    )

    val_loader = create_dataloader(
        arrow_file='data/processed/val.arrow',
        feature_columns=['lsstg', 'lsstr', 'lssti', 'lsstz', 'lssty', 'lsstu'],
        label_column='class',
        batch_size=32,
        shuffle=False,
    )

    # Model
    model = T2Model(config)
    lightning_model = AstroNetLightningModule(
        model=model,
        num_classes=config.num_classes,
        learning_rate=1e-3,
        weight_decay=0.01,
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath='models/checkpoints',
            filename='t2-{epoch:02d}-{val_loss:.2f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            mode='min',
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]

    # Logger
    logger = WandbLogger(
        project='astronet',
        name='t2-plasticc',
        log_model=True,
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=100,
        accelerator='gpu',
        devices=-1,  # Use all GPUs
        strategy='ddp',  # Distributed Data Parallel
        precision='16-mixed',  # Mixed precision training
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(lightning_model, train_loader, val_loader)

    # Test (optional)
    # trainer.test(lightning_model, test_loader)


if __name__ == '__main__':
    train()
```

---

## References

### PyTorch Resources

- PyTorch Documentation: https://pytorch.org/docs/stable/
- PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/
- PyTorch Examples: https://github.com/pytorch/examples
- Best Practices: https://pytorch.org/tutorials/recipes/recipes_index.html

### Polars Resources

- Polars Documentation: https://pola-rs.github.io/polars/
- Polars Performance Guide: https://pola-rs.github.io/polars-book/
- Arrow Integration: https://arrow.apache.org/docs/python/

### Data Versioning

- DVC Documentation: https://dvc.org/doc
- DVC with PyTorch: https://dvc.org/doc/use-cases/versioning-data-and-model-files

### Experiment Tracking

- Weights & Biases: https://docs.wandb.ai/
- MLflow: https://mlflow.org/docs/latest/index.html

### Astronomy ML

- Fink Broker: https://fink-broker.org/

### Model Optimization

- ONNX Runtime: https://onnxruntime.ai/
- Quantization: https://pytorch.org/docs/stable/quantization.html

### Testing

- pytest: https://docs.pytest.org/
- Hypothesis: https://hypothesis.readthedocs.io/
- pytest-cov: https://pytest-cov.readthedocs.io/

---

## Contact & Support

- [issues](https://github.com/tallamjr/astronet/issues)

--

## Changelog

### Version 2.0 (2025-10-12)

- Added Phase 0 for bridge layer and migration verification
- Adopted PyTorch Lightning for training infrastructure
- Simplified configuration management (Pydantic only)
- Enhanced testing strategy with ML-specific tests
- Added data versioning with DVC
- Added experiment tracking (W&B/MLflow)
- Improved zero-copy implementation with verification
- Added profiling checkpoints
- Enhanced Optuna with Hyperband pruning
- Added architecture decision records
- Improved risk mitigation strategies

### Version 1.0 (2025-10-11)

- Initial migration plan
- 7 phases
- Pure PyTorch approach
- Basic testing strategy

---

**Document Version**: 2.0 (REVISED)
**Last Updated**: 2025-10-12
**Status**: Planning Phase
**Next Review**: Start of Phase 0
