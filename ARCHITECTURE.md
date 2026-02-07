# Project Architecture

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         MAIN.PY                                  │
│                   (Orchestration Layer)                          │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├──────────────────────────────────────────────────┐
             │                                                   │
             v                                                   v
┌────────────────────────┐                        ┌────────────────────────┐
│   DATA LAYER           │                        │   CONFIG LAYER         │
├────────────────────────┤                        ├────────────────────────┤
│ • DataLoader           │◄───────────────────────│ • config.py            │
│   - extract_dataset()  │                        │   - Hyperparameters    │
│   - verify_dataset()   │                        │   - Paths              │
│   - get_image_paths()  │                        │   - Model settings     │
└────────┬───────────────┘                        └────────────────────────┘
         │                                                   ▲
         │                                                   │
         v                                                   │
┌────────────────────────┐                                  │
│ PREPROCESSING LAYER    │                                  │
├────────────────────────┤                                  │
│ • ImageProcessor       │──────────────────────────────────┘
│   - load_and_resize()  │
│   - prepare_dataset()  │
│   - encode_labels()    │
│   - split_dataset()    │
└────────┬───────────────┘
         │
         │ (X_train, Y_train, X_val, Y_val)
         │
         v
┌────────────────────────┐
│   MODEL LAYER          │
├────────────────────────┤
│ • LungCancerCNN        │
│   - build_model()      │
│   - compile_model()    │
│   - save_model()       │
│   - load_model()       │
└────────┬───────────────┘
         │
         │ (compiled model)
         │
         v
┌────────────────────────┐                        ┌────────────────────────┐
│  TRAINING LAYER        │                        │   UTILS LAYER          │
├────────────────────────┤                        ├────────────────────────┤
│ • ModelTrainer         │◄───────────────────────│ • Callbacks            │
│   - train()            │                        │   - EarlyStopping      │
│   - get_summary()      │                        │   - ReduceLROnPlateau  │
│   - print_summary()    │                        │   - CustomCallback     │
└────────┬───────────────┘                        │                        │
         │                                         │ • Visualizer           │
         │ (trained model + history)               │   - visualize_images() │
         │                                         │   - plot_history()     │
         v                                         │   - plot_predictions() │
┌────────────────────────┐                        └────────────────────────┘
│ EVALUATION LAYER       │
├────────────────────────┤
│ • ModelEvaluator       │
│   - predict()          │
│   - evaluate()         │
│   - print_results()    │
│   - get_misclassified()│
└────────────────────────┘
```

## Data Flow

```
1. Dataset (ZIP) 
   ↓
2. DataLoader → Extract & Verify
   ↓
3. ImageProcessor → Load, Resize, Encode, Split
   ↓
4. LungCancerCNN → Build & Compile Model
   ↓
5. ModelTrainer → Train with Callbacks
   ↓
6. Visualizer → Plot Training History
   ↓
7. ModelEvaluator → Evaluate & Report Metrics
   ↓
8. Save Model → saved_models/lung_cancer_cnn_model.h5
```

## Module Dependencies

```
main.py
├── config.config
├── data.data_loader
│   └── config.config
├── preprocessing.image_processor
│   ├── config.config
│   └── data.data_loader
├── models.cnn_model
│   └── config.config
├── training.trainer
│   ├── config.config
│   └── utils.callbacks
├── evaluation.evaluator
│   └── config.config
└── utils.visualization
    └── config.config
```

## CNN Model Architecture

```
Input Image (256x256x3)
         ↓
┌─────────────────────┐
│  Conv2D (32 filters)│
│  Kernel: 5x5        │
│  Activation: ReLU   │
│  Padding: Same      │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  MaxPooling2D (2x2) │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Conv2D (64 filters)│
│  Kernel: 3x3        │
│  Activation: ReLU   │
│  Padding: Same      │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  MaxPooling2D (2x2) │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Conv2D (128 filters)│
│  Kernel: 3x3        │
│  Activation: ReLU   │
│  Padding: Same      │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  MaxPooling2D (2x2) │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│      Flatten        │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Dense (256 units)  │
│  Activation: ReLU   │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ BatchNormalization  │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│  Dense (128 units)  │
│  Activation: ReLU   │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│   Dropout (0.3)     │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ BatchNormalization  │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│   Dense (3 units)   │
│ Activation: Softmax │
└──────────┬──────────┘
           ↓
    Output Classes
    [lung_aca, lung_n, lung_scc]
```

## Class Relationships

```
┌──────────────────┐
│   DataLoader     │
└────────┬─────────┘
         │ uses
         ↓
┌──────────────────┐
│ ImageProcessor   │
└────────┬─────────┘
         │ provides data to
         ↓
┌──────────────────┐         ┌──────────────────┐
│ LungCancerCNN    │────────→│  ModelTrainer    │
└──────────────────┘  model  └────────┬─────────┘
                                       │ uses
                                       ↓
                              ┌──────────────────┐
                              │   Callbacks      │
                              └──────────────────┘

┌──────────────────┐
│ ModelEvaluator   │
└────────┬─────────┘
         │ uses
         ↓
┌──────────────────┐
│   Visualizer     │
└──────────────────┘
```

## Design Patterns Used

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Configuration Management**: Centralized configuration in `config/config.py`
3. **Class-Based Design**: Object-oriented approach for better encapsulation
4. **Pipeline Architecture**: Clear data flow from loading to evaluation
5. **Modular Design**: Independent, reusable components

## Key Features

- ✅ Modular architecture
- ✅ Centralized configuration
- ✅ Clear separation of concerns
- ✅ Reusable components
- ✅ Comprehensive documentation
- ✅ Easy to extend and maintain
- ✅ Type hints and docstrings
- ✅ Error handling
- ✅ Logging and progress tracking
