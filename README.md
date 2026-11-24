# Marine Debris Edge Inference & Actuation Engine

An edge-native, real-time computer vision pipeline engineered for autonomous UAV coastal surveillance, marine hazard detection, and hardware-synchronized payload marker deployment.

---

## Architectural Topology

```
+--------------------+      +-------------------------+      +-------------------------+
|  Camera Feed       | ---> |  Frame Preprocessor     | ---> |  ONNX Edge Runtime      |
|  (IMX219 / Video)  |      |  (Letterbox / NCHW)     |      |  (YOLO Detection Head)  |
+--------------------+      +-------------------------+      +-------------------------+
                                                                          |
                                                                          v
+--------------------+      +-------------------------+      +-------------------------+
|  Physical Payload  | <--- |  Hardware PWM Actuator  | <--- |  Gated Bounding Logic   |
|  Drop Mechanism    |      |  (pigpio / GPIO 18)     |      |  (NMS + Confidence)     |
+--------------------+      +-------------------------+      +-------------------------+
```

---

## Key Features

- **Decoupled Edge Pipeline:** Complete separation of frame ingestion, tensor preprocessing, neural inference, and GPIO hardware control.
- **Hardware PWM Servo Control:** Sub-millisecond angle-to-pulse-width translation using `pigpio` with safe fallback to software simulation.
- **Simulation & Dry-Run Mode:** Full suite of testing modes (`--dry-run` and `--mock-inference`) enabling development on standard macOS/Linux workstations without physical drone hardware or proprietary weights.
- **Quantized Edge Footprint:** Engineered to execute on resource-constrained ARM Cortex-A53 microprocessors (Raspberry Pi 3B+) under 50 ms loop latency.

---

## Repository Structure

```
marine-debris-edge-inference/
├── cli.py                    # Production command-line entrypoint
├── marine_edge/              # Core modular library
│   ├── __init__.py           # Package exports
│   ├── actuator.py           # Hardware PWM servo controller with dry-run support
│   ├── camera.py             # Frame capture abstraction (camera, video, synthetic)
│   ├── detector.py           # ONNX inference wrapper with NMS post-processing
│   └── pipeline.py           # End-to-end surveillance pipeline orchestrator
├── tests/                    # Comprehensive unit tests
│   ├── test_actuator.py      # Servo pulse width and angle boundary tests
│   ├── test_detector.py      # Tensor preprocessing and NMS logic tests
│   └── test_pipeline.py      # End-to-end pipeline and trigger cooldown tests
├── requirements.txt          # Production dependencies
└── README.md
```

---

## Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Sanjyit-S/marine-debris-edge-inference.git
cd marine-debris-edge-inference

# Install dependencies
pip install -r requirements.txt
```

### 2. Run in Dry-Run Simulation Mode (No Drone Hardware Required)

You can run the full pipeline immediately using synthetic camera frames and simulated actuation:

```bash
python3 cli.py --mock-inference --dry-run --headless --max-frames 10
```

### 3. Run with Custom Video & ONNX Weights

```bash
python3 cli.py \
    --source coastal_patrol.mp4 \
    --weights custom_detector.onnx \
    --conf-threshold 0.45 \
    --servo-pin 18 \
    --dry-run
```

### 4. Running Test Suite

```bash
pytest tests/ -v
```

---

## Hardware Pinout Reference (Raspberry Pi 3B+)

- **Camera:** Raspberry Pi Camera Module V2 (Sony IMX219) via CSI ribbon cable.
- **Servo Signal:** GPIO 18 (Physical Pin 12, Hardware PWM0).
- **Servo Power:** Dedicated 5V external power supply (common ground tied to Pi Ground Pin 6).

---

## Intellectual Property & Confidentiality Notice

The proprietary neural network model weights (`*.onnx`), fine-tuned hyperparameters, and specialized coastal training image datasets are held under enterprise intellectual property protection and are omitted from this public repository. This codebase provides the production edge execution wrapper, preprocessing harness, and actuation pipeline.
