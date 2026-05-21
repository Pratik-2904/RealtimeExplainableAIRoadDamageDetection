# DPR-RIA Architecture Diagrams

These Mermaid diagrams map out the high-level system flow and low-level component designs for the Dual-Path Road Infrastructure Analytics (DPR-RIA) system. They have been colored using your system's UI palette (Neon Blues, Crimson, and Amber) so they will look stunning on dark-mode presentation slides.

## 1. High-Level System Architecture

This diagram illustrates the end-to-end flow from the camera input through the DL engine, the analytics filters, and finally to the dashboard presentation and AI auditor.

```mermaid
graph TD
    %% Styling based on Cyber-Tactical UI
    classDef input fill:#10131a,stroke:#00f2ff,stroke-width:2px,color:#fff,rx:8px,ry:8px
    classDef core fill:#1d2026,stroke:#ff506e,stroke-width:2px,color:#fff,rx:8px,ry:8px
    classDef tracker fill:#1d2026,stroke:#ffb866,stroke-width:2px,color:#fff,rx:8px,ry:8px
    classDef ui fill:#1d2026,stroke:#74f5ff,stroke-width:2px,color:#fff,rx:8px,ry:8px
    classDef cluster fill:none,stroke:#3a494b,stroke-width:1px,stroke-dasharray: 5 5
    
    A[Camera Feed / Video Stream]:::input --> B(Dual-Path YOLO Backbone):::core
    
    subgraph Core DL Engine
        B -->|Frame Data| C[Real-Time Safety Branch]:::core
        B -->|High-Res Slices| D[Auditor Reports Branch]:::core
    end
    
    C --> E(TSCM CDKF Tracker):::tracker
    D --> E
    
    subgraph Analytics & Tracking
        E --> F[MBTP Physical Area Calc]:::tracker
        F --> G[GPS Back-projection]:::tracker
        G --> H[IRC:82 Severity Grading]:::tracker
    end
    
    H --> I[Flask / Socket.IO Backend]:::ui
    H --> J[GeoJSON Exporter]:::ui
    
    subgraph Presentation & AI
        I --> K[Stitch Cyber-Tactical Dashboard]:::ui
        J --> L[Gemini Auditor AI Chat & Report]:::ui
    end
```

---

## 2. LLD: Deep Learning Dual-Path YOLO Backbone

This diagram breaks down the custom PyTorch modifications. It highlights the `SPD-Conv` for retaining small defect data, the `BMS-SPPF` for spatial pooling, and the dual inference paths.

```mermaid
graph LR
    %% Styling
    classDef layer fill:#191c22,stroke:#00dbe7,stroke-width:2px,color:#fff,rx:5px,ry:5px
    classDef branch fill:#272a31,stroke:#ff506e,stroke-width:2px,color:#fff,rx:5px,ry:5px
    classDef output fill:#10131a,stroke:#00f2ff,stroke-width:3px,color:#fff,rx:10px,ry:10px

    Input[Input Image]:::layer --> SPD[SPD-Conv Module]:::layer
    SPD --> Backbone[YOLOv11 Backbone]:::layer
    Backbone --> BMS[BMS-SPPF Module]:::layer
    
    BMS --> P2[P2 Head 160x160]:::layer
    BMS --> P3[P3 Head 80x80]:::layer
    BMS --> P4[P4 Head 40x40]:::layer
    BMS --> P5[P5 Head 20x20]:::layer
    
    P2 & P3 & P4 & P5 --> Split{Dual-Path Split}
    
    Split --> B1[Path 1: Real-Time Edge]:::branch
    B1 --> TRT[TensorRT INT8 Optimizer]:::branch
    TRT --> O1[Low Latency BBoxes]:::output
    
    Split --> B2[Path 2: AI Auditor]:::branch
    B2 --> DA[DepthAnything V2]:::branch
    B2 --> GC[Grad-CAM Heatmaps]:::branch
    DA & GC --> O2[Rich Explainable Metadata]:::output
```

---

## 3. LLD: TSCM (Temporal-Spatial Consistency Module)

This diagram details the logic for the CDKF (Custom Dual Kalman Filter). It explains how the system avoids counting the same pothole multiple times by updating existing tracks based on geographic Haversine distance.

```mermaid
graph TD
    %% Styling
    classDef process fill:#191c22,stroke:#00f2ff,stroke-width:2px,color:#fff,rx:5px,ry:5px
    classDef db fill:#32353c,stroke:#ffb866,stroke-width:2px,color:#fff,rx:10px,ry:10px
    classDef condition fill:#272a31,stroke:#ff506e,stroke-width:2px,color:#fff,rx:10px,ry:10px

    Det[New Frame Detections]:::db --> Match{Data Association}:::condition
    Active[(Active Tracked Assets)]:::db --> Match
    
    Match -- "Dist < 3.0m (Haversine)" --> Update[Update Existing Asset]:::process
    Match -- "No Match" --> Create[Create New Track]:::process
    
    Update --> KF[Apply Dual Kalman Filter]:::process
    Create --> KF
    
    KF -->|Average Depth/Area| Active
    
    Active --> Stale{Unmatched > 40 Frames?}:::condition
    Stale -- Yes --> Archive[Prune & Archive]:::process
    Stale -- No --> Active
    
    Archive --> Exp[(Final Export Inventory)]:::db
```

---

## 4. LLD: Live Web Dashboard & GenAI Flow

This diagram maps out the interaction between the frontend client, the Flask/Socket.IO backend running the background video processing thread, and the external Gemini 2.5 API for AI insights.

```mermaid
graph TD
    %% Styling
    classDef client fill:#10131a,stroke:#74f5ff,stroke-width:2px,color:#fff,rx:8px,ry:8px
    classDef server fill:#272a31,stroke:#00dbe7,stroke-width:2px,color:#fff,rx:8px,ry:8px
    classDef ai fill:#32353c,stroke:#ff506e,stroke-width:2px,color:#fff,rx:8px,ry:8px
    classDef doc fill:#1d2026,stroke:#ffb866,stroke-width:2px,color:#fff,rx:0px,ry:0px

    Client[Browser: Cyber-Tactical UI]:::client <-->|WebSocket: frame_update| Server[Flask + Socket.IO Server]:::server
    Client -->|POST /chat or /generate_report| Server
    
    subgraph Background Thread
        Server -->|Spawn Thread| CV[OpenCV Capture]:::server
        CV --> Model[Model Inference & TSCM]:::server
        Model -->|Throttle to ~15fps| Encoder[Base64 JPEG Encoder]:::server
        Encoder --> Server
    end
    
    Model -->|Pipeline Complete| GeoJSON[/output_assets.geojson/]:::doc
    GeoJSON -->|Read Document| Server
    
    Server -->|Context + User Prompt| Gemini[Gemini 2.5 Flash API]:::ai
    Gemini -->|LLM Report / Chat Response| Server
```
