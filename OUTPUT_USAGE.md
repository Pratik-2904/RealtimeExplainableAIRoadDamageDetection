# Output Usage Guide

The DPR-RIA system produces several outputs designed specifically for civil engineers, municipal bodies, and road maintenance departments.

## 1. Live Web Dashboard (Stitch Cyber-Tactical UI)
**Target Audience:** Dispatchers and System Auditors
The live dashboard (`http://localhost:5001`) provides an immediate, real-time overview of road conditions as the vehicle drives.
- **Detections Stream**: Used to monitor the frequency and severity of damage.
- **Critical Defects Counter**: Alerts the auditor immediately if a high-priority defect (e.g., a deep pothole) is detected, allowing for immediate emergency dispatch.

## 2. GeoJSON Export (`results/output_assets.geojson`)
**Target Audience:** GIS Teams and Route Planners
Upon completion of a video or live stream, the system exports a deduplicated GeoJSON file.
- **Usage**: This file can be directly imported into QGIS, ArcGIS, or Google Earth.
- **Content**: Each feature is a Point geometry representing the exact location of the defect, with properties detailing the severity (`High`, `Medium`, `Low`), physical area (`cm²`), and depth (`mm`). Planners use this to group repairs geographically to optimize contractor dispatch routes.

## 3. Processed Video Overlay (`results/output.mp4`)
**Target Audience:** Civil Engineers and Quality Assurance
A copy of the raw video is saved with bounding boxes, confidence scores, and severity color-coding overlaid.
- **Usage**: Before approving an expensive patching contract based on the GeoJSON data, engineers can scrub to the exact timestamp in the video to visually verify the defect's nature.
- **Color Coding**: Red (Critical/Emergency Repair), Orange (Moderate/Patching), Green (Minor/Routine Monitoring).

## 4. Maintenance Action Recommendations
**Target Audience:** Procurement and Budgeting
The system automatically classifies defects based on the **IRC:82-2015** standards:
- **< 25 mm depth**: Routine Monitoring (No immediate budget required).
- **25 - 50 mm depth**: Patching / Sealing (Routine maintenance budget).
- **> 50 mm depth**: Emergency Repair (Emergency funds required).
This allows automated budgeting tools to ingest the severity data and estimate required repair materials.
