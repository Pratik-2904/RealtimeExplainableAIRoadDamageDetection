import numpy as np
import cv2

def estimate_mbtp_area(mask_points, depth_map, focal_length, pixel_area_constant=1.0):
    """
    Estimates the physical area of a defect (e.g., pothole) using the 
    Minimum Bounding Triangulated Pixel (MBTP) method.
    
    Formula: A ≈ Σ (Z_i^2 / f^2) * Δp
    where Z_i is the depth at pixel i, f is focal length, and Δp is pixel area constant.
    
    Args:
        mask_points: List of (u, v) tuples for the pixels belonging to the defect mask.
        depth_map: 2D numpy array of metric depths.
        focal_length: Camera focal length in pixels.
        pixel_area_constant: Constant Δp for the sensor pixel area.
        
    Returns:
        float: Estimated physical area in squared metric units (e.g., cm^2 if Z is cm).
    """
    area = 0.0
    for (u, v) in mask_points:
        # Clamp coordinates to ensure they fall within depth map bounds
        v = min(max(v, 0), depth_map.shape[0] - 1)
        u = min(max(u, 0), depth_map.shape[1] - 1)
        
        z_i = depth_map[v, u]
        area += ((z_i ** 2) / (focal_length ** 2)) * pixel_area_constant
    return area


def grade_severity(metric_depth_mm):
    """
    Assigns a severity grade based on IRC:82-2015 standards for potholes.
    
    Low:  < 25 mm     -> Routine Monitoring
    Medium: 25 - 50 mm -> Patching / Sealing
    High: > 50 mm     -> Emergency Repair
    
    Args:
        metric_depth_mm: Maximum or average depth of the pothole in mm.
        
    Returns:
        tuple: (Severity Grade string, Maintenance Action string)
    """
    if metric_depth_mm < 25:
        return "Low", "Routine Monitoring"
    elif 25 <= metric_depth_mm <= 50:
        return "Medium", "Patching / Sealing"
    else:
        return "High", "Emergency Repair"


def backproject_to_gps(u, v, z, K_inv, camera_height_m, vehicle_gps, heading_rad):
    """
    Maps an image pixel (u,v) to world GPS coordinates assuming a planar road surface (Z=0).
    
    Args:
        u, v: Image pixel coordinates of the defect center.
        z: Metric depth to the defect center.
        K_inv: Inverse intrinsic camera matrix (3x3).
        camera_height_m: Height of the camera from the ground plane in meters.
        vehicle_gps: Tuple of (Latitude, Longitude) of the vehicle.
        heading_rad: Vehicle heading in radians (0 is North, pi/2 is East).
        
    Returns:
        tuple: (Latitude, Longitude) of the defect.
    """
    # Create homogeneous pixel coordinate
    pixel_coords = np.array([u * z, v * z, z])
    
    # Cast ray through inverse camera intrinsic matrix K_inv
    camera_coords = K_inv @ pixel_coords
    
    # Simple planar assumption mapping (depends on exact camera tilt/yaw setup)
    # Assuming camera Z axis points forward, Y points down, X points right
    # camera_coords = [X_c, Y_c, Z_c]
    # The relative offset on the ground plane from directly below the camera:
    offset_x = camera_coords[0]  # East/West relative offset
    offset_y = camera_coords[2]  # North/South relative offset (Forward depth)
    
    # Add offset to vehicle GPS (simplified standard Earth radius approximation)
    EARTH_RADIUS_M = 6378137.0
    lat, lon = vehicle_gps
    
    # Rotate offset depending on vehicle heading
    # Assuming standard coordinate frame: X is right, Y is forward.
    # Heading: 0 = North. If vehicle faces North, offset_y is added to Latitude.
    dx = offset_x * np.cos(heading_rad) - offset_y * np.sin(heading_rad)
    dy = offset_x * np.sin(heading_rad) + offset_y * np.cos(heading_rad)
    
    delta_lat = (dy / EARTH_RADIUS_M) * (180.0 / np.pi)
    delta_lon = (dx / (EARTH_RADIUS_M * np.cos(np.pi * lat / 180.0))) * (180.0 / np.pi)
    
    return (lat + delta_lat, lon + delta_lon)

if __name__ == "__main__":
    # Simple unit test or playground
    fake_depth = np.ones((480, 640)) * 100.0 # 100 cm depth
    fake_mask = [(320, 240), (321, 240), (320, 241), (321, 241)]
    area = estimate_mbtp_area(fake_mask, fake_depth, focal_length=800.0, pixel_area_constant=1.0)
    print(f"Estimated Area: {area:.4f} cm^2")
    
    grade, action = grade_severity(30.0)
    print(f"Severity: {grade} -> {action}")
    
    # Identity inverse intrinsic
    K_inv = np.eye(3)
    gps = backproject_to_gps(320, 240, 10.0, K_inv, 1.5, (37.7749, -122.4194), 0.0)
    print(f"Projected GPS: {gps}")
