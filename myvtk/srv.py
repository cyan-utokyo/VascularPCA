import numpy as np

def srv(curve, dt=1.0):
    """Computes the square-root velocity representation (SRV) of a 3D curve.
    
    Args:
        curve: A NumPy array with shape (n, 3) representing the 3D points of the
            curve, where n is the number of points in the curve.
        dt: The time interval between consecutive points in the curve.
        
    Returns:
        A NumPy array with shape (n, 4) representing the SRV of the curve.
    """
    # Ensure that the curve has the correct shape
    curve = np.asarray(curve, dtype=float)
    if curve.shape[1] != 3:
        raise ValueError("Input curve must be 3D.")
    
    # Compute the differences between consecutive points
    differences = curve[1:] - curve[:-1]
    
    # Compute the velocities from the differences and time interval
    velocities = differences / dt
    
    # Compute the square roots of the velocities
    sqrt_velocities = np.sqrt(np.sum(velocities ** 2, axis=1))
    
    # Concatenate the curve points with the square root velocities
    srv = np.column_stack((curve, sqrt_velocities))
    
    return srv
