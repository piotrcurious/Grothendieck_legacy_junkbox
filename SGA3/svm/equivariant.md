This is a fantastic leap! You are moving directly from the math of **classification** (what is this thing?) to the math of **scene understanding** (what is this thing, and where is it in the world?).
To build this inference engine, we need to make a crucial mathematical correction to our previous framework: **pure invariant theory is no longer enough.**
When you quotient out a group to find an invariant (like we did with z = xy), you deliberately *destroy* the information about where the data lives on the orbit. If you want the system to output the inferred shape **plus** the rotation and offset, you don't just want an *invariant* system; you want an **equivariant** system.
Here is how we adapt the algebraic geometry lens to build exactly the engine you described.
### 1. The Math: From Invariance to Equivariance
We are now dealing with the Special Euclidean Group SE(2), which consists of rotations and translations in 2D space. A group element g = (R_\theta, \mathbf{t}) acts on a shape \mathbf{X} (a set of points \mathbf{x}_i) like this:

To get the shape, rotation, and offset, our inference engine must split the data into two parts:
 1. **The Quotient (Shape):** The invariant representation of the data X/G.
 2. **The Fiber (Pose):** The specific group element g \in SE(2) that maps a "canonical" version of the shape to the observed data.
In geometry, this is solved using the method of **Moving Frames** (or constructing a *cross-section* of the orbits). Instead of just looking for invariant polynomials, we define a "canonical pose" for every shape. We then calculate the mathematical operations required to drag the observed data back to that canonical pose.
### 2. Building the SE(2) Inference Engine
To build this computationally, we can decompose the group action into measurable geometric moments.
 * **Extracting Translation (\mathbf{t}):** The centroid of the shape is an *equivariant* feature with respect to translation. If you shift the shape, the centroid shifts by the exact same amount.
 * **Extracting Rotation (\theta):** The eigenvectors of the shape's covariance matrix point along its principal axes. As the shape rotates, these vectors rotate with it.
 * **Extracting the Invariant (Shape):** Once we subtract the centroid and rotate by the inverse of the principal angle, we are left with the canonical shape, which we can safely pass to a classifier.
### 3. The Working Code
Here is a complete, working toy inference engine that does exactly what you asked for. It takes a raw, transformed shape; extracts the pose parameters; normalizes the shape; and classifies it.
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff

# 1. Define Canonical "Base" Shapes (The Quotient Space Representatives)
# L-shape
shape_A = np.array([[0,0], [0,1], [0,2], [0,3], [1,0], [2,0]]) 
# T-shape
shape_B = np.array([[0,2], [1,2], [2,2], [1,1], [1,0], [1,-1]])

# Function to center and align a shape (Moving Frame method)
def canonicalize_shape(X):
    # Extract Translation: Centroid
    centroid = np.mean(X, axis=0)
    X_centered = X - centroid
    
    # Extract Rotation: Covariance & Eigenvectors
    cov = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by eigenvalue to consistently identify the "main" axis
    sort_idx = np.argsort(eigenvalues)[::-1]
    main_vector = eigenvectors[:, sort_idx[0]]
    
    # Calculate angle
    angle = np.arctan2(main_vector[1], main_vector[0])
    
    # Create inverse rotation matrix to bring it back to canonical pose
    c, s = np.cos(-angle), np.sin(-angle)
    R_inv = np.array([[c, -s], [s, c]])
    
    # The Invariant Shape
    X_canonical = X_centered @ R_inv.T
    
    return X_canonical, centroid, angle

# Pre-calculate canonical versions of our known classes
canon_A, _, _ = canonicalize_shape(shape_A)
canon_B, _, _ = canonicalize_shape(shape_B)

# 2. The Inference Engine
def inference_engine(observed_shape):
    # Step 1: Decompose into Pose and Invariant Shape
    invariant_shape, translation, rotation = canonicalize_shape(observed_shape)
    
    # Step 2: Classify the Invariant Shape
    # (Using Hausdorff distance as a simple geometric classifier)
    dist_A = max(directed_hausdorff(invariant_shape, canon_A)[0], 
                 directed_hausdorff(canon_A, invariant_shape)[0])
    dist_B = max(directed_hausdorff(invariant_shape, canon_B)[0], 
                 directed_hausdorff(canon_B, invariant_shape)[0])
                 
    inferred_class = "L-Shape" if dist_A < dist_B else "T-Shape"
    
    return {
        "class": inferred_class,
        "translation": translation,
        "rotation_rad": rotation,
        "normalized_shape": invariant_shape
    }

# 3. Test the Engine
rng = np.random.default_rng(42)

# Create a test case: A T-shape, rotated by 45 degrees, shifted to (10, -5)
true_angle = np.pi / 4
c, s = np.cos(true_angle), np.sin(true_angle)
R = np.array([[c, -s], [s, c]])
t = np.array([10.0, -5.0])

# Apply the group action g = (R, t)
observed_data = (shape_B @ R.T) + t

# Add a tiny bit of noise to simulate real data
observed_data += rng.normal(0, 0.05, observed_data.shape)

# Run Inference
results = inference_engine(observed_data)

print(f"Inferred Class:      {results['class']}")
print(f"Inferred Offset:     [{results['translation'][0]:.2f}, {results['translation'][1]:.2f}]")
print(f"Inferred Rotation:   {np.degrees(results['rotation_rad']):.2f} degrees")

```
### Why this is mathematically sound
In our earlier SVM example, the classifier just spat out a 0 or 1 based on the quotient coordinate. It didn't know *why* the data was where it was.
By upgrading to an **equivariant** approach using canonical forms, the neural network (or in our toy case, the principal component analysis) acts as a function that maps the input space to the group G. The classifier then strictly operates on X/G, while the pose parameters (R_\theta, \mathbf{t}) are extracted cleanly as secondary outputs.
This exact mathematical architecture is the foundation for modern equivariant neural networks (like SE(3)-Transformers used in predicting 3D molecular structures).
How would you handle a case where the object has intrinsic symmetries—like a perfect square—where the rotation angle becomes mathematically ambiguous because multiple different group elements map the shape back onto itself?
