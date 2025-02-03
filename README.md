# **Binary Logistic Regression Speed-Up in Python with Numba**

## **Overview**
This project implements **Binary Logistic Regression** using **NumPy** and **Numba** to compare the speed of:
- **A traditional NumPy-based logistic regression model**
- **A parallelized, Numba-optimized version of the model**

The goal is to **accelerate computation using Numba's just-in-time (JIT) compilation** and **parallel execution**, making logistic regression more efficient, especially for large datasets.

The dataset used in this project is the **Breast Cancer dataset** from `sklearn.datasets`.

---

## **Project File**
- **logistic_regression_speedup.ipynb**: The main Jupyter Notebook containing:
  - Implementation of **standard logistic regression**
  - Implementation of **Numba-accelerated logistic regression**
  - Performance comparison and analysis

---

## **Installation & Setup**
### **1. Install Jupyter Notebook**
If Jupyter Notebook is not installed, install it with:

pip install notebook

Launch Jupyter Notebook:

jupyter notebook

Open **logistic_regression_speedup.ipynb**.

### **2. Install Required Libraries**
Ensure all required Python packages are installed:

pip install numpy scikit-learn numba

---

## **How to Run the Notebook**
1. Open **logistic_regression_speedup.ipynb** in Jupyter Notebook.
2. Run each cell sequentially to:
   - Load and preprocess the dataset.
   - Implement logistic regression using NumPy.
   - Implement an optimized version using Numba.
   - Compare execution times and accuracy.

---

## **Implementation Details**
### **1. Standard Logistic Regression**
- Uses **NumPy** for matrix operations.
- Implements **gradient descent** for optimization.
- Computes **loss, gradients, and updates weights** iteratively.

### **2. Numba-Optimized Logistic Regression**
- **Parallelized computations using `@njit(parallel=True)`**
- **Optimized mathematical operations with `fastmath=True`**
- **Accelerates loss computation, weight updates, and gradient calculations**

---

## **Performance Comparison**
| Model | Avg. Time per Epoch (seconds) | Accuracy (%) |
|-------|-------------------------------|-------------|
| NumPy Logistic Regression | 0.0066 | ~97% |
| Numba-Optimized Logistic Regression | 0.0048 | ~97% |

**Key Findings:**
- The **Numba-optimized model is ~27% faster** than the standard NumPy version.
- Initial **Numba overhead is noticeable** when replacing NumPy’s vectorized functions.
- **Significant speedup occurs in compute-heavy operations** like **loss calculation**.

---

## **Key Optimizations & Learnings**
### **1. Parallelizing with `prange`**
- Numba's **parallel execution (`prange`)** enables multiple threads to process loops.
- The biggest performance improvement came from **modifying the loss function**:

Old Code:
  
  epsilon = 1e-9  
  loss = y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon)  
  return -np.mean(loss)

New Optimized Code with Numba:

  @njit(parallel=True, fastmath=True)  
  def compute_loss(y_train, y_pred):  
      epsilon = 1e-9  
      m = len(y_train)  
      loss = 0.0  
      y_train = y_train.reshape(-1)  
      y_pred = y_pred.reshape(-1)  

      for i in prange(m):  
          loss += y_train[i] * math.log(y_pred[i] + epsilon) + (1 - y_train[i]) * math.log(1 - y_pred[i] + epsilon)  

      return -loss / m  

- This change **reduced the compute time significantly** compared to using NumPy’s built-in vectorized functions.

### **2. Using `fastmath=True`**
- `fastmath=True` enables floating-point optimizations that **speed up calculations**.
- Applied in compute-heavy operations like **loss calculation and weight updates**.

### **3. Numba Overhead Considerations**
- Some **NumPy operations are already highly optimized**, making Numba unnecessary.
- Replacing NumPy’s built-in vectorized functions with Numba sometimes **increased** execution time due to compilation overhead.

### **4. Addressing Overflow in the Sigmoid Function**
- Encountered the warning:  
  "RuntimeWarning: overflow encountered in exp return 1 / (1 + np.exp(-x))"
- This happens when **x is a very large negative value**, causing `e^-x` to exceed floating-point limits.
- Considered **clipping values** but left it unchanged for now.

---

## **Future Improvements**
- Implement **multi-threading in weight updates** to enhance optimization.
- Apply **Numba to additional mathematical operations** in logistic regression.
- Explore **GPU acceleration with CuPy** for further speed improvements.
