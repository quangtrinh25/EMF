Here are the objectives and methods of the paper entirely in English, formatted in Markdown:

## 1. Objectives

* [cite_start]**Propose a Novel Localization Approach:** Develop a 6-degrees-of-freedom (6-DoF) localization framework for wireless capsule endoscopy (WCE) based on electromagnetic induction and machine learning methods[cite: 21, 23, 67].
* [cite_start]**Overcome Traditional Limitations:** Address the difficulties faced by conventional nonlinear optimization methods, which heavily depend on initial conditions, tend to fall into local optima, and exhibit instability under dynamic capsule motion[cite: 22, 45].
* [cite_start]**Optimize Computational Resources:** Reduce the high training costs and large dataset requirements associated with deep learning models to maintain real-time performance and low inference latency[cite: 62, 66, 89].
* [cite_start]**Evaluate and Balance Performance:** Analyze various classical machine learning models to find the optimal balance between accuracy, inference speed, and training cost[cite: 69, 305].
* [cite_start]**Experimental Validation:** Prove the practical feasibility of the proposed framework by testing it on a real-world physical localization system[cite: 70, 306].

---

## 2. Methods

### 2.1. System Mathematical Modeling
* [cite_start]**Forward Problem Formulation:** The system is formulated using a nonlinear forward electromagnetic measurement model that maps the capsule's 6-DoF pose (Cartesian position $P\in\mathbb{R}^{3}$ and orientation matrix $R\in SO(3)$) to the induced electromotive force (EMF) signals[cite: 73, 74].
* [cite_start]**Measurement Output:** The output is represented as a 9-dimensional vector $\epsilon\in\mathbb{R}^{9}$, which consists of the peak EMF amplitudes measured by 3 orthogonal receiving (RX) coils excited by 3 transmitting (TX) coils operating at distinct frequencies[cite: 75].
* [cite_start]**Mapping Equation:** The electromagnetic induction physics governing this system is compactly expressed as $\epsilon=\mathcal{F}(P,R)$[cite: 76].

### 2.2. Machine Learning Framework
* [cite_start]**Inverse Mapping:** To avoid impractical analytical inversion, learning-based regression models are employed to approximate the inverse mapping from raw EMF measurements to the capsule pose: $(P,R)\approx\mathcal{F}^{-1}(\epsilon)$[cite: 86, 87].
* [cite_start]**Algorithms Tested:** The study evaluates four machine learning models: Decision Tree (DT), Random Forest (RF), AdaBoost, and XGBoost[cite: 128, 129, 130, 131].
* [cite_start]**Hyperparameter Tuning:** Models were implemented using the `scikit-learn` library in Python, and optimal combinations of hyperparameters were selected through cross-validation using the `GridSearchCV` method[cite: 90, 91].
* [cite_start]**Data Splitting:** The generated dataset is divided into 70% for training, 15% for validation, and 15% for testing[cite: 170].

### 2.3. Experimental Setup
* [cite_start]**Hardware Design:** Three function generators supply sinusoidal signals at frequencies of 4000, 4500, and 5000 Hz (with a 10 V peak-to-peak voltage) to the 3 TX coils[cite: 134].
* [cite_start]**Data Acquisition:** The capsule uses a 16-bit ADC, an STM32 microcontroller, and a 433 MHz RF transceiver to process and transmit EMF data wirelessly to a central computer[cite: 136].
* [cite_start]**Ground Truth System:** An ABB robotic arm is used as a reference system in conjunction with the central computer for accurate pose evaluation[cite: 148, 191].

### 2.4. Hardware Calibration and Performance Evaluation
* [cite_start]**Hardware Calibration:** To correct physical discrepancies between design parameters and the actual fabricated system (e.g., coil positions, number of turns, and assembly tolerances), a nonlinear least squares method with the Trust Region Reflective solver is used to recalibrate the system[cite: 200, 201, 281].
* [cite_start]**Test Scenarios:** Model performance is tested using simulated and real-world data tracking two trajectories: a 2D square spiral trajectory and a 3D circular spiral trajectory[cite: 126, 172].
* [cite_start]**Evaluation Metrics:** Accuracy is measured using Root Mean Square Error (RMSE) for both position and orientation, alongside the coefficient of determination ($R^{2}$) to measure the goodness of fit[cite: 174, 176]. [cite_start]Computational time (inference and training speed) is also compared across models[cite: 160, 287].