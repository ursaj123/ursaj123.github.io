---
# the default layout is 'page'
icon: fas fa-info-circle
order: 1
---

👋 I am a **research-leaning machine learning engineer** currently working as a **Data Scientist** at **Ola Electric**, where I build and deploy ML systems for real-world decision-making problems. I enjoy working at the intersection of **theory, experimentation, and production**, especially on problems where mathematical structure informs model design.

I graduated with a degree in **Mathematics and Computing from IIT (BHU)**. My academic and industry work has centered around **optimization, recommender systems, and modern deep learning**, with a growing focus on **LLMs** and **generative models**.

My long-term goal is to work in applied research roles where I can **translate ideas from papers into robust systems**, while also contributing back through research, writing, and open-source projects.

A concise academic and industry overview is available in my [resume](https://drive.google.com/file/d/1wjdnyFnEAZmIGREyhXG4F6xNiG23BQM0/view?usp=sharing). More details are outlined below.

<!-- https://drive.google.com/file/d/1wjdnyFnEAZmIGREyhXG4F6xNiG23BQM0/view?usp=sharing -->

---

## 🎓 Education

### **Indian Institute of Technology (BHU), Varanasi**  
**Integrated Dual Degree in Mathematics and Computing**  
*Nov 2020 - July 2025*  
**CGPA:** 9.01

**Relevant Coursework:**  
Data Structures and Algorithms, Operating Systems, Computer Systems Architecture,  
Graph Theory, Differential Equations, Set Theory, Topology, Real Analysis,  
Digital Image Processing, Machine Learning, Signal Processing

---

## 💼 Work Experience

### **Ola Electric**  
**Data Scientist**  
*July 2025 - Present*

#### **Online Vehicle Mass Estimation**
- Designed and deployed an **online vehicle mass estimation system** using real-world CAN data.
- Initially formulated the problem using **vehicle dynamics and state estimation** (RLS, Kalman Filtering), and later transitioned to **ML and DL approaches** due to model mismatch.
- Trained and evaluated models including **XGBoost, Random Forest, and LSTM**, supported by extensive **signal preprocessing** (FFT, IIR filtering, noise suppression).
- Integrated the final model into **Simulink**, generated embedded code, and flashed it onto the target microcontroller, achieving **±15 kg accuracy** under most operating conditions.

#### **Vehicle Digital Twin (Simulink)**
- Built a **system-level digital twin of the vehicle in Simulink** by integrating multiple ECUs into a unified model.
- Enabled faster experimentation and testing by providing a bridge between **physical modeling and learning-based components**.
- Developed strong intuition for **vehicular signal flow, control logic, and ECU interactions**, which later informed ML model design.

#### **IMU-Based Modeling (Ongoing)**
- Beginning work with **IMU data** to improve understanding of **vehicular dynamics and sensor-driven modeling**.
- Aiming to deepen exposure to **sensor fusion, signal processing, and real-time constraints** in embedded systems.
- This work supports my broader interest in **hardware-aware ML systems**, relevant to robotics and embodied AI.

---

### **Info Edge (Naukri.com)**  
**Data Science Intern**  
*May 2024 - July 2024*

- Worked on improving **job recommendation relevance** for India’s largest job search and recruitment platform.
- Designed a **structured text representation** for users and jobs using custom start and end tokens (inspired by BERT CLS-style encoding) to capture designations, experience, and profile attributes.
- Pre-trained a **BERT-based model** on large-scale recent user and job data (6-8 months), validating representation learning via masked token prediction.
- Fine-tuned the model for **user-job alignment** with **contrastive loss** using:
  - positives (applied jobs),
  - hard negatives (shown but not applied jobs),
  - soft negatives (random user-job pairs).
- Deployed the system using **FastAPI and HNSW-based vector search**, achieving up to **~80% lift in applications during A/B testing**, stabilizing at **~45% improvement** after scaling.

---

### **CluCloud**  
**Data Science Intern**  
*July 2023 - September 2023*

- Performed **exploratory data analysis** on financial and operational data for loan recovery workflows.
- Built **feature-rich customer representations** using financial history, geographic information, and identity-level attributes.
- Trained ML models to **predict customer behavior** and derive **behavior-based segmentations** for operational decision-making.
- Explored **vehicle routing problem (VRP)** formulations to optimize field visit schedules and reduce operational costs by batching nearby visits.
- Gained early exposure to **data-driven optimization** in real-world, cost-sensitive systems.

---

## 🔬 Research Experience

### **Extension of ADMM for Multi-Objective Optimization**  
*Jan 2025 - July 2025*

- Conducted an extensive **literature survey** on multi-objective optimization methods and accelerated variants of single-objective algorithms.
- Explored extending **ADMM** to multi-objective settings for problems with **convex, structured constraints** and non-differentiable components.
- Formulated and experimented with problems of the form  
  `min_{x,z} f(x) + g(z)  s.t.  Ax + Bz = c`,  
  evaluating Pareto fronts across multiple setups (identity and rectangular constraints).
- Observed strong performance in **smooth cases (g = 0)**, with challenges in **l1-regularized settings**, reflected in low hypervolume and coinciding Pareto points.

---

### **Indian Institute of Science (IISc)**  
**Research Intern**  
*Aug 2023 - Sep 2023*

- Investigated **distractor-aware fine-tuning** for robust domain learning under adversarial settings.
- Generated hard negative samples using **diffusion-based data generation techniques**.
- Fine-tuned models with **contrastive losses** to improve separation between positives and hard negatives.

---

### **Indraprastha Institute of Information Technology, Delhi (IIITD)**  
**Research Intern**  
*May 2023 - Jun 2023*

- Explored **low-resource code-mixed language translation**, focusing on data-centric approaches.
- Designed transliteration-based dictionaries and **joint vocabulary tokenization** strategies.
- Trained **Transformer-based models** and evaluated **parameter-efficient techniques** including distillation, quantization-aware training, and pruning.
