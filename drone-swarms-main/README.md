# 🚁 Drone Swarm Optimization under Inclement Weather using PSO

> A research-driven simulation of drone swarm coordination under dynamic environmental disturbances using Particle Swarm Optimization (PSO).

---

## 📌 Overview

This project models a multi-drone swarm system operating in a 3D environment under **inclement weather conditions** such as wind disturbances.  

Traditional swarm systems fail under such conditions — this project introduces:

👉 **Weather-aware optimization using PSO**  
👉 **Dynamic disturbance modeling**  
👉 **Performance evaluation through error metrics**

---

## 🧠 Key Contributions

- 🌪️ Dynamic **wind disturbance modeling**
- 🤖 **PSO-based optimization** for swarm stabilization
- ⚙️ **Toggle system (PSO ON/OFF)** for comparison
- 📊 Real-time **formation error tracking**
- 🧪 Multi-scenario experimentation:
  - No wind
  - Medium wind
  - Strong wind

---

## ⚙️ System Architecture


Drone Swarm → Formation Control → Weather Disturbance → PSO Optimization → Stable Formation


---

## 🔬 Methodology

1. Initialize drones in 3D space  
2. Apply formation control (line, circle, etc.)  
3. Introduce wind disturbance  
4. Apply PSO to minimize formation error  
5. Track convergence using error metrics  

---

## 📉 Experimental Results

### 🔹 Observations

| Scenario | Result |
|--------|-------|
| No Wind | Stable formation |
| Medium Wind | Slight disturbance |
| Strong Wind (No PSO) | ❌ Unstable (High error ~50) |
| Strong Wind (With PSO) | ✅ Stable (Low error ~4) |

---

## 📊 Performance Comparison

| Metric | Without PSO | With PSO |
|------|------------|----------|
| Error Magnitude | Very High | Low |
| Stability | Poor | High |
| Convergence | Slow | Fast |

---

## 📸 Output Visualizations

### 🌤️ No Wind
![No Wind](outputs/no_wind.png)

### 🌧️ Medium Wind
![Medium Wind](outputs/medium_wind.png)

### ⛈️ Strong Wind (No PSO)
![No PSO](outputs/strong_no_pso.png)

### 🤖 Strong Wind (With PSO)
![With PSO](outputs/strong_with_pso.png)

---

## 🛠️ Tech Stack

- Python  
- NumPy  
- Matplotlib  
- Tkinter  

---

## 📂 Project Structure


├── main.py
├── drone.py
├── pso_optimizer.py
├── metrics.py
├── behaviors/
├── visualizer.py
├── outputs/
├── README.md


---

## ▶️ How to Run

```bash
git clone https://github.com/Sanjayram3269/drone-swarm-optimization.git
cd drone-swarm-optimization
pip install numpy matplotlib
python main.py
🧪 Experiments

Modify in drone.py:

WEATHER_MODE = "none" / "medium" / "strong"
USE_PSO = True / False
🚀 Applications
Autonomous drone delivery
Military swarm coordination
Disaster response
Multi-agent robotic systems
🔮 Future Scope
Reinforcement Learning integration
Real-time sensor-based control
Obstacle avoidance
Real drone deployment
👨‍💻 Authors
Sanjayram P
Arindam Sushil Katoch
⭐ Project Highlights

✔ Real-world disturbance modeling
✔ Optimization + simulation + analysis
✔ Comparative study (with vs without PSO)

📜 License

This project extends an open-source drone swarm simulation with additional optimization and environmental modeling.

💥 Final Note

This project demonstrates how intelligent optimization techniques like PSO can maintain stability in complex, uncertain environments — bridging the gap between simulation and real-world swarm robotics.