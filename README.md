# 🇱🇰 AI-Powered Intelligent Tourism Support System for Sri Lanka

> An intelligent web application that transforms how tourists explore Sri Lanka’s heritage — by identifying landmarks from photos, delivering AI-generated animated storytelling, predicting hyper-local weather, and recommending nearby attractions — all in one seamless experience.

Developed by **Group 17** —  
👨‍🎓 [Jaleel Muhiyadeen](https://github.com/jaleel-muhiyadeen-206) | 👨‍🎓 [Kevin Herath](https://github.com/kevinHerath123/)  
👩‍🎓 [Krishanthi Segar](https://github.com/your-gh) | 👩‍🎓 [Niveka Perera](https://github.com/your-gh)  
🎓 *BSc (Hons) Artificial Intelligence & Data Science — IIT in collaboration with RGU*  
📅 November 2024 | 📄 [SRS](./docs/SRS_G17.pdf) | 📄 [Proposal](./docs/PP_Group_17_updated.pdf)

---

## 🌟 Key Features

✅ **Landmark Recognition**  
Upload a photo → instantly get the landmark name & location (e.g., “Sigiriya – Matale”).

🎙️ **Digital Storyteller Engine** *(Jaleel’s Contribution)*  
Receive an **AI-generated animated video + voice narration** explaining the landmark’s history, culture, and significance.  

☀️ **Hyper-Local Weather Prediction**  
Get a 48-hour forecast + “visit suitability score” for your exact location.  

🗺️ **Smart Recommendations**  
Discover:  
- Similar attractions 
- Top-rated places  
- Nearby hotels/restaurants

---

## 🧠 Project Structure

```bash
├──  landmark_recognition/                 # Flask API + ML models
├── digital_story_teller/ 
├── weather_prediction/
├── recommendation/              # REST API endpoints
├── frontend/                # React web app (mobile-responsive)
│   │   └── (Frontend application)
├── docs/
│   ├── SRS_G17.pdf         # Software Requirements Specification
│   └── PP_Group_17_updated.pdf
└── README.md
