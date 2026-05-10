# ✨ SkinAware — AI-Powered Skin Analysis Platform

> Upload a photo of your skin. Get instant AI-powered analysis — dryness, redness, pigmentation, texture, porosity — and receive personalized product recommendations for your unique skin type.

> 🏆 Submitted to **Microsoft IdeaCup Hackathon**

---

## 🌟 What It Does

Most people don't know their actual skin type or what's causing their skin issues. SkinAware solves this — upload a clear skin photo, and the AI analyzes it across 5 key attributes, assigns a concern level, generates a skin score, and recommends products tailored specifically to you.

---

## ✨ Features

- 🔬 **AI Skin Analysis** — detects Dryness, Redness, Pigmentation, Texture & Porosity
- 📊 **Skin Score** — personalized score out of 100
- ⚠️ **Concern Level Classification** — Low / Medium / High
- 💊 **Product Recommendations** — personalized based on your skin analysis
- 🕓 **Analysis History** — track your skin changes over time
- 🔐 **User Authentication** — secure login & signup

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React.js |
| Backend | FastAPI (Python) |
| Database | MongoDB |
| AI / Analysis | Azure OpenAI + Google Gemini API |
| Auth | JWT-based Authentication |

---

## 🏗️ Architecture Flow

```
User uploads skin photo
        ↓
FastAPI backend receives image
        ↓
Azure OpenAI + Gemini API analyze image
        ↓
Returns: Dryness, Redness, Pigmentation, Texture, Porosity scores
        ↓
Concern Level assigned (Low / Medium / High)
        ↓
Skin Score generated
        ↓
Personalized recommendations returned
        ↓
Results + History saved to MongoDB
```

---

## ⚙️ Installation & Setup

### Prerequisites

- Node.js 16+
- Python 3.9+
- MongoDB (local or Atlas)
- Azure OpenAI API Key
- Google Gemini API Key

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/skinaware.git
cd skinaware
```

### Step 2: Backend Setup (FastAPI)

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Frontend Setup (React)

```bash
cd frontend
npm install
npm run dev
```

### Step 4: Environment Variables

Create a `.env` file in `backend/`:

```env
MONGODB_URI=your_mongodb_connection_string
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
GEMINI_API_KEY=your_gemini_api_key
JWT_SECRET=your_jwt_secret
```

### Step 5: Run the App

```bash
# Backend
uvicorn main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm run dev
```

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:5173`

---

## 📁 Project Structure

```
skinaware/
├── backend/
│   ├── main.py              # FastAPI app entry point
│   ├── routes/              # API routes (auth, analysis, history)
│   ├── models/              # MongoDB schemas
│   ├── services/            # Azure OpenAI + Gemini integration
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/           # Home, Analyze, Results, History
│   │   ├── components/      # Reusable UI components
│   │   └── utils/           # API helpers
│   └── package.json
└── README.md
```

---

## 🔗 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/analyze` | Upload image & get skin analysis |
| GET | `/api/history` | Get user's analysis history |
| DELETE | `/api/history` | Delete all history |
| POST | `/api/auth/signup` | User registration |
| POST | `/api/auth/login` | User login |

---

## 💡 Key Concepts

- ✅ **Multimodal AI** — image input processed by Azure OpenAI + Gemini
- ✅ **FastAPI async** — high performance Python backend
- ✅ **JWT Authentication** — secure user sessions
- ✅ **MongoDB** — flexible document storage for analysis results
- ✅ **React** — clean, responsive frontend UI

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">Built with ✨ to help everyone understand their skin better</p>
