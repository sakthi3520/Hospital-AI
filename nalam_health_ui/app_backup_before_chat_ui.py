import streamlit as st
import requests
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from datetime import date

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="Nalam Health AI", page_icon="🏥", layout="wide")
API_URL = "http://127.0.0.1:8000"

# ---------------- API FUNCTIONS ----------------
def api_get(endpoint, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.get(f"{API_URL}{endpoint}", headers=headers, timeout=15)
    response.raise_for_status()
    return response.json()


def api_post(endpoint, payload, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.post(f"{API_URL}{endpoint}", json=payload, headers=headers, timeout=15)
    response.raise_for_status()
    return response.json()


def api_put(endpoint, payload, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.put(f"{API_URL}{endpoint}", json=payload, headers=headers, timeout=15)
    response.raise_for_status()
    return response.json()


def show_api_error(error):
    if isinstance(error, requests.exceptions.HTTPError):
        response = error.response
        if response is not None:
            st.error(f"API Error: {response.status_code}")
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            st.code(str(detail))
    elif isinstance(error, requests.exceptions.ConnectionError):
        st.error("FastAPI connect aagala. Backend server running-aa check pannu.")
    else:
        st.error(f"Error: {error}")


def get_doctors():
    data = api_get("/doctors")
    return data if isinstance(data, list) else []


def get_status_label(status):
    return {"Confirmed": "✅ Confirmed", "Rejected": "❌ Rejected", "Pending": "⏳ Pending"}.get(status, status)


# ---------------- JITSI MEETING LINK ----------------
def get_jitsi_meeting_link(appointment_id):
    safe_id = "".join(
        char for char in appointment_id.upper()
        if char.isalnum() or char == "-"
    )
    room_name = f"NalamHealth-{safe_id}"
    return f"https://meet.jit.si/{room_name}"

# ---------------- SESSION STATE ----------------
if "doctor_session" not in st.session_state:
    st.session_state.doctor_session = None

# ---------------- APP HEADER ----------------
st.title("🏥 NALAM Health AI")
st.caption("Smart Hospital Appointment Management Demo")
st.info(
    "Educational demo only. This app does not provide a medical diagnosis. "
    "Do not use real patient data. Doctor API endpoints must also enforce token "
    "verification before this app is suitable for deployment."
)

# ---------------- NALAM CHAT ASSISTANT ----------------

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! 👋 Welcome to Nalam Health AI.\n\n"
                "What health issue are you experiencing today? "
                "You can describe your symptoms in simple words."
            )
        }
    ]

with st.container(border=True):
    st.subheader("🩺 Nalam Health Assistant")
    st.caption("General health information and doctor consultation support")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_message = st.chat_input(
        "Describe your symptoms...",
        key="nalam_symptom_chat"
    )

    if user_message:
        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_message
        })

        with st.chat_message("user"):
            st.markdown(user_message)

        assistant_reply = (
            "Thank you for sharing your concern. 💙\n\n"
            "I can provide general health information, but I cannot "
            "diagnose a medical condition.\n\n"
            "**Would you like to consult a doctor?**\n\n"
            "You can use the Appointment Booking tab below to request "
            "a consultation. Your appointment will remain pending until "
            "the doctor responds."
        )

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": assistant_reply
        })

        with st.chat_message("assistant"):
            st.markdown(assistant_reply)

    if st.button("🗑️ Clear Chat", key="nalam_clear_chat"):
        st.session_state.chat_messages = []
        st.rerun()


# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🧑 Patient Registration", "📅 Appointment Booking", "🔎 Appointment Status", "👨‍⚕️ Doctor Login / Dashboard"
])

# ==================================================
# TAB 1: PATIENT REGISTRATION
# ==================================================
with tab1:
    st.subheader("🧑 Patient Registration")
    with st.form("patient_registration"):
        patient_name = st.text_input("Patient Name")
        age = st.number_input("Age", min_value=1, max_value=120, value=25)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        symptoms = st.text_area("Describe your symptoms", placeholder="Example: Fever and cough")
        register_clicked = st.form_submit_button("Register Patient")

    if register_clicked:
        if not patient_name.strip() or not symptoms.strip():
            st.warning("Name and symptoms enter pannu.")
        else:
            try:
                result = api_post("/patients", {
                    "patient_name": patient_name.strip(), "age": int(age),
                    "gender": gender, "symptoms": symptoms.strip()
                })
                st.success("Patient registered successfully!")
                st.write("**Patient ID:**", result["patient_id"])
                st.info("Patient ID-a save pannikko.")
            except requests.exceptions.RequestException as error:
                show_api_error(error)

# ==================================================
# TAB 2: APPOINTMENT BOOKING
# ==================================================
with tab2:
    st.subheader("📅 Book an Appointment")
    patient_id = st.text_input("Patient ID", placeholder="Example: PAT-D21BC4C9", key="booking_patient_id")
    try:
        doctors = get_doctors()
        active_doctors = [d for d in doctors if d.get("is_active", False)]
        if not active_doctors:
            st.warning("Active doctors list empty-aa irukku.")
        else:
            doctor_options = {
                f'{d["doctor_name"]} — {d["specialty"]} ({d["doctor_id"]})': d["doctor_id"]
                for d in active_doctors
            }
            selected_doctor = st.selectbox("Choose Doctor", list(doctor_options.keys()))
            appointment_date = st.date_input("Appointment Date", min_value=date.today())
            appointment_time = st.selectbox("Appointment Time", ["09:00", "10:00", "11:00", "12:00", "14:00", "15:00", "16:00"])
            if st.button("📅 Book Appointment"):
                if not patient_id.strip():
                    st.warning("Patient ID enter pannu.")
                else:
                    result = api_post("/appointments", {
                        "patient_id": patient_id.strip(),
                        "doctor_id": doctor_options[selected_doctor],
                        "appointment_date": appointment_date.isoformat(),
                        "appointment_time": appointment_time
                    })
                    st.success("Appointment request submitted!")
                    st.write("**Appointment ID:**", result["appointment_id"])
                    st.warning("Status: Pending ⏳")
                    st.info("Appointment ID save pannikko. Doctor decision varaikkum Pending.")
    except requests.exceptions.RequestException as error:
        show_api_error(error)
    except (KeyError, TypeError, ValueError) as error:
        st.error(f"Doctor data error: {error}")

# ==================================================
# TAB 3: APPOINTMENT STATUS
# ==================================================
with tab3:
    st.subheader("🔎 Check Appointment Status")
    appointment_id = st.text_input("Appointment ID", placeholder="Example: APT-4B52D882", key="status_appointment_id")
    if st.button("Check Appointment Status"):
        if not appointment_id.strip():
            st.warning("Appointment ID enter pannu.")
        else:
            try:
                result = api_get(f"/appointments/{appointment_id.strip()}")
                st.success("Appointment details found!")
                st.write("**Appointment ID:**", result["appointment_id"])
                st.write("**Patient ID:**", result["patient_id"])
                st.write("**Doctor ID:**", result["doctor_id"])
                st.write("**Date:**", result["appointment_date"])
                st.write("**Time:**", result["appointment_time"])
                status = result["status"]
                if status == "Confirmed":
                    st.success("Appointment Confirmed ✅")
                    
                    meeting_link = get_jitsi_meeting_link(result["appointment_id"])
                    st.link_button("🎥 Join Video Consultation", meeting_link)
                    st.info("Doctor-kum same meeting link-ai share pannu.")

                elif status == "Rejected":
                    st.error("Appointment Rejected ❌")
                elif status == "Pending":
                    st.warning("Appointment Pending ⏳")
                else:
                    st.write("Status:", status)
            except requests.exceptions.RequestException as error:
                show_api_error(error)

# ==================================================
# TAB 4: DOCTOR LOGIN + DASHBOARD
# ==================================================
with tab4:
    st.subheader("👨‍⚕️ Doctor Login")
    session = st.session_state.doctor_session

    if session is None:
        st.write("Login using the doctor account created through the local demo API.")
        with st.form("doctor_login_form"):
            username = st.text_input("Doctor Username", placeholder="doctor001")
            password = st.text_input("Password", type="password")
            login_clicked = st.form_submit_button("🔐 Login")

        if login_clicked:
            if not username.strip() or not password:
                st.warning("Username and password enter pannu.")
            else:
                try:
                    login_result = api_post("/doctor/login", {
                        "username": username.strip(), "password": password
                    })
                    st.session_state.doctor_session = {
                        "doctor_id": login_result["doctor_id"],
                        "username": login_result["username"],
                        "access_token": login_result["access_token"]
                    }
                    st.success("Doctor login successful!")
                    st.rerun()
                except requests.exceptions.RequestException as error:
                    show_api_error(error)
    else:
        st.success(f'Logged in as {session["username"]} ({session["doctor_id"]})')
        if st.button("🚪 Logout"):
            st.session_state.doctor_session = None
            st.rerun()

        doctor_id = session["doctor_id"]
        st.caption("Dashboard displays appointments assigned to this doctor account.")
        if st.button("🔄 Refresh Appointments"):
            st.rerun()

        try:
            # Token is sent, but the current backend must verify it to enforce access control.
            appointments = api_get(f"/doctors/{doctor_id}/appointments", token=session["access_token"])
            if not appointments:
                st.info("Indha doctor-ku appointments edhuvum illa.")
            else:
                pending_count = sum(1 for a in appointments if a.get("status") == "Pending")
                col1, col2 = st.columns(2)
                col1.metric("Total Appointments", len(appointments))
                col2.metric("Pending Requests", pending_count)
                st.divider()

                for appointment in appointments:
                    apt_id = appointment["appointment_id"]
                    status = appointment["status"]
                    with st.container(border=True):
                        st.write(f"### Appointment: {apt_id}")
                        st.write("**Patient ID:**", appointment["patient_id"])
                        st.write("**Date:**", appointment["appointment_date"])
                        st.write("**Time:**", appointment["appointment_time"])
                        st.write("**Status:**", get_status_label(status))
                        if status == "Pending":
                            col_accept, col_reject = st.columns(2)
                            with col_accept:
                                if st.button("✅ Accept", key=f"accept_{apt_id}", use_container_width=True):
                                    try:
                                        result = api_put(f"/appointments/{apt_id}/decision", {"status": "Confirmed"}, token=session["access_token"])
                                        st.success("Appointment confirmed!")
                                        st.json(result)
                                        st.rerun()
                                    except requests.exceptions.RequestException as error:
                                        show_api_error(error)
                            with col_reject:
                                if st.button("❌ Reject", key=f"reject_{apt_id}", use_container_width=True):
                                    try:
                                        result = api_put(f"/appointments/{apt_id}/decision", {"status": "Rejected"}, token=session["access_token"])
                                        st.warning("Appointment rejected.")
                                        st.json(result)
                                        st.rerun()
                                    except requests.exceptions.RequestException as error:
                                        show_api_error(error)
                        else:
                            st.caption("Indha appointment-ku decision already saved.")
        except requests.exceptions.RequestException as error:
            show_api_error(error)

st.divider()
st.caption("NALAM Health AI | Educational demo | Not for real patient-care decisions")

# ---------------- VIDEO CALL PREPARATION ----------------

# ---------------- VIDEO CALL TEST ----------------
st.divider()
st.subheader("🎥 Nalam Video Consultation")

st.info(
    "Step 1: Camera and microphone test. "
    "This is not yet a two-person video call."
)

webrtc_streamer(
    key="nalam-camera-test",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },
    media_stream_constraints={
        "video": True,
        "audio": True,
    },
)


# ---------------- NALAM CHAT UI ----------------
import streamlit as st

st.set_page_config(
    page_title="Nalam Health AI",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Nalam Health AI")
st.caption("Your virtual health assistant")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! 👋 Welcome to Nalam Health AI. "
                "Tell me what health issue you are experiencing."
            )
        }
    ]

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_message = st.chat_input(
    "Describe your symptoms..."
)

if user_message:
    st.session_state.chat_messages.append({
        "role": "user",
        "content": user_message
    })

    with st.chat_message("user"):
        st.markdown(user_message)

    response = (
        "Thank you for sharing. I can provide general health "
        "information, but I cannot diagnose your condition. "
        "Would you like to consult a doctor?"
    )

    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": response
    })

    with st.chat_message("assistant"):
        st.markdown(response)


