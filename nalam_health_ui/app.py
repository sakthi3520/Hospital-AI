import streamlit as st
import requests
from datetime import date
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="Nalam Health AI", page_icon="🏥", layout="wide")
API_URL = "http://127.0.0.1:8000"
UPLOAD_ENDPOINT = f"{API_URL}/upload-medical-file"

# ---------------- API HELPERS ----------------
def api_get(endpoint, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.get(f"{API_URL}{endpoint}", headers=headers, timeout=20)
    response.raise_for_status()
    return response.json()


def api_post(endpoint, payload, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.post(f"{API_URL}{endpoint}", json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def api_put(endpoint, payload, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response = requests.put(f"{API_URL}{endpoint}", json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def upload_medical_file(uploaded_file):
    """Upload a selected file to the FastAPI demo endpoint."""
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")}
    response = requests.post(UPLOAD_ENDPOINT, files=files, timeout=60)
    response.raise_for_status()
    return response.json()


def show_api_error(error):
    if isinstance(error, requests.exceptions.HTTPError):
        response = error.response
        st.error(f"API Error: {response.status_code if response is not None else 'HTTP'}")
        if response is not None:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            st.code(str(detail))
    elif isinstance(error, requests.exceptions.ConnectionError):
        st.error("FastAPI connect aagala. Backend server running-aa check pannu.")
    elif isinstance(error, requests.exceptions.Timeout):
        st.error("Request time out aagiduchu. FastAPI server-ai check pannunga.")
    else:
        st.error(f"Error: {error}")


def get_doctors():
    data = api_get("/doctors")
    return data if isinstance(data, list) else []


def get_status_label(status):
    return {"Confirmed": "✅ Confirmed", "Rejected": "❌ Rejected", "Pending": "⏳ Pending"}.get(status, status)


def get_jitsi_meeting_link(appointment_id):
    safe_id = "".join(char for char in str(appointment_id).upper() if char.isalnum() or char == "-")
    return f"https://meet.jit.si/NalamHealth-{safe_id}"


# ---------------- SESSION STATE ----------------
if "doctor_session" not in st.session_state:
    st.session_state.doctor_session = None

# ---------------- APP HEADER ----------------
st.title("🏥 NALAM Health AI")
st.caption("Smart Hospital Appointment Management Demo")
st.info(
    "Educational demo only. This app does not provide a medical diagnosis. "
    "Do not upload real patient data. Uploaded files are stored locally by the demo API "
    "and are not yet securely associated with a patient or doctor."
)

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

        st.subheader("📷 Optional Photo / Video")
        uploaded_photo = st.file_uploader(
            "Upload skin rash, wound or other relevant photo",
            type=["jpg", "jpeg", "png", "webp"], key="patient_photo"
        )
        uploaded_video = st.file_uploader(
            "Upload a short video (optional)",
            type=["mp4", "mov", "webm"], key="patient_video"
        )
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
                patient_id = result.get("patient_id")
                st.write("**Patient ID:**", patient_id)
                st.info("Patient ID-a save pannikko.")

                # Upload selected files after registration succeeds.
                # The current API does not save a patient ID/file relationship.
                for label, selected_file in [("Photo", uploaded_photo), ("Video", uploaded_video)]:
                    if selected_file is not None:
                        try:
                            upload_result = upload_medical_file(selected_file)
                            st.success(f"{label} upload successful!")
                            st.caption("Saved filename: " + str(upload_result.get("saved_filename", "Not returned")))
                            st.warning("Demo limitation: this file is not yet linked to the Patient ID or shown in the doctor dashboard.")
                        except requests.exceptions.RequestException as error:
                            st.error(f"{label} upload failed.")
                            show_api_error(error)
            except requests.exceptions.RequestException as error:
                show_api_error(error)

# ==================================================
# TAB 2: APPOINTMENT BOOKING
# ==================================================
# with tab2:
#     st.subheader("📅 Book an Appointment")
# patient_id = st.text_input(
#     "Patient ID",
#     placeholder="Example: PAT-D21BC4C9",
#     key="booking_patient_id"
# )

# try:
#     doctors = get_doctors()
#     active_doctors = [
#         d for d in doctors if d.get("is_active", False)
#     ]

#     if not active_doctors:
#         st.warning("Active doctors list empty-aa irukku.")
#     else:
#         doctor_options = {
#             f'{d["doctor_name"]} — {d["specialty"]} ({d["doctor_id"]})':
#             d["doctor_id"]
#             for d in active_doctors
#         }

#         selected_doctor = st.selectbox(
#             "Choose Doctor",
#             list(doctor_options.keys())
#         )
#         doctor_id = doctor_options[selected_doctor]

#         appointment_date = st.date_input(
#             "Appointment Date",
#             min_value=date.today()
#         )

#         if st.button("🔍 Check Available Slots"):
#             try:
#                 slots = api_get(
#                     f"/doctors/{doctor_id}/free-slots"
#                 )
#                 st.session_state["available_slots"] = slots
#                 st.session_state["slots_doctor_id"] = doctor_id
#                 st.session_state["slots_date"] = appointment_date.isoformat()
#             except requests.exceptions.RequestException as error:
#                 show_api_error(error)

#         slots = st.session_state.get("available_slots", [])

#         if (
#             st.session_state.get("slots_doctor_id") == doctor_id
#             and st.session_state.get("slots_date") == appointment_date.isoformat()
#         ):
#             if slots:
                
#                     slot_options = [
#                         s.get("time", s.get("start_time", ""))
#                         for s in slots
#                         if s.get("time", s.get("start_time", ""))
#                     ]


#             if slot_options:
#                     selected_time = st.selectbox(
#                         "Available Appointment Time",
#                         slot_options
#                     )

#                     if st.button("📅 Book Appointment"):
#                         if not patient_id.strip():
#                             st.warning("Patient ID enter pannu.")
#                         else:
#                             result = api_post("/appointments", {
#                                 "patient_id": patient_id.strip(),
#                                 "doctor_id": doctor_id,
#                                 "appointment_date": appointment_date.isoformat(),
#                                 "appointment_time": selected_time
#                             })

#                             st.success("Appointment request submitted!")
#                             st.write("Appointment ID:", result["appointment_id"])
#                             st.warning(
#                                 "Status: Pending. Doctor approval varum varai wait pannu."
#                             )
#                 else:
#                     st.info("API response-la slot time field kidaikkala.")
#             else:
#                 st.info("Indha date-ku free slots kidaikkala.")
#         else:
#             st.caption("Date and doctor select pannitu Check Available Slots click pannu.")

# except requests.exceptions.RequestException as error:
#     show_api_error(error)
# except (KeyError, TypeError, ValueError) as error:
#     st.error(f"Doctor data error: {error}")


# TAB 2: APPOINTMENT BOOKING

# ==================================================

# with tab2:
# st.subheader("📅 Book an Appointment")

# patient_id = st.text_input(
#     "Patient ID",
#     placeholder="Example: PAT-D21BC4C9",
#     key="booking_patient_id",
# )

with tab2:
    st.subheader("📅 Book an Appointment")
    patient_id = st.text_input(
        "Patient ID",
        placeholder="Example: PAT-D21BC4C9",
        key="booking_patient_id",
    )

try:
    doctors = get_doctors()
    active_doctors = [
        d for d in doctors if d.get("is_active", False)
    ]

    if not active_doctors:
        st.warning("Active doctors list empty-aa irukku.")
    else:
        doctor_options = {
            f'{d["doctor_name"]} — {d["specialty"]} ({d["doctor_id"]})':
            d["doctor_id"]
            for d in active_doctors
        }

        selected_doctor = st.selectbox(
            "Choose Doctor",
            list(doctor_options.keys()),
        )
        doctor_id = doctor_options[selected_doctor]

        appointment_date = st.date_input(
            "Appointment Date",
            min_value=date.today(),
        )

        if st.button("🔍 Check Available Slots"):
            try:
                slots = api_get(
                    f"/doctors/{doctor_id}/free-slots"
                )

                st.session_state["available_slots"] = (
                    slots if isinstance(slots, list) else []
                )
                st.session_state["slots_doctor_id"] = doctor_id
                st.session_state["slots_date"] = (
                    appointment_date.isoformat()
                )

            except requests.exceptions.RequestException as error:
                show_api_error(error)

        slots = st.session_state.get("available_slots", [])

        same_selection = (
            st.session_state.get("slots_doctor_id") == doctor_id
            and st.session_state.get("slots_date")
            == appointment_date.isoformat()
        )

        if same_selection:
            if slots:
                slot_options = [
                    s.get("time", s.get("start_time", ""))
                    for s in slots
                    if s.get("time", s.get("start_time", ""))
                ]

                if slot_options:
                    selected_time = st.selectbox(
                        "Available Appointment Time",
                        slot_options,
                    )

                    if st.button("📅 Book Appointment"):
                        if not patient_id.strip():
                            st.warning("Patient ID enter pannu.")
                        else:
                            try:
                                result = api_post(
                                    "/appointments",
                                    {
                                        "patient_id": patient_id.strip(),
                                        "doctor_id": doctor_id,
                                        "appointment_date":
                                            appointment_date.isoformat(),
                                        "appointment_time": selected_time,
                                    },
                                )

                                st.success(
                                    "Appointment request submitted!"
                                )
                                st.write(
                                    "Appointment ID:",
                                    result.get(
                                        "appointment_id",
                                        "Not returned",
                                    ),
                                )
                                st.warning(
                                    "Status: Pending. Doctor approval "
                                    "varum varai wait pannu."
                                )

                                # Clear old slot results after booking.
                                st.session_state["available_slots"] = []
                                st.session_state["slots_doctor_id"] = None
                                st.session_state["slots_date"] = None

                            except requests.exceptions.RequestException as error:
                                show_api_error(error)
                else:
                    st.info(
                        "API response-la slot time field kidaikkala."
                    )
            else:
                st.info("Indha date-ku free slots kidaikkala.")
        else:
            st.caption(
                "Date and doctor select pannitu "
                "Check Available Slots click pannu."
            )

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
                    st.info("Demo link only; this is not a private, authenticated medical video room.")
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
                    login_result = api_post("/doctor/login", {"username": username.strip(), "password": password})
                    st.session_state.doctor_session = {
                        "doctor_id": login_result["doctor_id"], "username": login_result["username"],
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
            # The current backend must verify this token on protected endpoints.
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

# ==================================================
# VIDEO CALL CAMERA / MICROPHONE TEST
# ==================================================
st.divider()
st.subheader("🎥 Nalam Video Consultation")
st.info("Step 1: Camera and microphone test. This is not yet a two-person video call.")
webrtc_streamer(
    key="nalam-camera-test",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": True},
)
st.divider()
st.caption("NALAM Health AI | Educational demo | Not for real patient-care decisions")
