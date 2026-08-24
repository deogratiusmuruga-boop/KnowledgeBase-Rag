import React, { useEffect, useState } from "react";

import ReminderCard from "../components/reminders/ReminderCard";
import { getReminders } from "../services/reminderApi";

export default function HealthScreen() {
  const [reminders, setReminders] = useState([]);
  const [showMedication, setShowMedication] = useState(false);

  const lang =
    localStorage.getItem("preferred_language") === "en"
      ? "en"
      : "ko";

  const text =
    lang === "en"
      ? {
          title: "Care State",
          subtitle:
            "Your current care situation based on recent health activity",

          currentState: "Current Care State",
          stateLabel: "Stable",
          stateDescription:
            "Your current care information is being monitored.",

          physical: "Physical",
          medication: "Medication",
          activity: "Daily Activity",
          checkin: "Daily Check-in",

          physicalStatus: "No recent changes",
          medicationStatus: "Medication information available",
          activityStatus: "Activity information being monitored",
          checkinStatus: "Check-in data being observed",

          medications: "Medications",
          medicationDescription:
            "Manage your medication information",

          appointments: "Appointments",
          appointmentDescription:
            "Doctor visits and schedules",

          lifestyle: "Lifestyle",
          lifestyleDescription:
            "Water, exercise, and meals",

          monitoring: "Health Monitoring",
          monitoringDescription:
            "Health measurements and preventive care",

          medicationComingSoon:
            "Medication management is being developed.",

          appointmentComingSoon:
            "Appointment management is being developed.",

          lifestyleComingSoon:
            "Lifestyle management is being developed.",

          monitoringComingSoon:
            "Health monitoring is being developed.",
        }
      : {
          title: "돌봄 상태",
          subtitle:
            "최근 건강 활동을 기반으로 현재 돌봄 상태를 확인합니다",

          currentState: "현재 돌봄 상태",
          stateLabel: "안정",
          stateDescription:
            "현재 건강 및 돌봄 정보가 지속적으로 관찰되고 있습니다.",

          physical: "신체 상태",
          medication: "복약 상태",
          activity: "일상 활동",
          checkin: "일일 안부",

          physicalStatus: "최근 큰 변화 없음",
          medicationStatus: "복약 정보 확인 가능",
          activityStatus: "일상 활동 정보 관찰 중",
          checkinStatus: "안부 확인 데이터 관찰 중",

          medications: "복약 관리",
          medicationDescription:
            "복약 정보를 확인하고 관리합니다",

          appointments: "진료 일정",
          appointmentDescription:
            "병원 방문 및 진료 일정을 확인합니다",

          lifestyle: "생활 관리",
          lifestyleDescription:
            "수분, 운동 및 식사를 관리합니다",

          monitoring: "건강 모니터링",
          monitoringDescription:
            "건강 수치 및 예방 관리를 확인합니다",

          medicationComingSoon:
            "복약 관리 기능은 준비 중입니다.",

          appointmentComingSoon:
            "진료 일정 관리 기능은 준비 중입니다.",

          lifestyleComingSoon:
            "생활 관리 기능은 준비 중입니다.",

          monitoringComingSoon:
            "건강 모니터링 기능은 준비 중입니다.",
        };

  useEffect(() => {
    const loadReminders = async () => {
      try {
        const data = await getReminders();
        setReminders(data);
      } catch (error) {
        console.error(
          "Failed to load reminders:",
          error
        );
      }
    };

    loadReminders();
  }, []);

  const medicationReminders = reminders.filter(
    (item) => item.type === "medication"
  );

  return (
    <div
      style={{
        maxWidth: "900px",
        margin: "0 auto",
        padding: "30px 24px 110px",
      }}
    >
      {/* Page Header */}

      <div
        style={{
          marginBottom: "24px",
        }}
      >
        <h1
          style={{
            margin: 0,
            color: "#0f172a",
            fontSize: "2rem",
            fontWeight: "800",
          }}
        >
          ❤️ {text.title}
        </h1>

        <p
          style={{
            marginTop: "8px",
            color: "#64748b",
            fontSize: "1rem",
          }}
        >
          {text.subtitle}
        </p>
      </div>

      {/* Dynamic Care State */}

      <section
        style={{
          background:
            "linear-gradient(135deg, #eff6ff, #ffffff)",
          border: "1px solid #bfdbfe",
          borderRadius: "24px",
          padding: "24px",
          marginBottom: "24px",
          boxShadow:
            "0 10px 30px rgba(37, 99, 235, 0.08)",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            gap: "20px",
            marginBottom: "20px",
          }}
        >
          <div>
            <p
              style={{
                margin: "0 0 6px",
                color: "#64748b",
                fontSize: "0.9rem",
                fontWeight: "700",
                textTransform: "uppercase",
                letterSpacing: "0.05em",
              }}
            >
              {text.currentState}
            </p>

            <h2
              style={{
                margin: 0,
                color: "#1d4ed8",
                fontSize: "2rem",
                fontWeight: "800",
              }}
            >
              {text.stateLabel}
            </h2>

            <p
              style={{
                margin: "8px 0 0",
                color: "#475569",
                lineHeight: "1.6",
              }}
            >
              {text.stateDescription}
            </p>
          </div>

          <div
            style={{
              width: "54px",
              height: "54px",
              borderRadius: "50%",
              background: "#dcfce7",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "1.5rem",
            }}
          >
            ✓
          </div>
        </div>

        {/* State Dimensions */}

        <div
          style={{
            display: "grid",
            gridTemplateColumns:
              "repeat(auto-fit, minmax(180px, 1fr))",
            gap: "12px",
          }}
        >
          <StateItem
            title={text.physical}
            value={text.physicalStatus}
            icon="🩺"
          />

          <StateItem
            title={text.medication}
            value={text.medicationStatus}
            icon="💊"
          />

          <StateItem
            title={text.activity}
            value={text.activityStatus}
            icon="🚶"
          />

          <StateItem
            title={text.checkin}
            value={text.checkinStatus}
            icon="☀️"
          />
        </div>
      </section>

      {/* Care Management */}

      <h2
        style={{
          color: "#0f172a",
          marginBottom: "14px",
        }}
      >
        {lang === "en"
          ? "Care Management"
          : "돌봄 관리"}
      </h2>

      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            "repeat(auto-fit, minmax(220px, 1fr))",
          gap: "16px",
          marginBottom: "28px",
        }}
      >
        {/* Medication */}

        <CareCard
          icon="💊"
          title={text.medications}
          description={text.medicationDescription}
          background="#eff6ff"
          onClick={() =>
            setShowMedication(!showMedication)
          }
        />

        {/* Appointments */}

        <CareCard
          icon="🏥"
          title={text.appointments}
          description={text.appointmentDescription}
          background="#fefce8"
          onClick={() =>
            alert(text.appointmentComingSoon)
          }
        />

        {/* Lifestyle */}

        <CareCard
          icon="💧"
          title={text.lifestyle}
          description={text.lifestyleDescription}
          background="#ecfdf5"
          onClick={() =>
            alert(text.lifestyleComingSoon)
          }
        />

        {/* Monitoring */}

        <CareCard
          icon="🩺"
          title={text.monitoring}
          description={text.monitoringDescription}
          background="#fef2f2"
          onClick={() =>
            alert(text.monitoringComingSoon)
          }
        />
      </div>

      {/* Medication Details */}

      {showMedication && (
        <section
          style={{
            background: "#ffffff",
            border: "1px solid #dbeafe",
            borderRadius: "20px",
            padding: "20px",
            marginBottom: "28px",
          }}
        >
          <h2
            style={{
              marginTop: 0,
              color: "#1e3a8a",
            }}
          >
            💊{" "}
            {lang === "en"
              ? "My Medications"
              : "복용 중인 약"}
          </h2>

          {medicationReminders.length === 0 ? (
            <p
              style={{
                color: "#64748b",
              }}
            >
              {lang === "en"
                ? "No medication reminders available."
                : "등록된 복약 알림이 없습니다."}
            </p>
          ) : (
            medicationReminders.map(
              (medication, index) => (
                <div
                  key={index}
                  style={{
                    padding: "14px 0",
                    borderBottom:
                      index <
                      medicationReminders.length - 1
                        ? "1px solid #e2e8f0"
                        : "none",
                  }}
                >
                  <strong>
                    {medication.title}
                  </strong>

                  <p
                    style={{
                      margin: "6px 0",
                      color: "#475569",
                    }}
                  >
                    {medication.details}
                  </p>

                  <span
                    style={{
                      color: "#64748b",
                      fontWeight: "600",
                    }}
                  >
                    ⏰ {medication.time}
                  </span>
                </div>
              )
            )
          )}
        </section>
      )}

      {/* Existing Reminder Data */}

      <ReminderCard reminders={reminders} />
    </div>
  );
}


/* -------------------------------------------------
   Care State Item
-------------------------------------------------- */

function StateItem({ icon, title, value }) {
  return (
    <div
      style={{
        background: "#ffffff",
        borderRadius: "16px",
        padding: "16px",
        border: "1px solid #dbeafe",
      }}
    >
      <div
        style={{
          fontSize: "1.3rem",
          marginBottom: "8px",
        }}
      >
        {icon}
      </div>

      <div
        style={{
          fontWeight: "700",
          color: "#334155",
          marginBottom: "5px",
        }}
      >
        {title}
      </div>

      <div
        style={{
          fontSize: "0.88rem",
          color: "#64748b",
          lineHeight: "1.4",
        }}
      >
        {value}
      </div>
    </div>
  );
}


/* -------------------------------------------------
   Care Management Card
-------------------------------------------------- */

function CareCard({
  icon,
  title,
  description,
  background,
  onClick,
}) {
  return (
    <div
      onClick={onClick}
      style={{
        padding: "20px",
        borderRadius: "20px",
        background,
        border: "1px solid rgba(148,163,184,0.2)",
        cursor: "pointer",
        transition: "transform 0.2s ease",
      }}
    >
      <div
        style={{
          fontSize: "1.8rem",
          marginBottom: "10px",
        }}
      >
        {icon}
      </div>

      <h3
        style={{
          margin: "0 0 8px",
          color: "#0f172a",
        }}
      >
        {title}
      </h3>

      <p
        style={{
          margin: 0,
          color: "#64748b",
          lineHeight: "1.5",
        }}
      >
        {description}
      </p>
    </div>
  );
}