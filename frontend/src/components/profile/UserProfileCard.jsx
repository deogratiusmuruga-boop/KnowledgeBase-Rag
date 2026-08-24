import React, { useState } from "react";

import { saveUserProfile } from "../../services/ragApi";

export default function UserProfileCard({ onClose, lang }) {

  const [age, setAge] = useState("");
  const [location, setLocation] = useState("");
  const [conditions, setConditions] = useState("");
  const [medications, setMedications] = useState("");
  const text = lang === "en" ? {
    title: "👤 Elderly Profile",
    age: "Age",
    location: "Location",
    conditions: "Chronic Conditions",
    medications: "Medications",
    save: "💾 Save Profile",
    saved: "✅ Profile saved!",
    error: "❌ Cannot connect to the backend."
  } : {
    title: "👤 어르신 프로필",
    age: "나이",
    location: "거주 지역",
    conditions: "만성 질환",
    medications: "복용 중인 약",
    save: "💾 프로필 저장",
    saved: "✅ 프로필이 저장되었습니다!",
    error: "❌ 백엔드에 연결할 수 없습니다."
  };
  const saveProfile = async () => {

  try {

    const data = await saveUserProfile({
      age: age ? Number(age) : null,
      location: location.trim(),
      chronic_conditions: conditions.split(",").map(item => item.trim()).filter(Boolean),
      medications: medications.split(",").map(item => item.trim()).filter(Boolean),
      preferred_language: lang,
      speech_speed: "slow"
    });

    localStorage.setItem("user_id", data.user_id);

    alert(text.saved);

    onClose();

  }

  catch (err) {

    console.error(err);

    alert(text.error);

  }

};

  return (
    

    <div
      style={{
        background: "#f8fafc",
        borderRadius: "16px",
        padding: "20px",
        marginBottom: "20px",
        border: "1px solid #cbd5e1"
      }}
    >

      <div
  style={{
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "16px"
  }}
>
  <h3 style={{ margin: 0 }}>
    {text.title}
  </h3>

  <button
    onClick={onClose}
    style={{
      border: "none",
      background: "transparent",
      cursor: "pointer",
      fontSize: "22px",
      fontWeight: "bold"
    }}
  >
    ✖
  </button>
</div>

      <div
        style={{
          display: "grid",
          gap: "12px"
        }}
      >

        <input
          value={age}
          onChange={(e) => setAge(e.target.value)}
          placeholder={text.age}
          style={{
            padding: "10px",
            borderRadius: "8px",
            border: "1px solid #cbd5e1"
          }}
        />

        <input
          value={location}
          onChange={(e) => setLocation(e.target.value)}
          placeholder={text.location}
          style={{
            padding: "10px",
            borderRadius: "8px",
            border: "1px solid #cbd5e1"
          }}
        />

        <input
          value={conditions}
          onChange={(e) => setConditions(e.target.value)}
          placeholder={text.conditions}
          style={{
            padding: "10px",
            borderRadius: "8px",
            border: "1px solid #cbd5e1"
          }}
        />

        <input
          value={medications}
          onChange={(e) => setMedications(e.target.value)}
          placeholder={text.medications}
          style={{
            padding: "10px",
            borderRadius: "8px",
            border: "1px solid #cbd5e1"
          }}
        />

        <button
  onClick={saveProfile}
  style={{
    marginTop: "12px",
    padding: "12px",
    background: "#2563eb",
    color: "white",
    border: "none",
    borderRadius: "10px",
    cursor: "pointer",
    fontWeight: "bold"
  }}
>
  {text.save}
</button>

      </div>

    </div>

  );

}
