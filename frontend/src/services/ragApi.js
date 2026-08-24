
// CareBuddy API Service

const BASE_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";


// Ask CareBuddy
export async function askCareBuddy(
  question,
  userId = null,
  userProfile = null,
  conversationHistory = [],
  preferredLanguage = "ko"
) {
  const response = await fetch(`${BASE_URL}/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      user_id: userId,
      user_profile: userProfile,
      conversation_history: conversationHistory,
      preferred_language: preferredLanguage,
      language: preferredLanguage,
    }),
  });

  if (!response.ok) {
    throw new Error(`API Error: ${response.status}`);
  }

  return await response.json();
}



// Save User Profile
export async function saveUserProfile(profile) {

  const response = await fetch(`${BASE_URL}/profile/`, {

    method: "POST",

    headers: {
      "Content-Type": "application/json",
    },

    body: JSON.stringify(profile),

  });

  if (!response.ok) {
    throw new Error("Failed to save profile");
  }

  return await response.json();

}



// Save Medication
export async function saveMedication(medication) {

  const response = await fetch(`${BASE_URL}/medications/`, {

    method: "POST",

    headers: {
      "Content-Type": "application/json",
    },

    body: JSON.stringify(medication),

  });

  if (!response.ok) {
    throw new Error("Failed to save medication");
  }

  return await response.json();

}



// Get User Medications
export async function getUserMedications(userId) {

  const response = await fetch(
    `${BASE_URL}/medications/${userId}`
  );

  if (!response.ok) {
    throw new Error("Failed to load medications");
  }

  return await response.json();

}
