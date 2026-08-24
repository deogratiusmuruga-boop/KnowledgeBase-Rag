const API_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:8000";


export const getReminders = async () => {

  try {

    const userId = localStorage.getItem("user_id") || 2;


    const response = await fetch(
      `${API_URL}/reminders/${userId}`
    );


    if (!response.ok) {

      throw new Error(
        "Failed to fetch reminders"
      );

    }


    const data = await response.json();

    return data;


  } catch (error) {

    console.error(
      "Reminder API error:",
      error
    );

    return [];

  }

};