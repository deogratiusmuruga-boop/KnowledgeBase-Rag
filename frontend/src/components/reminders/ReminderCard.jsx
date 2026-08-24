import React from "react";
import { Pill, CalendarDays, Droplets, Dumbbell, Utensils } from "lucide-react";


import "./ReminderCard.css";

export default function ReminderCard({ reminders }) {


  const getIcon = (type) => {

    switch(type) {

      case "medication":
        return <Pill size={28} color="#2563eb" />;

      case "appointment":
        return <CalendarDays size={28} color="#16a34a" />;

      case "water":
        return <Droplets size={28} color="#0891b2" />;

      case "exercise":
        return <Dumbbell size={28} color="#ea580c" />;

      case "meal":
        return <Utensils size={28} color="#ca8a04" />;

      default:
        return <Pill size={28} />;

    }

  };


  return (

    <div className="reminder-section">

      <h2>
        🌸 Today's Health
      </h2>


      {
        reminders.map((item, index)=>(

          <div
            key={index}
            className="reminder-card"
          >

            <div className="reminder-icon">
              {getIcon(item.type)}
            </div>


            <div className="reminder-content">

              <h3>
                {item.title}
              </h3>


              <p>
                {item.details}
              </p>


              <span>
                ⏰ {item.time}
              </span>

            </div>


          </div>

        ))
      }


    </div>

  );

}
