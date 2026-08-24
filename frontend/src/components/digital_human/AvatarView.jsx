import React, { useState, useEffect } from 'react';
import { Volume2 } from 'lucide-react';

export default function AvatarView({
  isSpeaking,
  isListening,
  response,
  language,
  onSpeak,
  onStop
}) {
  const [mouthOpen, setMouthOpen] = useState(false);
  const [blinking, setBlinking] = useState(false);
  const text = language === 'en' ? {
    speaking: 'AI companion is speaking...',
    listening: 'Listening to you...',
    ready: 'AI Digital Human is ready',
    name: 'Minji, AI Care Companion',
    description: 'Real-time speech and lip-sync conversation',
    stop: 'Stop speech',
    replay: 'Play response again'
  } : {
    speaking: 'AI 상담원 음성 안내 중...',
    listening: '음성을 듣고 있어요...',
    ready: 'AI Digital Human 준비 완료',
    name: '김민지 AI 전문 상담원',
    description: '실시간 음성 합성 & 립싱크 대화 시스템',
    stop: '음성 정지',
    replay: '음성 다시 듣기'
  };


useEffect(() => {

  let interval;

  if (isSpeaking) {

    interval = setInterval(() => {
      setMouthOpen(prev => !prev);
    }, 180);

  } else {

    setMouthOpen(false);

  }


  return () => clearInterval(interval);


}, [isSpeaking]);

useEffect(() => {

  const blinkTimer = setInterval(() => {

    setBlinking(true);

    setTimeout(() => {
      setBlinking(false);
    }, 150);

  }, 3500);


  return () => clearInterval(blinkTimer);

}, []);

  console.log("AvatarView loaded");

  return (
    <div style={{
      background: 'linear-gradient(145deg, #1e293b, #0f172a)',
      borderRadius: '24px',
      padding: '24px',
      color: '#FFF',
      boxShadow: '0 20px 40px rgba(0,0,0,0.3)',
      maxWidth: '420px',
      width: '100%',
      margin: '0 auto',
      border: '1px solid #334155'
    }}>

      {/* Avatar Display Frame */}
      <div style={{
        position: 'relative',
        width: '100%',
        height: '340px',
        borderRadius: '20px',
        overflow: 'hidden',
        background: 'radial-gradient(circle, #3b82f6 0%, #1e1b4b 100%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',

        // SAFE SPEAKING EFFECT
        border: isSpeaking 
          ? '3px solid #60a5fa' 
          : '2px solid #475569',

        boxShadow: isSpeaking
          ? '0 0 35px rgba(96,165,250,0.5)'
          : 'none',

        transition: 'all 0.3s ease'
      }}>


        {/* ORIGINAL SVG AVATAR */}
        <div style={{
  transform: isSpeaking 
    ? 'scale(1.03) rotate(1deg)' 
    : 'scale(1) rotate(0deg)',
  transition: 'transform 0.4s ease',
  textAlign: 'center'
}}>

          <svg 
            width="200" 
            height="240" 
            viewBox="0 0 200 240"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >

            {/* Face */}
            <circle 
              cx="100" 
              cy="80" 
              r="50" 
              fill="#FDBA74"
            />


            {/* Hair */}
            <path 
              d="M50 80C50 52.3858 72.3858 30 100 30C127.614 30 150 52.3858 150 80C150 85 148 95 145 100C135 75 115 65 100 65C85 65 65 75 55 100C52 95 50 85 50 80Z"
              fill="#1E293B"
            />


            {/* Animated Eyes */}

{blinking ? (

  <>
    <path
      d="M75 80 Q82 75 89 80"
      stroke="#1E293B"
      strokeWidth="4"
      fill="none"
    />

    <path
      d="M111 80 Q118 75 125 80"
      stroke="#1E293B"
      strokeWidth="4"
      fill="none"
    />
  </>

) : (

  <>
    <circle cx="82" cy="80" r="6" fill="#1E293B"/>
    <circle cx="118" cy="80" r="6" fill="#1E293B"/>

    <circle cx="84" cy="78" r="2" fill="#FFFFFF"/>
    <circle cx="120" cy="78" r="2" fill="#FFFFFF"/>
  </>

)}

           {/* Natural Speaking Mouth */}

{mouthOpen ? (

  <ellipse
    cx="100"
    cy="108"
    rx="16"
    ry="12"
    fill="#7f1d1d"
  />

) : (

  <path
    d="M75 105C85 115 115 115 125 105"
    stroke="#E11D48"
    strokeWidth="4"
    strokeLinecap="round"
    fill="none"
  />

)}



            {/* Body */}
            <path
              d="M30 220C30 170 60 140 100 140C140 140 170 170 170 220V240H30V220Z"
              fill="#2563EB"
            />


            {/* Collar */}
            <path
              d="M85 140L100 175L115 140"
              stroke="#FFFFFF"
              strokeWidth="3"
            />

          </svg>

        </div>



        {/* Status Badge */}
        <div style={{
          position:'absolute',
          top:'16px',
          left:'16px',
          backgroundColor:'rgba(15,23,42,0.85)',
          padding:'6px 14px',
          borderRadius:'20px',
          fontSize:'0.85rem',
          fontWeight:'600',
          display:'flex',
          alignItems:'center',
          gap:'8px'
        }}>


          <span style={{
            width:'10px',
            height:'10px',
            borderRadius:'50%',
            backgroundColor: isSpeaking
              ? '#38bdf8'
              : isListening
                ? '#f59e0b'
                : '#10b981',

            boxShadow: isSpeaking
              ? '0 0 10px #38bdf8'
              : 'none'
          }}>
          </span>


          {isSpeaking ? text.speaking : isListening ? text.listening : text.ready}


        </div>


      </div>



      {/* Controls */}
      <div style={{
        marginTop:'20px',
        textAlign:'center'
      }}>


        <h3 style={{
          fontSize:'1.4rem',
          fontWeight:'bold',
          margin:'0 0 4px 0',
          color:'#f8fafc'
        }}>
          {text.name}
        </h3>


        <p style={{
          color:'#94a3b8',
          fontSize:'0.95rem'
        }}>
          {text.description}
        </p>



        <button
          onClick={isSpeaking ? onStop : () => onSpeak(response, language)}
          style={{
            width:'100%',
            padding:'14px',
            borderRadius:'12px',
            border:'none',
            backgroundColor: isSpeaking 
              ? '#ef4444'
              : '#2563eb',

            color:'white',
            fontSize:'1.1rem',
            fontWeight:'bold',
            cursor:'pointer',

            display:'flex',
            alignItems:'center',
            justifyContent:'center',
            gap:'8px'
          }}
        >

          <Volume2 size={20}/>

          <span>
            {isSpeaking ? text.stop : text.replay}
          </span>


        </button>


      </div>


    </div>
  );
}
