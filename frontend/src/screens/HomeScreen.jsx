
import React, { useState, useEffect, useRef } from "react";
import AvatarView from "../components/digital_human/AvatarView";
import {
  Mic,
  Search,
  Languages,
  Heart,
  Pill,
  PhoneCall,
  SunMedium,
  Activity,
  ShieldCheck,
  Clock3,
  
} from "lucide-react";

import {
  askCareBuddy,
  getUserMedications,
} from "../services/ragApi";

import UserProfileCard from "../components/profile/UserProfileCard";


export default function HomeScreen() {

  const [showProfile, setShowProfile] = useState(false);

  const [lang, setLang] = useState(() =>
    localStorage.getItem("preferred_language") === "en" ? "en" : "ko"
  );

  const [query, setQuery] = useState("");

  const [response, setResponse] = useState(() =>
    localStorage.getItem("preferred_language") === "en"
      ? "Hello! How has your day been? Feel free to talk to me about anything."
      : "안녕하세요 어르신! 오늘 하루는 어떠셨나요? 식사나 약 챙겨 드셨는지 궁금해요."
  );

  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [voices, setVoices] = useState([]);

  /*
   * Dynamic Care State
   *
   * This is intentionally separated from the static profile.
   * The backend can progressively provide:
   *
   * - current_state
   * - state
   * - care_state
   * - confidence
   * - transition
   *
   * We normalize whichever field the backend returns.
   */
  const [careState, setCareState] = useState({
    state: "stable",
    confidence: null,
    transition: null,
    source: "initial",
  });

  const speechRequestId = useRef(0);


  const t = {

    ko: {

      title: "어르신의 다정한 AI 동반자",
      subtitle: "현재 돌봄 상태를 이해하고 상황에 맞는 도움을 제공합니다",

      placeholder:
        "예: 오늘 몸이 좀 피곤한 것 같아요 / 약을 먹어도 될까요?",

      searchBtn: "말 걸기",
      micBtn: "음성",

      answerHeader: "💬 CareBuddy 답변",

      defaultAnswer:
        "안녕하세요 어르신! 오늘 하루는 어떠셨나요?",

      buttonLabel: "English",

      medsBtn: "약 복용",
      checkinBtn: "오늘 안부",
      sosBtn: "긴급 도움",

      medsMsg:
        "오늘 복용하실 약을 확인해 드릴게요.",

      checkinMsg:
        "오늘 컨디션은 어떠신가요? 불편한 점이 있다면 말씀해 주세요.",

      sosMsg:
        "긴급 도움 요청을 확인했습니다.",

      profileBtn: "프로필",

      speechUnsupported:
        "이 브라우저에서는 음성 인식이 지원되지 않습니다.",

      noMedications:
        "등록된 복용 약이 없습니다.",

      medicationIntro:
        "복용하실 약은 다음과 같습니다:\n\n",

      medicationAt:
        " 시간",

      medicationLoadError:
        "복용 약 정보를 불러올 수 없습니다.",

      careStateTitle:
        "현재 돌봄 상태",

      stable:
        "안정 상태",

      attention:
        "관찰 필요",

      elevated:
        "주의 필요",

      critical:
        "집중적인 돌봄 필요",

      transition:
        "최근 상태 변화",

      confidence:
        "상태 신뢰도",

      initial:
        "아직 충분한 활동 데이터가 없습니다.",

      profileRequired:
        "프로필을 등록하면 더 개인화된 돌봄 상태를 확인할 수 있습니다.",

    },


    en: {

      title: " Senior AI Companion",

      subtitle:
        "Understands the current care state and adapts assistance accordingly",

      placeholder:
        "e.g., I feel tired today / Can I take my medicine?",

      searchBtn: "Talk",
      micBtn: "Voice",

      answerHeader:
        "💬 CareBuddy Response",

      defaultAnswer:
        "Hello! How has your day been?",

      buttonLabel: "한국어",

      medsBtn: "Medication",
      checkinBtn: "Daily Check-in",
      sosBtn: "Emergency",

      medsMsg:
        "Let me check your medication schedule.",

      checkinMsg:
        "How are you feeling today? Please tell me if anything feels uncomfortable.",

      sosMsg:
        "I have registered your request for emergency assistance.",

      profileBtn:
        "Profile",

      speechUnsupported:
        "Speech recognition is not supported in this browser.",

      noMedications:
        "No medications were found.",

      medicationIntro:
        "Your medications are:\n\n",

      medicationAt:
        " at ",

      medicationLoadError:
        "Unable to load medications.",

      careStateTitle:
        "Current Care State",

      stable:
        "Stable",

      attention:
        "Needs Observation",

      elevated:
        "Needs Attention",

      critical:
        "Requires Intensive Care",

      transition:
        "Recent State Change",

      confidence:
        "State Confidence",

      initial:
        "There is not enough activity data yet.",

      profileRequired:
        "Register your profile to enable more personalized care-state analysis.",

    }

  };


  const activeT = t[lang];


  /*
   * Save language preference.
   */
  useEffect(() => {

    localStorage.setItem(
      "preferred_language",
      lang
    );

  }, [lang]);


  /*
   * Load browser voices.
   */
  useEffect(() => {

    if (!("speechSynthesis" in window)) {
      return undefined;
    }

    const loadVoices = () => {

      setVoices(
        window.speechSynthesis.getVoices()
      );

    };

    loadVoices();

    if (window.speechSynthesis.onvoiceschanged !== undefined) {

      window.speechSynthesis.onvoiceschanged =
        loadVoices;

    }

    return () => {

      if (
        window.speechSynthesis.onvoiceschanged ===
        loadVoices
      ) {

        window.speechSynthesis.onvoiceschanged =
          null;

      }

    };

  }, []);


  /*
   * Stop speech.
   */
  const stopSpeech = () => {

    speechRequestId.current += 1;

    if ("speechSynthesis" in window) {

      window.speechSynthesis.cancel();

    }

    setIsSpeaking(false);

  };


  /*
   * Text-to-speech.
   */
  const speakResponse = (
    textToSpeak,
    targetLang = lang
  ) => {

    if (!textToSpeak) return;

    if (!("speechSynthesis" in window)) return;

    const speechSynthesis =
      window.speechSynthesis;

    const requestId =
      speechRequestId.current + 1;

    speechRequestId.current =
      requestId;

    speechSynthesis.cancel();

    const text =
      textToSpeak ||
      t[targetLang].defaultAnswer;

    const utterance =
      new SpeechSynthesisUtterance(text);

    utterance.lang =
      targetLang === "ko"
        ? "ko-KR"
        : "en-US";

    utterance.rate = 0.85;

    const locale =
      targetLang === "ko"
        ? "ko-KR"
        : "en-US";

    const matchedVoice =
      voices.find(
        (v) =>
          v.lang.toLowerCase() ===
          locale.toLowerCase()
      ) ||
      voices.find(
        (v) =>
          v.lang
            .toLowerCase()
            .startsWith(targetLang)
      );

    if (matchedVoice) {

      utterance.voice =
        matchedVoice;

    }

    utterance.onstart = () => {

      if (
        speechRequestId.current ===
        requestId
      ) {

        setIsSpeaking(true);

      }

    };

    utterance.onend = () => {

      if (
        speechRequestId.current ===
        requestId
      ) {

        setIsSpeaking(false);

      }

    };

    utterance.onerror = () => {

      if (
        speechRequestId.current ===
        requestId
      ) {

        setIsSpeaking(false);

      }

    };

    speechSynthesis.speak(
      utterance
    );

  };


  /*
   * Normalize care-state information
   * returned by the backend.
   */
  const updateCareState = (data) => {

    if (!data) return;

    const stateData =
      data.care_state ||
      data.current_care_state ||
      data.current_state ||
      data.adaptive_context ||
      data.state_info;

    if (!stateData) return;


    if (typeof stateData === "string") {

      setCareState({
        state: stateData.toLowerCase(),
        confidence: null,
        transition: null,
        source: "backend",
      });

      return;

    }


    if (typeof stateData === "object") {

      const nextState =
        stateData.state ||
        stateData.current_state ||
        stateData.label ||
        "stable";

      const confidence =
        stateData.confidence ??
        stateData.state_confidence ??
        null;

      const transition =
        stateData.transition ||
        stateData.state_transition ||
        stateData.previous_state ||
        null;

      setCareState({
        state: String(nextState).toLowerCase(),
        confidence,
        transition,
        source: "backend",
      });

    }

  };


  /*
   * Convert backend state into a
   * human-readable label.
   */
  const getCareStateLabel = () => {

    const state =
      careState.state;

    if (
      state.includes("critical") ||
      state.includes("high")
    ) {

      return activeT.critical;

    }

    if (
      state.includes("elevated") ||
      state.includes("warning")
    ) {

      return activeT.elevated;

    }

    if (
      state.includes("attention") ||
      state.includes("observe")
    ) {

      return activeT.attention;

    }

    return activeT.stable;

  };


  /*
   * Visual indicator for care state.
   */
  const getCareStateIndicator = () => {

    const state =
      careState.state;

    if (
      state.includes("critical") ||
      state.includes("high")
    ) {

      return {
        background: "#fef2f2",
        border: "#fecaca",
        text: "#b91c1c",
        dot: "#dc2626"
      };

    }

    if (
      state.includes("elevated") ||
      state.includes("warning")
    ) {

      return {
        background: "#fff7ed",
        border: "#fed7aa",
        text: "#c2410c",
        dot: "#ea580c"
      };

    }

    if (
      state.includes("attention") ||
      state.includes("observe")
    ) {

      return {
        background: "#fefce8",
        border: "#fde68a",
        text: "#a16207",
        dot: "#eab308"
      };

    }

    return {
      background: "#f0fdf4",
      border: "#bbf7d0",
      text: "#15803d",
      dot: "#22c55e"
    };

  };


  /*
   * Language toggle.
   */
  const toggleLanguage = () => {

    const nextLang =
      lang === "ko"
        ? "en"
        : "ko";

    const nextGreeting =
      t[nextLang].defaultAnswer;

    setLang(nextLang);

    setResponse(
      nextGreeting
    );

    speakResponse(
      nextGreeting,
      nextLang
    );

  };


  /*
   * Quick actions.
   */
  const handleAction =
    async (action) => {

      if (
        action ===
        "medication"
      ) {

        try {

          const userId =
            Number(
              localStorage.getItem(
                "user_id"
              )
            );

          if (
            !Number.isInteger(userId) ||
            userId <= 0
          ) {

            throw new Error(
              "Profile required."
            );

          }

          const medications =
            await getUserMedications(
              userId
            );

          if (
            medications.length === 0
          ) {

            setResponse(
              activeT.noMedications
            );

            speakResponse(
              activeT.noMedications,
              lang
            );

            return;

          }

          let reminder =
            activeT.medicationIntro;

          medications.forEach(
            (med) => {

              reminder +=
                `• ${med.medicine_name}`;

              if (med.dosage) {

                reminder +=
                  ` (${med.dosage})`;

              }

              if (med.time) {

                reminder +=
                  `${activeT.medicationAt}${med.time}`;

              }

              reminder += "\n";

            }
          );

          setResponse(
            reminder
          );

          speakResponse(
            reminder,
            lang
          );

        } catch (err) {

          console.error(err);

          setResponse(
            activeT.medicationLoadError
          );

          speakResponse(
            activeT.medicationLoadError,
            lang
          );

        }

        return;

      }


      if (
        action ===
        "checkin"
      ) {

        setResponse(
          activeT.checkinMsg
        );

        speakResponse(
          activeT.checkinMsg,
          lang
        );

        return;

      }


      if (
        action ===
        "sos"
      ) {

        setResponse(
          activeT.sosMsg
        );

        speakResponse(
          activeT.sosMsg,
          lang
        );

      }

    };


  /*
   * Ask CareBuddy.
   */
  const handleSearch =
    async () => {

      const userQuestion =
        query ||
        activeT.defaultAnswer;

      try {

        const savedUserId =
          Number(
            localStorage.getItem(
              "user_id"
            )
          );

        const data =
          await askCareBuddy(
            userQuestion,
            Number.isInteger(
              savedUserId
            ) &&
            savedUserId > 0
              ? savedUserId
              : null,
            null,
            [],
            lang
          );


        /*
         * IMPORTANT:
         *
         * The answer remains exactly
         * compatible with the existing API.
         */
        if (data.answer) {

          setResponse(
            String(
              data.answer
            )
          );

          speakResponse(
            String(
              data.answer
            ),
            lang
          );

        } else {

          setResponse(
            activeT.defaultAnswer
          );

          speakResponse(
            activeT.defaultAnswer,
            lang
          );

        }


        /*
         * Extract dynamic care-state
         * information if supplied.
         */
        updateCareState(data);

      } catch (err) {

        console.error(
          "CareBuddy error:",
          err
        );

        setResponse(
          activeT.defaultAnswer
        );

        speakResponse(
          activeT.defaultAnswer,
          lang
        );

      }

    };


  /*
   * Voice input.
   */
  const handleMic = () => {

    const SpeechRecognition =
      window.SpeechRecognition ||
      window.webkitSpeechRecognition;

    if (!SpeechRecognition) {

      alert(
        activeT.speechUnsupported
      );

      return;

    }

    const rec =
      new SpeechRecognition();

    rec.lang =
      lang === "ko"
        ? "ko-KR"
        : "en-US";

    rec.onstart = () =>
      setIsListening(true);

    rec.onresult = (e) =>
      setQuery(
        e.results[0][0].transcript
      );

    rec.onend = () =>
      setIsListening(false);

    rec.onerror = (error) => {

      console.error(
        "Speech recognition error:",
        error
      );

      setIsListening(false);

    };

    rec.start();

  };


  const stateStyle =
    getCareStateIndicator();


  return (

    <div
      style={{
        maxWidth: "760px",
        margin: "20px auto",
        padding: "24px",
        fontFamily:
          "'Pretendard', sans-serif",
        backgroundColor:
          "#f8fafc",
        borderRadius: "24px",
      }}
    >

      {/* HEADER */}

      <div
        style={{
          display: "flex",
          justifyContent:
            "space-between",
          alignItems: "center",
          marginBottom: "20px",
        }}
      >

        <div>

          <h2
            style={{
              margin: 0,
              color: "#0f172a",
              fontSize: "1.6rem",
              fontWeight: "800",
            }}
          >
            {activeT.title}
          </h2>

          <span
            style={{
              fontSize: "0.95rem",
              color: "#64748b",
            }}
          >
            {activeT.subtitle}
          </span>

        </div>


        <button
          onClick={
            toggleLanguage
          }
          style={{
            display: "flex",
            alignItems: "center",
            gap: "6px",
            padding: "8px 16px",
            backgroundColor:
              "#ffffff",
            border:
              "1px solid #cbd5e1",
            borderRadius: "20px",
            cursor: "pointer",
            fontSize:
              "0.95rem",
            fontWeight: "600",
            color: "#334155",
          }}
        >

          <Languages
            size={18}
          />

          {activeT.buttonLabel}

        </button>

      </div>


      {/* DYNAMIC CARE STATE */}

      <div
        style={{
          marginBottom: "20px",
          padding: "18px",
          borderRadius: "18px",
          background:
            stateStyle.background,
          border:
            `1px solid ${stateStyle.border}`,
        }}
      >

        <div
          style={{
            display: "flex",
            justifyContent:
              "space-between",
            alignItems:
              "flex-start",
            gap: "12px",
          }}
        >

          <div
            style={{
              display: "flex",
              alignItems:
                "center",
              gap: "10px",
            }}
          >

            <div
              style={{
                width: "42px",
                height: "42px",
                borderRadius:
                  "12px",
                background:
                  "#ffffff",
                display: "flex",
                alignItems:
                  "center",
                justifyContent:
                  "center",
              }}
            >

              <Activity
                size={23}
                color={
                  stateStyle.text
                }
              />

            </div>


            <div>

              <div
                style={{
                  fontSize:
                    "0.82rem",
                  color:
                    "#64748b",
                  fontWeight:
                    "600",
                }}
              >
                {activeT.careStateTitle}
              </div>

              <div
                style={{
                  fontSize:
                    "1.15rem",
                  fontWeight:
                    "800",
                  color:
                    stateStyle.text,
                  marginTop:
                    "3px",
                }}
              >

                <span
                  style={{
                    display:
                      "inline-block",
                    width: "9px",
                    height: "9px",
                    borderRadius:
                      "50%",
                    background:
                      stateStyle.dot,
                    marginRight:
                      "7px",
                  }}
                />

                {getCareStateLabel()}

              </div>

            </div>

          </div>


          <ShieldCheck
            size={22}
            color={
              stateStyle.text
            }
          />

        </div>


        {careState.confidence !== null && (

          <div
            style={{
              marginTop: "14px",
              fontSize:
                "0.85rem",
              color:
                "#64748b",
            }}
          >

            {activeT.confidence}:{" "}

            <strong>
              {Math.round(
                Number(
                  careState.confidence
                ) *
                  100
              )}%
            </strong>

          </div>

        )}


        {careState.transition && (

          <div
            style={{
              marginTop: "10px",
              paddingTop: "10px",
              borderTop:
                `1px solid ${stateStyle.border}`,
              display: "flex",
              alignItems:
                "center",
              gap: "7px",
              fontSize:
                "0.85rem",
              color:
                "#64748b",
            }}
          >

            <Clock3
              size={15}
            />

            <span>
              {activeT.transition}:{" "}
              {String(
                careState.transition
              )}
            </span>

          </div>

        )}

      </div>


      {/* DIGITAL COMPANION */}

      <div
        style={{
          marginBottom: "18px",
        }}
      >

        <AvatarView
          isSpeaking={
            isSpeaking
          }
          isListening={
            isListening
          }
          response={
            response
          }
          language={
            lang
          }
          onSpeak={
            speakResponse
          }
          onStop={
            stopSpeech
          }
        />

      </div>


      {/* PROFILE */}

      <div
        style={{
          display: "flex",
          justifyContent:
            "flex-end",
          marginBottom:
            "16px",
        }}
      >

        <button
          onClick={() =>
            setShowProfile(
              true
            )
          }
          style={{
            padding:
              "10px 18px",
            borderRadius:
              "20px",
            border:
              "1px solid #cbd5e1",
            background:
              "#ffffff",
            cursor:
              "pointer",
            fontWeight:
              "600",
            display:
              "flex",
            alignItems:
              "center",
            gap:
              "8px",
          }}
        >

          <Heart
            size={17}
          />

          {activeT.profileBtn}

        </button>

      </div>


      {showProfile && (

        <UserProfileCard
          lang={lang}
          onClose={() =>
            setShowProfile(
              false
            )
          }
        />

      )}


      {/* QUICK CARE ACTIONS */}

      <div
        style={{
          display: "grid",
          gridTemplateColumns:
            "repeat(3, 1fr)",
          gap: "10px",
          marginBottom:
            "20px",
        }}
      >

        <button
          onClick={() =>
            handleAction(
              "medication"
            )
          }
          style={{
            padding:
              "14px 8px",
            borderRadius:
              "14px",
            border:
              "1px solid #bfdbfe",
            background:
              "#eff6ff",
            color:
              "#1d4ed8",
            fontWeight:
              "700",
            cursor:
              "pointer",
            display:
              "flex",
            flexDirection:
              "column",
            alignItems:
              "center",
            gap:
              "6px",
          }}
        >

          <Pill
            size={22}
          />

          {activeT.medsBtn}

        </button>


        <button
          onClick={() =>
            handleAction(
              "checkin"
            )
          }
          style={{
            padding:
              "14px 8px",
            borderRadius:
              "14px",
            border:
              "1px solid #fde68a",
            background:
              "#fefce8",
            color:
              "#a16207",
            fontWeight:
              "700",
            cursor:
              "pointer",
            display:
              "flex",
            flexDirection:
              "column",
            alignItems:
              "center",
            gap:
              "6px",
          }}
        >

          <SunMedium
            size={22}
          />

          {activeT.checkinBtn}

        </button>


        <button
          onClick={() =>
            handleAction(
              "sos"
            )
          }
          style={{
            padding:
              "14px 8px",
            borderRadius:
              "14px",
            border:
              "1px solid #fecaca",
            background:
              "#fef2f2",
            color:
              "#dc2626",
            fontWeight:
              "700",
            cursor:
              "pointer",
            display:
              "flex",
            flexDirection:
              "column",
            alignItems:
              "center",
            gap:
              "6px",
          }}
        >

          <PhoneCall
            size={22}
          />

          {activeT.sosBtn}

        </button>

      </div>


      {/* CAREBUDDY INPUT */}

      <div
        style={{
          display: "flex",
          gap: "10px",
          marginBottom:
            "20px",
        }}
      >

        <input
          type="text"
          value={query}
          onChange={(e) =>
            setQuery(
              e.target.value
            )
          }
          onKeyDown={(e) =>
            e.key === "Enter" &&
            handleSearch()
          }
          placeholder={
            activeT.placeholder
          }
          style={{
            flex: 1,
            padding:
              "16px 20px",
            fontSize:
              "1.05rem",
            borderRadius:
              "14px",
            border:
              "2px solid #e2e8f0",
            outline:
              "none",
            background:
              "#ffffff",
          }}
        />


        <button
          onClick={
            handleSearch
          }
          style={{
            padding:
              "16px 20px",
            background:
              "#2563eb",
            color:
              "white",
            border:
              "none",
            borderRadius:
              "14px",
            cursor:
              "pointer",
            fontWeight:
              "bold",
            display:
              "flex",
            alignItems:
              "center",
            gap:
              "6px",
          }}
        >

          <Search
            size={20}
          />

          {activeT.searchBtn}

        </button>


        <button
          onClick={
            handleMic
          }
          style={{
            padding:
              "16px 20px",
            background:
              "#10b981",
            color:
              "white",
            border:
              "none",
            borderRadius:
              "14px",
            cursor:
              "pointer",
            fontWeight:
              "bold",
            display:
              "flex",
            alignItems:
              "center",
            gap:
              "6px",
          }}
        >

          <Mic
            size={20}
          />

          {activeT.micBtn}

        </button>

      </div>


      {/* RESPONSE */}

      {response && (

        <div
          style={{
            padding:
              "24px",
            background:
              "#ffffff",
            borderRadius:
              "18px",
            borderLeft:
              "5px solid #2563eb",
            boxShadow:
              "0 4px 12px rgba(0,0,0,0.04)",
          }}
        >

          <h4
            style={{
              margin:
                "0 0 10px 0",
              color:
                "#1e293b",
              fontSize:
                "1.15rem",
              display:
                "flex",
              alignItems:
                "center",
              gap:
                "8px",
            }}
          >

            <Heart
              color="#e11d48"
              size={20}
            />

            {activeT.answerHeader}

          </h4>


          <p
            style={{
              margin: 0,
              fontSize:
                "1.15rem",
              lineHeight:
                "1.7",
              color:
                "#334155",
              whiteSpace:
                "pre-line",
            }}
          >

            {response}

          </p>

        </div>

      )}

    </div>

  );

}
