import unittest
from types import SimpleNamespace

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.appointment import (
    AppointmentCreate,
    create_appointment,
    delete_appointment,
)
from api.medication import MedicationCreate, create_medication
from api.reminder import get_reminders
from database import Base
from models.user import UserProfile
from scripts.build_grounded_prompt import build_grounded_prompt
from api.main import normalize_response_language


class CareBuddyApiTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.db = sessionmaker(bind=self.engine)()
        user = UserProfile(age=70)
        self.db.add(user)
        self.db.commit()
        self.user_id = user.id

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def test_reminders_include_medications_and_appointments(self):
        create_medication(
            MedicationCreate(user_id=self.user_id, medicine_name="Example"), self.db
        )
        create_appointment(
            AppointmentCreate(
                user_id=self.user_id,
                title="Checkup",
                appointment_date="2026-08-10",
                appointment_time="09:00",
            ),
            self.db,
        )

        self.assertEqual(len(get_reminders(self.user_id, self.db)), 2)

    def test_mutations_reject_unknown_user(self):
        with self.assertRaises(HTTPException) as error:
            create_medication(
                MedicationCreate(user_id=999, medicine_name="Example"), self.db
            )
        self.assertEqual(error.exception.status_code, 404)

    def test_delete_missing_appointment_returns_not_found(self):
        with self.assertRaises(HTTPException) as error:
            delete_appointment(999, self.db)
        self.assertEqual(error.exception.status_code, 404)

    def test_grounded_prompt_separates_diabetes_profile_from_evidence(self):
        prompt = build_grounded_prompt(
            query="What are safe ways to take medicines?",
            evidence_items=[
                {
                    "source_document": "tips-take-medicines-safely.pdf",
                    "text": "Take medicines exactly as directed.",
                }
            ],
            reliability={
                "authority": 1.0,
                "relevance": 1.0,
                "support": 1.0,
                "coverage": 1.0,
                "consistency": 1.0,
            },
            decision={"decision": "ACCEPT"},
            user_profile=SimpleNamespace(
                age=72,
                location="Seoul",
                chronic_conditions=["diabetes"],
                medications=["metformin"],
                preferred_language="en",
                speech_speed="slow",
            ),
        )

        self.assertIn("USER PROFILE CONTEXT", prompt)
        self.assertIn("- Chronic conditions: diabetes", prompt)
        self.assertIn("- Medications: metformin", prompt)
        self.assertIn("What are safe ways to take medicines?", prompt)
        self.assertIn(
            "It is not evidence and must not be used as a source for medical claims.",
            prompt,
        )

    def test_korean_profile_adds_korean_answer_instruction(self):
        prompt = self._build_prompt(response_language="ko")
        self.assertIn("Generate the answer body naturally in Korean.", prompt)
        self.assertIn("Use polite Korean suitable for elderly users.", prompt)
        self.assertIn("Sources:\n- document_name.pdf", prompt)

    def test_english_profile_adds_english_answer_instruction(self):
        prompt = self._build_prompt(response_language="en")
        self.assertIn("Generate the answer body naturally in English.", prompt)
        self.assertIn("Use clear and simple language suitable for elderly users.", prompt)
        self.assertIn("Sources:\n- document_name.pdf", prompt)

    def test_invalid_or_missing_language_defaults_to_english(self):
        self.assertEqual(normalize_response_language("KO"), "ko")
        self.assertEqual(normalize_response_language("ja"), "en")
        self.assertEqual(normalize_response_language(None), "en")

    @staticmethod
    def _build_prompt(response_language):
        return build_grounded_prompt(
            query="What are safe ways to take medicines?",
            evidence_items=[
                {
                    "source_document": "tips-take-medicines-safely.pdf",
                    "text": "Take medicines exactly as directed.",
                }
            ],
            reliability={
                "authority": 1.0,
                "relevance": 1.0,
                "support": 1.0,
                "coverage": 1.0,
                "consistency": 1.0,
            },
            decision={"decision": "ACCEPT"},
            response_language=response_language,
        )


if __name__ == "__main__":
    unittest.main()
