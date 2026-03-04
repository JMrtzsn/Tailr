"""CV-to-job fit analysis — model, prompt, and chain invocation."""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

from tailr.providers import Provider, list_gemini_models, list_openai_models

_FIT_SYSTEM_PROMPT = """\
You are a career-fit analyst.

Your job is to read a candidate's CV and a job description, then produce an honest,
evidence-based assessment of role readiness and the candidate's *fit strength*.
Optimise for clarity, credibility, and actionable interview prep.

IMPORTANT — this is NOT an ATS keyword checker. Do NOT penalise the candidate
for missing technologies they have never claimed to know. Instead:
- Proof points highlight what the candidate clearly brings to the table.
- Risk flags highlight credibility concerns *within* the CV itself:
  vague claims, unclear scope/ownership, inflated language, missing metrics,
  time-gaps, or areas where the CV is ambiguous about depth vs. exposure.
  They should NEVER be a wishlist of job-description keywords absent from the CV.

CORE PRINCIPLES:
- Be specific and grounded in what the CV actually says.
- Prefer evidence of impact and ownership over buzzwords.
- Don't infer facts that aren't stated.
- Keep every bullet crisp, plain-language, and useful to a human reader.
- Do NOT quote the CV back verbatim. Summarise the signal in your own words.

ROLE READINESS SCORE (0–100): "career signal score"
Weight the score using this rubric:
- 40% Role primitives: core skills/tools + job fundamentals.
- 30% Impact evidence: outcomes, scope, complexity, ownership.
- 20% Execution maturity: collaboration, leadership, reliability, decision making.
- 10% Narrative quality: clarity, focus, progression, credibility.

VERDICT CRITERIA (keep these exact labels):
- STRONG FIT: 65%+ and no material blockers; strong evidence of readiness.
- POSSIBLE FIT: 40–64% with some gaps/unclear signals but credible upside.
- WEAK FIT: <40% OR missing a critical must-have that the job clearly requires.

For STRONG FIT and POSSIBLE FIT, include practical interview focus areas.
"""

_FIT_USER_TEMPLATE = """\
Analyse this CV against the job description and return a structured result.

JOB DESCRIPTION:
{job_description}

CANDIDATE'S CV:
{cv}

Output requirements — follow these carefully:

- strengths (4–10 bullets): What the candidate clearly brings. State each signal
  in plain language — do NOT paste CV quotes or use "Evidence: …" formatting.
  Good: "8+ years building distributed backend systems, most recently at IKEA."
  Bad:  "Experience — Evidence: 'Senior Backend Engineer with 8+ years…'"

- gaps (3–8 bullets): Credibility risks or weak spots *in the CV itself*.
  Examples of good risk flags:
    • "Observability experience is mentioned but scope/scale is unclear."
    • "Claims 'led' a migration but no detail on team size or decision ownership."
    • "No measurable outcomes listed for the most recent role."
  Do NOT list job-description requirements the CV doesn't mention — that is the
  old ATS approach and we explicitly do not want it.

- recommendation_reason: 3–6 sentences explaining the verdict.

- knowledge_gains: Skills or technologies the candidate would develop in this role.

- interview_focus_areas (only if STRONG FIT or POSSIBLE FIT, 4–8 bullets):
  Each bullet: what to probe + what the candidate should be ready to demonstrate.

- gap_coverage (only if STRONG FIT or POSSIBLE FIT, 3–8 bullets):
  Honest ways to contextualise risk flags without bluffing; concrete next steps.

Also extract job_title and company_name (or 'Unknown').
"""

_FIT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _FIT_SYSTEM_PROMPT),
        ("human", _FIT_USER_TEMPLATE),
    ]
)


class FitAnalysis(BaseModel):
    """Fit analysis result for a CV matched against a job description."""

    job_title: str = Field(description="Extracted job title from job description")
    company_name: str = Field(description="Extracted company name or 'Unknown'")
    score: int = Field(
        description="Career signal score 0–100 based on role readiness",
        ge=0,
        le=100,
    )
    recommendation: Literal["STRONG FIT", "POSSIBLE FIT", "WEAK FIT"] = Field(
        description="Verdict based on score, requirements coverage, and growth potential",
    )
    recommendation_reason: str = Field(
        description=(
            "Explanation of the verdict, referencing specific evidence"
            " from the CV and job description"
        ),
    )
    strengths: list[str] = Field(
        default_factory=list,
        description="What the candidate clearly brings — plain-language signals, no CV quotes",
    )
    gaps: list[str] = Field(
        default_factory=list,
        description=(
            "Credibility risks in the CV: vague claims, unclear scope, missing metrics, "
            "ambiguous depth. NOT a list of missing job-description keywords."
        ),
    )
    knowledge_gains: list[str] = Field(
        default_factory=list,
        description="Skills or technologies the candidate would develop in this role",
    )
    interview_focus_areas: list[str] = Field(
        default_factory=list,
        description=(
            "Topics worth probing in an interview — useful to both recruiter (screening questions) "
            "and applicant (preparation). Only for STRONG FIT or POSSIBLE FIT."
        ),
    )
    gap_coverage: list[str] = Field(
        default_factory=list,
        description=(
            "How identified risk flags might be contextualised honestly"
            " — for STRONG FIT or POSSIBLE FIT only"
        ),
    )


class ModelNotFoundError(Exception):
    """Raised when the requested model does not exist on the provider."""

    def __init__(
        self,
        model: str,
        provider: Provider,
        available: list[str] | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self.available = available or []
        super().__init__(f"Model '{model}' not found on {provider.value}")


class FitAnalyzer:
    """Holds LLM configuration and exposes fit analysis + model listing."""

    def __init__(
        self,
        *,
        provider: Provider,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 8_000,
    ) -> None:
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # -- public API ----------------------------------------------------------

    def analyze(self, cv: str, job_description: str) -> FitAnalysis:
        """Run a fit analysis and return the structured result."""
        chat_model = self._create_chat_model()
        chain = _FIT_PROMPT | chat_model.with_structured_output(FitAnalysis)

        try:
            result: FitAnalysis = chain.invoke(  # type: ignore[assignment]
                {"cv": cv, "job_description": job_description},
            )
        except Exception as exc:
            self._raise_if_model_not_found(exc)
            raise

        return result

    def list_models(self) -> list[str]:
        """Fetch available model names from the configured provider."""
        match self.provider:
            case Provider.GEMINI:
                return list_gemini_models(self.api_key)
            case Provider.OPENAI:
                return list_openai_models(self.api_key)
            case _:  # pragma: no cover
                raise ValueError(f"Unsupported provider: {self.provider}")

    # -- internals -----------------------------------------------------------

    def _create_chat_model(self) -> BaseChatModel:
        """Instantiate the appropriate LangChain chat model."""
        match self.provider:
            case Provider.OPENAI:
                return ChatOpenAI(
                    api_key=SecretStr(self.api_key),
                    model=self.model,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens,
                )
            case Provider.GEMINI:
                return ChatGoogleGenerativeAI(
                    api_key=SecretStr(self.api_key),
                    model=self.model,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
            case _:  # pragma: no cover
                raise ValueError(f"Unsupported provider: {self.provider}")

    def _raise_if_model_not_found(self, error: Exception) -> None:
        """Re-raise as ``ModelNotFoundError`` when the provider reports a missing model."""
        msg = str(error).lower()
        markers = ("not_found", "not found", "does not exist")
        if not any(m in msg for m in markers):
            return

        try:
            available = self.list_models()
        except Exception:
            available = []

        raise ModelNotFoundError(self.model, self.provider, available) from error
