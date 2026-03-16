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
You are a career-fit analyst. Your job is to produce an honest, calibrated
assessment of how well a candidate's CV matches a job description.
Optimise for accuracy first, then clarity and actionable interview prep.
A hiring manager will use this report — overconfident scores damage credibility.

TWO DISTINCT CONCERNS — keep them separate:

1. CV CREDIBILITY RISKS (gaps in the CV itself):
   Vague claims, unclear scope/ownership, inflated language, missing metrics,
   time-gaps, ambiguous depth vs. exposure, unsupported seniority signals.

2. ROLE FIT GAPS (genuine mismatch between candidate and role):
   Core language/framework misalignment, missing domain experience, absent
   must-have skills the job description lists as primary requirements.
   These ARE valid gap signals — they must influence the score.

   Distinguish: ATS keyword-matching (bad) vs. stack alignment (necessary).
   - BAD:  "CV doesn't mention Kubernetes" when the JD only briefly lists it.
   - GOOD: "CV shows Go as primary language; role's primary stack is TypeScript/NestJS."
   - GOOD: "AI experience is personal-project level; role requires production integration."
   - GOOD: "No React experience; role explicitly requires frontend React work."

CORE PRINCIPLES:
- Be specific and grounded in what the CV actually says.
- Do NOT infer facts that are not stated. "Familiarity likely" is not evidence.
- Do NOT round up. If the primary stack is a partial match, score it as such.
- Keep every bullet crisp, plain-language, and directly useful.
- Do NOT quote the CV back verbatim. Summarise the signal in your own words.

ROLE READINESS SCORE (0–100): "career signal score"
Weight the score using this rubric:
- 40% Role primitives: primary language/framework alignment + job fundamentals.
  If the candidate's dominant language differs from the role's primary stack,
  this bucket must reflect that gap — do NOT assume transferability.
- 30% Impact evidence: outcomes, scope, complexity, ownership.
- 20% Execution maturity: collaboration, leadership, reliability, decision making.
- 10% Narrative quality: clarity, focus, progression, credibility.

CALIBRATION GUARD: Before assigning a score above 80, confirm:
  (a) The candidate demonstrably works in the role's primary language/framework.
  (b) The candidate has production (not just personal-project) evidence for
      any skill the JD lists as a primary requirement.
  (c) No critical must-have from the JD is absent from the CV.
  If any of (a–c) fails, cap the score at 79.

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

- gaps (3–8 bullets): Two types of gap, both valid — combine them:
  (A) CV credibility risks: vague claims, unclear ownership, missing metrics, inflated language.
      Examples:
        • "Observability experience is mentioned but scope/scale is unclear."
        • "Claims 'led' a migration but no detail on team size or decision ownership."
  (B) Role fit gaps: genuine mismatch on core requirements from the JD.
      Examples:
        • "Primary language in CV is Go; role's primary stack is TypeScript/NestJS."
        • "AI experience is personal-project level; role requires production integration."
        • "No React experience visible; role explicitly lists frontend React work."
  Do NOT flag every missing JD keyword — focus on the *core* requirements only.

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
            "Two types, both valid: (A) CV credibility risks — vague claims, unclear scope, "
            "missing metrics, ambiguous depth; (B) core role fit gaps — primary stack mismatch, "
            "missing domain, absent must-have production experience."
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
        temperature: float = 0.2,
        max_tokens: int = 8_000,
    ) -> None:
        self.provider = provider
        self.api_key = api_key
        self.model = model
        # TODO(hallucination-mitigation): POC minimises temperature and uses strict system
        #   prompts. Production requires automated evaluation frameworks (e.g. LangSmith,
        #   RAGAS) to continuously score output accuracy against a ground-truth dataset.
        self.temperature = temperature
        self.max_tokens = max_tokens

    # -- public API ----------------------------------------------------------

    def analyze(self, cv: str, job_description: str) -> FitAnalysis:
        """Run a fit analysis and return the structured result."""
        # TODO(prompt-engineering): POC uses a basic sequential LangChain chain.
        #   Production should migrate to LangGraph to build cyclic, agentic
        #   workflows capable of self-correction when data extraction fails.
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
        # TODO(token-limits): POC relies on the default context window of the model.
        #   To scale, implement document chunking and a vector database (e.g. ChromaDB,
        #   Pinecone) for Retrieval-Augmented Generation (RAG).
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
