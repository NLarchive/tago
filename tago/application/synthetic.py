"""Ring 2 - application: synthetic case generator and independent dataset auditor agents."""

# %% cell: synthetic-agents
from tago.core.domain import AuditVerdict, CaseError
from tago.config import GENERATION_ATTEMPTS
from tago.application.packs import GENERATOR_PROMPT, AUDITOR_PROMPT

SCENARIO_CONSTRAINTS = {
    'E01': 'Refund must exceed the self-service cap: choose refund_cents 5000-900000 and a smaller self_service_limit_cents.', 'E02': 'Next step is schedule_demo.',
    'E03': 'Billed units exceed received units.', 'E04': 'Privileged access has no manager approval.',
    'E05': 'PostgreSQL pairs with transaction_integrity; SQLite with embedded_deployment.',
    'E06': 'Use the specified defective expression; maximum_n is 5..100.',
    'E07': 'Stock below target; unique vendor IDs; at least one vendor meets lead limit.',
    'E08': 'Fictional jurisdiction only; employee count meets threshold.',
    'E09': 'Login count meets threshold; distinct source and target hosts.',
    'E10': 'Unique offers; at least one free-hour offer is within budget.'}

COMMON_CONSTRAINTS = ('The reference field must be a concrete identifier containing digits (never the field name itself) and must appear in the request verbatim, as a standalone token '
                      '(no "#", quotes or extra prefix characters attached to it). Amounts are integers in cents/units within the schema limits. '
                      'Where the scenario involves amounts, quantities or limits, state each one in the request as a plain number identical to the world value.')

class SyntheticCaseAgent:
    """Generates one fictional request + world. Invalid output is recorded, never replaced by a fixture.

    v38.2: a rejected draft is regenerated with the exact validation error (same lesson as the repair loop). The Colab T4
    run lost 3 of 6 synthetic cases to first-draft slips (id mismatch, refund above schema maximum). Every attempt and its
    error is kept in `attempt_errors`; there is still no fixture fallback.
    """
    def __init__(self, backend, attempts=GENERATION_ATTEMPTS):
        self.backend, self.attempts, self.attempt_errors = backend, max(1, attempts), []
    def generate(self, pack, mode, seed):
        self.attempt_errors = []; feedback = ''
        for attempt in range(self.attempts):
            payload = {'objective': pack.objective, 'reference_field': pack.reference_key, 'mode': mode,
                       'constraints': SCENARIO_CONSTRAINTS[pack.workflow_id] + ' ' + COMMON_CONSTRAINTS}
            if feedback: payload['previous_draft_was_rejected_because'] = feedback
            try:
                result = self.backend.chat('generator', GENERATOR_PROMPT, payload, pack.case_schema, seed + attempt, .8)
                return pack.validate_synthetic(result.model_dump())
            except (CaseError, ValueError) as error:   # pydantic ValidationError is a ValueError
                feedback = f'{type(error).__name__}: {str(error)[:400]}'; self.attempt_errors.append(feedback)
                if attempt + 1 == self.attempts: raise

class DatasetAuditor:
    """Audits the generated input (not agent output). Same-model auditing is correlated, not independent."""
    def __init__(self, backend): self.backend = backend
    def audit(self, pack, case, mode, seed):
        case = pack.validate_case(case)
        verdict = self.backend.chat('auditor', AUDITOR_PROMPT, {'objective': pack.objective, 'case': case, 'mode': mode}, AuditVerdict, seed)
        data = verdict.model_dump()
        data['accepted'] = verdict.valid and verdict.request_matches_world and verdict.style_matches
        return data
