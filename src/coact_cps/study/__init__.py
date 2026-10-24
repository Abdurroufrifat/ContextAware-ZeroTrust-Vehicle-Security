"""Human-study records and sequence preparation.

Do not collect data from people until the relevant ethics/IRB approval is active.
"""

from .records import CONDITIONS, StudyTrial, append_trial, new_participant_code

__all__ = ["CONDITIONS", "StudyTrial", "append_trial", "new_participant_code"]
