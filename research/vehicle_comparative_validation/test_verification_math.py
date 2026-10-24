import pytest
from verification_math import midpoint_integer_valid
def test_independent_midpoint_check_handles_half_mm_float_ties_only():
 assert midpoint_integer_valid(7372,7.1,7.645)
 assert midpoint_integer_valid(7373,7.100000000000001,7.645)
 assert not midpoint_integer_valid(7374,7.1,7.645)
 assert not midpoint_integer_valid(7373,7.1,7.644)
