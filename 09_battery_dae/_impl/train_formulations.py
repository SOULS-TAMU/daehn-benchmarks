"""Train matched DAE formulation comparisons without inequality projection."""

from pathlib import Path as _CasePath
import sys as _case_sys
_CASE_DIR = _CasePath(__file__).resolve().parents[1]
_case_sys.path.insert(0, str(_CASE_DIR))
_case_sys.path.insert(0, str(_CASE_DIR / "_impl"))
from train_suite import main
if __name__=='__main__': main(('daehn','fixed_capacity','detached_ocv','two_rc','dynamic_hysteresis','two_rc_hysteresis'))
