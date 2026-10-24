from guard_core import corroborate as select_base_guard

def corroborate(selected,base):
 value,reason=select_base_guard(selected,base)
 if reason=="base_fallback":return None,"unresolved_source_conflict"
 return value,reason
