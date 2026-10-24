from head_reference_core import *
if __name__=='__main__':
    if (ROOT/'results.json').exists():raise ValueError('one-shot reference diagnostic already exists')
    results,arrays=compute();write(ROOT/'results.json',results);atomic_npz(ROOT/'row_identity.npz',arrays)
    print('FROZEN_HEAD_REFERENCE_COMPARISON_COMPLETE_NO_PROMOTION')
