from frame_reference_core import *
if __name__=='__main__':
 if (ROOT/'results.json').exists() or (ROOT/'reference_pairs.npz').exists():raise ValueError('audit already generated; no overwrite')
 result,arrays=compute();write(ROOT/'results.json',result);np.savez_compressed(ROOT/'reference_pairs.npz',**arrays);print(result['decision']);print({s:v['comparison_rows'] for s,v in result['populations'].items()});print('Raw IMU messages',sum(v['records'] for v in result['raw_imu_orientation'].values()))
