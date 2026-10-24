from aux_core import *
def main():
 if (ROOT/'results.json').exists() or (ROOT/'pilot_arrays.npz').exists():raise ValueError('pilot outputs already exist; do not overwrite')
 result,arrays=compute();write(ROOT/'results.json',result);np.savez_compressed(ROOT/'pilot_arrays.npz',**arrays)
 print('AUXILIARY IMU PILOT GENERATED',result['pilot_chunks'],'chunks;',result['all_connection_records_checked'],'all-connection records',flush=True)
 for name,s in result['pilot_channels'].items():print(name,s['decoded_records'],'records; future headers at writer',s['future_header_at_writer_records'],flush=True)
 print(result['decision'],flush=True)
if __name__=='__main__':main()
